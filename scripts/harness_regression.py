"""Freeze/run one development regression, isolated from protected evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cutagent_evaluation.harness_regression import (
    DATASET_VERSION,
    ILLEGAL_ERRORS,
    SCORER_VERSION,
    SOURCE_GROUP,
    HarnessCase,
    development_cases,
    score_run,
)
from cutagent_evaluation.m4b5_recovery import (
    DeterministicFailureInjectingRegistry,
    FailureInjectionConfig,
    finalize_trigger,
)

from cutagent.agent.harness import HarnessOutcomeBuilder, create_harness_registry, harness_config
from cutagent.agent.harness_retrieval import local_artifact_path, prepared_search_tool
from cutagent.agent.m4b_runtime import M4BAgentRuntime
from cutagent.core.artifacts import ArtifactRef
from cutagent.models.harness_policy import HarnessPolicyBackend
from cutagent.retrieval.adaptive import AdaptiveHybridRetriever
from cutagent.retrieval.retrievers import DenseTextRetriever, VisualRetriever
from cutagent.schemas.agent import PolicyModelSpec
from cutagent.schemas.m4b_agent import M4BAgentTrajectory, M4BProtocolVariant
from cutagent.schemas.media import IngestionResult
from cutagent.schemas.perception import PerceptionResult
from cutagent.schemas.task_input import DurationConstraint, TaskInput
from cutagent.tools.artifacts import ArtifactStore

VARIANTS: tuple[M4BProtocolVariant, ...] = ("handoff_only", "compact_recovery")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def source_files() -> list[Path]:
    return sorted(
        {
            *Path("src/cutagent").rglob("*.py"),
            *Path("evaluation/src/cutagent_evaluation").rglob("*.py"),
            Path("scripts/harness_regression.py"),
        }
    )


def freeze(root: Path, m2b: Path) -> None:
    if root.exists():
        raise ValueError("freeze requires a new directory; never overwrite an experiment")
    ingestion_path = m2b / "ingestion" / f"{SOURCE_GROUP}.json"
    perception_path = m2b / "perception" / f"{SOURCE_GROUP}.json"
    ingestion = IngestionResult.model_validate_json(ingestion_path.read_text(encoding="utf-8"))
    annotations = json.loads((m2b / "heldout_source_manifest.json").read_text(encoding="utf-8"))
    annotation = next(item for item in annotations if item["source_group_id"] == SOURCE_GROUP)
    if annotation["source_sha256"] != ingestion.video.source.sha256:
        raise ValueError("development source annotation hash mismatch")
    for index, scene in enumerate(annotation["scenes"]):
        if (
            scene["ocr_text"] != f"M2B01 S{index + 1}"
            or scene["nominal_time_range"]["start_ms"] != index * 3000
            or scene["nominal_time_range"]["end_ms"] != (index + 1) * 3000
        ):
            raise ValueError("predeclared marker scenes differ from source annotations")
    if len(annotation["scenes"]) != 4 or ingestion.video.duration_ms != 12000:
        raise ValueError("unexpected development source duration/scenes")
    source = local_artifact_path(ingestion.video.source.uri)
    actual = ArtifactRef.from_path(
        source, artifact_id=ingestion.video.video_id, media_type="video/mp4"
    )
    if actual.sha256 != ingestion.video.source.sha256:
        raise ValueError("development source bytes changed")
    manifest = {
        "dataset_version": DATASET_VERSION,
        "scorer_version": SCORER_VERSION,
        "frozen_at": datetime.now(UTC).isoformat(),
        "scope": "development regression; one reused generated M2B source; not generalization",
        "source_group_id": SOURCE_GROUP,
        "model": PolicyModelSpec().model_dump(mode="json"),
        "source": actual.model_dump(mode="json"),
        "ingestion_path": str(ingestion_path.resolve()),
        "perception_path": str(perception_path.resolve()),
        "ingestion_sha256": digest(ingestion_path),
        "perception_sha256": digest(perception_path),
        "configs": {v: harness_config(v).model_dump(mode="json") for v in VARIANTS},
        "cases": [c.model_dump(mode="json") for c in development_cases()],
        "injection": "six one-shot first-trim pre-execution timeouts; safe private wrapper",
        "scoring": "SUCCESS + authentic output + full decode + duration +/-250ms + original-source "
        "trim endpoints +/-125ms; semantic cases require preceding nonempty retrieval",
        "illegal_errors": sorted(ILLEGAL_ERRORS),
        "denominators": "registry attempts; unparsed policy attempts reported separately",
        "cache_policy": "fresh tool/query caches per trajectory; shared read-only index",
        "code_sha256": {p.as_posix(): digest(p) for p in source_files()},
    }
    write_json(root / "private_manifest.json", manifest)
    write_json(root / "freeze.json", {"manifest_sha256": digest(root / "private_manifest.json")})


def summarize(root: Path) -> dict[str, Any]:
    rows = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("runs/*/*/score.json"))
    ]
    summary: dict[str, Any] = {"trajectory_count": len(rows), "variants": {}, "rows": rows}
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        groups = {
            k: [r for r in selected if r["category"] == k]
            for k in ("normal", "injected", "impossible")
        }
        summary["variants"][variant] = {
            "normal_edit_success": sum(r["editing_success"] for r in groups["normal"]),
            "normal_total": len(groups["normal"]),
            "injected_edit_success": sum(r["editing_success"] for r in groups["injected"]),
            "injected_total": len(groups["injected"]),
            "injection_triggered": sum(r["injection_triggered"] for r in groups["injected"]),
            "injection_recovery_success": sum(
                r["editing_success"] and r["injection_triggered"] for r in groups["injected"]
            ),
            "correct_refusal": sum(r["correct_refusal"] for r in groups["impossible"]),
            "impossible_total": len(groups["impossible"]),
            "illegal_tool_calls": sum(r["illegal_tool_calls"] for r in selected),
            "tool_call_attempts": sum(r["tool_call_attempts"] for r in selected),
            "all_replay_identical": all(r["state_replay_identical"] for r in selected),
            "private_context_leaks": sum(r["private_context_leak"] for r in selected),
        }
    write_json(root / "summary.json", summary)
    return summary


def run(root: Path, model_cache: Path) -> None:
    frozen = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    if frozen["manifest_sha256"] != digest(root / "private_manifest.json"):
        raise ValueError("frozen manifest changed")
    manifest = json.loads((root / "private_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["code_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"frozen code changed: {name}")
    for key in ("ingestion", "perception"):
        if digest(Path(manifest[f"{key}_path"])) != manifest[f"{key}_sha256"]:
            raise ValueError(f"frozen public {key} changed")
    if (root / "run_started.json").exists():
        raise ValueError("experiment already started; inspect evidence, never silently rerun")
    ingestion = IngestionResult.model_validate_json(Path(manifest["ingestion_path"]).read_text())
    perception = PerceptionResult.model_validate_json(Path(manifest["perception_path"]).read_text())
    source = local_artifact_path(ingestion.video.source.uri)
    if digest(source) != manifest["source"]["sha256"]:
        raise ValueError("source bytes differ from freeze")
    write_json(
        root / "run_started.json",
        {"started_at": datetime.now(UTC).isoformat(), "manifest_sha256": frozen["manifest_sha256"]},
    )
    started = time.perf_counter()
    search = prepared_search_tool(
        ingestion=ingestion,
        perception=perception,
        index_root=root / "index",
        model_cache=model_cache,
    )
    policy = HarnessPolicyBackend(model_cache=model_cache)
    policy._load()
    write_json(
        root / "startup.json",
        {
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "backend": policy.backend_version,
            "retrieval_index": search.memory.manifest.model_dump(mode="json"),
        },
    )
    for case in (HarnessCase.model_validate(c) for c in manifest["cases"]):
        for variant in VARIANTS:
            assert isinstance(search.retriever, AdaptiveHybridRetriever)
            for name, component in tuple(search.retriever.fusion.components.items()):
                if isinstance(component, DenseTextRetriever):
                    search.retriever.fusion.components[name] = DenseTextRetriever(component.encoder)
                elif isinstance(component, VisualRetriever):
                    search.retriever.fusion.components[name] = VisualRetriever(component.encoder)
            config = harness_config(variant)
            if config.model_dump(mode="json") != manifest["configs"][variant]:
                raise ValueError("runtime config differs from freeze")
            directory = root / "runs" / variant / case.task_id
            registry = create_harness_registry(directory / "tools")
            registry.register(search)
            reference = registry.artifact_store.import_file(
                source, media_type="video/mp4", artifact_id=ingestion.video.video_id
            )
            injected = None
            if case.category == "injected":
                injected = DeterministicFailureInjectingRegistry(
                    registry,
                    FailureInjectionConfig(
                        injection_id=f"private-timeout-{case.task_id}",
                        task_id=case.task_id,
                        failure_type="tool_timeout",
                        trigger_mode="pre_execute_failure",
                        trigger_tool_names=("trim_video",),
                        expected_recovery_operations=("retry_current_node",),
                    ),
                )
            task = TaskInput(
                task_id=case.task_id,
                video_ref=reference,
                instruction=case.instruction,
                user_constraints=(
                    DurationConstraint(
                        min_ms=case.duration_ms - 100, max_ms=case.duration_ms + 100
                    ),
                ),
            )
            trajectory = M4BAgentRuntime(
                registry=injected or registry,
                policy_model=policy,
                artifact_root=directory / "agent",
                outcome_builder=HarnessOutcomeBuilder(),
            ).run(task, config=config, run_id=f"{variant}-{case.task_id}")
            write_json(directory / "trajectory.json", trajectory.model_dump(mode="json"))
            score = score_run(case, trajectory, registry.artifact_store, directory)
            trigger = finalize_trigger(injected.private_trigger(), trajectory) if injected else None
            if trigger:
                write_json(directory / "private_trigger.json", trigger.model_dump(mode="json"))
            contexts = json.dumps(
                [s.context.payload for s in trajectory.policy_steps]
                + [f.context.payload for f in trajectory.policy_failures]
            )
            score.update(
                variant=variant,
                injection_triggered=bool(trigger and trigger.triggered),
                private_context_leak=any(
                    f'"{key}"' in contexts
                    for key in ("expected_interval", "injection_id", "source_group_id", "split")
                ),
                trajectory_sha256=digest(directory / "trajectory.json"),
            )
            if trajectory.final_output_artifact is not None:
                _, output = registry.artifact_store.get(
                    trajectory.final_output_artifact.artifact_id
                )
                shutil.copyfile(output, directory / "output.mp4")
            write_json(directory / "score.json", score)
            print(json.dumps(score), flush=True)
    summarize(root)
    write_json(
        root / "run_complete.json",
        {"completed_at": datetime.now(UTC).isoformat(), "trajectory_count": 40},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "summarize", "semantic-preflight"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--m2b-root", type=Path, default=Path("artifacts/m2b"))
    parser.add_argument("--model-cache", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.root.resolve(), args.m2b_root.resolve())
    elif args.phase == "run":
        if args.model_cache is None:
            parser.error("--model-cache required for run")
        run(args.root.resolve(), args.model_cache.resolve(strict=True))
    elif args.phase == "semantic-preflight":
        root = args.root.resolve()
        trajectory = M4BAgentTrajectory.model_validate_json(
            (root / "trajectory.json").read_text(encoding="utf-8")
        )
        score = score_run(
            development_cases()[9], trajectory, ArtifactStore(root / "tools" / "artifacts"), root
        )
        write_json(root / "offline_score.json", score)
        print(json.dumps(score, indent=2))
        if not score["editing_success"]:
            raise RuntimeError("semantic preflight did not pass independent scoring")
    else:
        print(json.dumps(summarize(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
