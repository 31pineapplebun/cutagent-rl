"""Run the frozen M4A and two M4B variants on one held-out validation set."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from cutagent_evaluation.m2b_dataset import HeldOutVideoGold
from cutagent_evaluation.m4b_agent import (
    M4BRecoveryCaseRecord,
    M4BTrajectoryEvaluation,
    M4BValidationCase,
    analyze_m4b_recovery_case,
    build_m4b_validation_set,
    evaluate_m4b_trajectory,
    summarize_m4b_evaluations,
)

from cutagent.agent.m4b_runtime import M4BAgentRuntime
from cutagent.agent.m4b_state_reducer import M4BStateReducer
from cutagent.agent.m4b_trajectory import M4BTrajectoryStore
from cutagent.agent.runtime import AgentRuntime
from cutagent.agent.state_reducer import StateReducer
from cutagent.agent.trajectory import TrajectoryStore
from cutagent.models.qwen_policy import Qwen3VLPolicyBackend
from cutagent.models.qwen_policy_m4b import Qwen3VLPolicyBackendM4B
from cutagent.retrieval.adaptive import AdaptiveHybridRetriever
from cutagent.retrieval.encoders import BGETextEncoder, Siglip2VisualEncoder
from cutagent.retrieval.fusion import WeightedRankFusion
from cutagent.retrieval.index import KeyframeAsset, RetrievalIndexBuilder
from cutagent.retrieval.protocols import MultimodalRetriever
from cutagent.retrieval.retrievers import DenseTextRetriever, SparseRetriever, VisualRetriever
from cutagent.schemas.agent import AgentRuntimeConfig, AgentTrajectory
from cutagent.schemas.m4b_agent import M4BAgentTrajectory, M4BRuntimeConfig
from cutagent.schemas.media import IngestionResult
from cutagent.schemas.perception import PerceptionResult
from cutagent.schemas.retrieval import AdaptiveRetrievalConfig, RetrievalChannel, RetrievalConfig
from cutagent.tools.factory import create_m3a_registry

VARIANTS = ("frozen_m4a", "handoff_only", "compact_recovery")
M4A_SOURCE_GROUPS = frozenset(f"m2b-heldout-source-{index:02d}" for index in range(1, 7))


def _artifact_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("M4B experiment requires local file artifacts")
    value = unquote(parsed.path)
    if len(value) >= 3 and value[0] == "/" and value[2] == ":":
        value = value[1:]
    return Path(value).resolve(strict=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_prepared(
    root: Path,
) -> tuple[tuple[IngestionResult, ...], tuple[PerceptionResult, ...]]:
    ingestions = tuple(
        IngestionResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((root / "ingestion").glob("*.json"))
    )
    perceptions = tuple(
        PerceptionResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((root / "perception").glob("*.json"))
    )
    if not ingestions or len(ingestions) != len(perceptions):
        raise RuntimeError("M2B public ingestion/perception artifacts are incomplete")
    return ingestions, perceptions


def _load_gold(root: Path) -> tuple[HeldOutVideoGold, ...]:
    payload = json.loads((root / "heldout_source_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("M2B held-out source manifest must be a list")
    return tuple(HeldOutVideoGold.model_validate(item) for item in payload)


def _keyframes(ingestions: tuple[IngestionResult, ...]) -> tuple[KeyframeAsset, ...]:
    values: dict[str, KeyframeAsset] = {}
    for ingestion in ingestions:
        for keyframe in ingestion.keyframes:
            values[keyframe.artifact.artifact_id] = KeyframeAsset(
                artifact=keyframe.artifact,
                path=_artifact_path(keyframe.artifact.uri),
            )
    return tuple(values[key] for key in sorted(values))


def _load_or_run_m4a(
    *,
    path: Path,
    runtime: AgentRuntime,
    case: M4BValidationCase,
    config: AgentRuntimeConfig,
    run_id: str,
) -> tuple[AgentTrajectory, bool]:
    if path.is_file():
        trajectory = AgentTrajectory.model_validate_json(path.read_text(encoding="utf-8"))
        if trajectory.task_input == case.task_input and trajectory.runtime_config == config:
            return trajectory, True
        raise RuntimeError(f"incompatible frozen M4A resume file: {path}")
    trajectory = runtime.run(case.task_input, config=config, run_id=run_id)
    _write_json(path, trajectory.model_dump(mode="json"))
    return trajectory, False


def _load_or_run_m4b(
    *,
    path: Path,
    runtime: M4BAgentRuntime,
    case: M4BValidationCase,
    config: M4BRuntimeConfig,
    run_id: str,
) -> tuple[M4BAgentTrajectory, bool]:
    if path.is_file():
        trajectory = M4BAgentTrajectory.model_validate_json(path.read_text(encoding="utf-8"))
        if trajectory.task_input == case.task_input and trajectory.runtime_config == config:
            return trajectory, True
        raise RuntimeError(f"incompatible M4B resume file: {path}")
    trajectory = runtime.run(case.task_input, config=config, run_id=run_id)
    _write_json(path, trajectory.model_dump(mode="json"))
    return trajectory, False


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))]


def _performance(
    trajectories: list[AgentTrajectory] | list[M4BAgentTrajectory],
) -> dict[str, Any]:
    steps = [step for trajectory in trajectories for step in trajectory.policy_steps]
    failures = [failure for trajectory in trajectories for failure in trajectory.policy_failures]
    records = [record for trajectory in trajectories for record in trajectory.tool_records]
    total_inferences = len(steps) + len(failures)
    return {
        "task_latency_p50_ms": _percentile([item.latency.total_ms for item in trajectories], 0.50),
        "task_latency_p95_ms": _percentile([item.latency.total_ms for item in trajectories], 0.95),
        "mean_policy_latency_per_inference_ms": (
            sum(step.stats.latency_ms for step in steps)
            + sum(failure.latency_ms for failure in failures)
        )
        / max(total_inferences, 1),
        "mean_input_tokens_per_valid_step": (
            sum(step.stats.input_tokens for step in steps) / max(len(steps), 1)
        ),
        "mean_output_tokens_per_valid_step": (
            sum(step.stats.output_tokens for step in steps) / max(len(steps), 1)
        ),
        "policy_latency_total_ms": sum(item.latency.policy_ms for item in trajectories),
        "recovery_policy_latency_total_ms": sum(
            getattr(item.latency, "recovery_policy_ms", 0) for item in trajectories
        ),
        "retrieval_latency_total_ms": sum(item.latency.retrieval_ms for item in trajectories),
        "editing_tool_latency_total_ms": sum(item.latency.editing_tool_ms for item in trajectories),
        "verification_latency_total_ms": sum(item.latency.verification_ms for item in trajectories),
        "qwen_peak_allocated_bytes": max(
            (step.stats.peak_allocated_bytes or 0 for step in steps), default=0
        ),
        "qwen_peak_reserved_bytes": max(
            (step.stats.peak_reserved_bytes or 0 for step in steps), default=0
        ),
        "tool_cache_hits": sum(record.trace.cache_hit for record in records),
        "tool_cache_misses": sum(not record.trace.cache_hit for record in records),
    }


def _validate_trajectories(
    variant: str,
    trajectories: list[AgentTrajectory] | list[M4BAgentTrajectory],
) -> dict[str, object]:
    replay_count = 0
    leak_count = 0
    for trajectory in trajectories:
        if isinstance(trajectory, M4BAgentTrajectory):
            replay_identical = (
                M4BStateReducer.replay(trajectory.initial_state, trajectory.events)
                == trajectory.final_state
            )
        else:
            replay_identical = (
                StateReducer.replay(trajectory.initial_state, trajectory.events)
                == trajectory.final_state
            )
        replay_count += replay_identical
        serialized = trajectory.model_dump_json().casefold()
        leak_count += any(
            token in serialized
            for token in ("source_group_id", '"split"', "gold_id", "benchmarkgold")
        )
    return {
        "variant": variant,
        "trajectory_count": len(trajectories),
        "deterministic_event_replay_count": replay_count,
        "private_label_leak_count": leak_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2b-root", type=Path, default=Path("artifacts/m2b"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m4b"))
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    root = args.artifact_root.resolve()
    m2b_root = args.m2b_root.resolve()
    ingestions, perceptions = _load_prepared(m2b_root)
    source_gold = _load_gold(m2b_root)
    validation_gold = tuple(
        item for item in source_gold if item.source_group_id not in M4A_SOURCE_GROUPS
    )
    if len(validation_gold) != 2:
        raise RuntimeError("M4B expects exactly two M4A-disjoint held-out source groups")
    ingestion_by_hash = {item.video.source.sha256: item for item in ingestions}
    if any(item.source_sha256 not in ingestion_by_hash for item in source_gold):
        raise RuntimeError("private source manifest and public ingestion corpus differ")

    torch: Any = importlib.import_module("torch")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    retrieval_config = RetrievalConfig()
    text_encoder = BGETextEncoder(
        model_cache=args.model_cache.resolve(),
        batch_size=retrieval_config.text_batch_size,
        maximum_tokens=retrieval_config.text_encoder_max_tokens,
    )
    visual_encoder = Siglip2VisualEncoder(
        model_cache=args.model_cache.resolve(),
        batch_size=retrieval_config.visual_batch_size,
    )
    memory = RetrievalIndexBuilder(m2b_root / "index").build(
        world_states=tuple(item.world_state for item in perceptions),
        keyframe_assets=_keyframes(ingestions),
        config=retrieval_config,
        text_encoder=text_encoder,
        visual_encoder=visual_encoder,
    )
    components: dict[RetrievalChannel, MultimodalRetriever] = {
        "bm25_transcript": SparseRetriever("transcript"),
        "bm25_ocr": SparseRetriever("ocr"),
        "bm25_structured": SparseRetriever("structured_semantic"),
        "bm25_combined": SparseRetriever("combined"),
        "dense_text": DenseTextRetriever(text_encoder),
        "visual": VisualRetriever(visual_encoder),
    }
    retriever = AdaptiveHybridRetriever(
        config=AdaptiveRetrievalConfig(native_video_enabled=False),
        fusion=WeightedRankFusion(components),
        strategy="query_aware",
        evidence_reranking=True,
        native_video=False,
    )
    registry = create_m3a_registry(
        root=root / "tool_runtime",
        retriever=retriever,
        retrieval_memory=memory,
        ffmpeg_executable=args.ffmpeg,
        ffprobe_executable=args.ffprobe,
    )
    source_refs = {
        item.source_group_id: registry.artifact_store.import_file(
            _artifact_path(ingestion_by_hash[item.source_sha256].video.source.uri),
            media_type="video/mp4",
            artifact_id=ingestion_by_hash[item.source_sha256].video.video_id,
        )
        for item in validation_gold
    }
    cases = build_m4b_validation_set(
        validation_gold,
        source_refs,
        excluded_source_groups=set(M4A_SOURCE_GROUPS),
    )
    if args.task_id is not None:
        cases = tuple(item for item in cases if item.task_input.task_id == args.task_id)
        if not cases:
            raise ValueError(f"unknown M4B validation task: {args.task_id}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        cases = cases[: args.limit]
    _write_json(
        root / "validation_tasks_public.json",
        [item.task_input.model_dump(mode="json") for item in cases],
    )
    _write_json(
        root / "private" / "validation_gold.json",
        [item.gold.model_dump(mode="json") for item in cases],
    )
    _write_json(
        root / "private" / "source_group_policy.json",
        {
            "m4a_development_source_groups": sorted(M4A_SOURCE_GROUPS),
            "m4b_validation_source_groups": sorted(
                item.source_group_id for item in validation_gold
            ),
            "overlap": [],
        },
    )

    experiment: dict[str, Any] = {
        "schema_version": "1.0",
        "task_count": len(cases),
        "split": "held-out validation",
        "used_for_protocol_tuning": False,
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "dtype": "bfloat16",
        "policy_view": "structured_state",
        "seed": 20_260_823,
        "retrieval_index_id": memory.manifest.index_id,
        "native_video_verification": "disabled",
        "variants": {},
    }
    selected = tuple(item for item in VARIANTS if args.variant in {None, item})
    m4a_store = TrajectoryStore(root / "trajectory_artifacts" / "frozen_m4a")
    m4b_store = M4BTrajectoryStore(root / "trajectory_artifacts" / "m4b")

    if "frozen_m4a" in selected:
        frozen_policy = Qwen3VLPolicyBackend(model_cache=args.model_cache.resolve())
        frozen_runtime = AgentRuntime(
            registry=registry,
            policy_model=frozen_policy,
            artifact_root=root / "runs" / "frozen_m4a",
        )
        frozen_config = AgentRuntimeConfig(
            baseline="hierarchical",
            policy_view_mode="structured_state",
            max_steps=12,
            max_tool_calls=10,
            max_search_calls=3,
            max_edit_calls=8,
            max_repeated_identical_actions=2,
            max_structured_output_repairs=2,
            max_wall_time_ms=600_000,
            max_model_tokens=64_000,
            maximum_visual_evidence=4,
            policy_maximum_new_tokens=768,
        )
        frozen_trajectories: list[AgentTrajectory] = []
        frozen_evaluations: list[M4BTrajectoryEvaluation] = []
        resume_hits = 0
        for index, case in enumerate(cases, start=1):
            frozen_trajectory, resumed = _load_or_run_m4a(
                path=(root / "progress" / "frozen_m4a" / f"{case.task_input.task_id}.json"),
                runtime=frozen_runtime,
                case=case,
                config=frozen_config,
                run_id=f"frozen-m4a-{case.task_input.task_id}",
            )
            resume_hits += resumed
            frozen_trajectories.append(frozen_trajectory)
            frozen_evaluations.append(
                evaluate_m4b_trajectory(frozen_trajectory, case.gold, variant="frozen_m4a")
            )
            m4a_store.write(frozen_trajectory)
            print(
                f"[frozen_m4a] {index}/{len(cases)} {case.task_input.task_id} "
                f"terminal={frozen_trajectory.terminal_reason} resumed={resumed}",
                flush=True,
            )
        _write_json(
            root / "evaluations" / "frozen_m4a.json",
            [item.model_dump(mode="json") for item in frozen_evaluations],
        )
        frozen_recovery_cases: list[M4BRecoveryCaseRecord] = []
        for frozen_trajectory, case, evaluation in zip(
            frozen_trajectories, cases, frozen_evaluations, strict=True
        ):
            recovery_case = analyze_m4b_recovery_case(frozen_trajectory, case.gold, evaluation)
            if recovery_case is not None:
                frozen_recovery_cases.append(recovery_case)
        _write_json(
            root / "recovery_cases" / "frozen_m4a.json",
            [item.model_dump(mode="json") for item in frozen_recovery_cases],
        )
        experiment["variants"]["frozen_m4a"] = {
            "metrics": summarize_m4b_evaluations(frozen_evaluations).model_dump(mode="json"),
            "performance": _performance(frozen_trajectories),
            "terminal_reason_counts": dict(
                Counter(item.terminal_reason for item in frozen_trajectories)
            ),
            "resume_hits": resume_hits,
            "validation": _validate_trajectories("frozen_m4a", frozen_trajectories),
            "policy_backend_version": frozen_policy.backend_version,
            "model_load_time_ms": frozen_policy.load_time_ms,
        }
        _write_json(
            root / "metrics" / "frozen_m4a.json",
            experiment["variants"]["frozen_m4a"],
        )
        del frozen_runtime, frozen_policy
        gc.collect()
        torch.cuda.empty_cache()

    m4b_selected = tuple(item for item in selected if item != "frozen_m4a")
    if m4b_selected:
        m4b_policy = Qwen3VLPolicyBackendM4B(model_cache=args.model_cache.resolve())
        m4b_runtime = M4BAgentRuntime(
            registry=registry,
            policy_model=m4b_policy,
            artifact_root=root / "runs" / "m4b",
        )
        for variant in m4b_selected:
            config = M4BRuntimeConfig(protocol_variant=variant)  # type: ignore[arg-type]
            trajectories: list[M4BAgentTrajectory] = []
            evaluations: list[M4BTrajectoryEvaluation] = []
            resume_hits = 0
            for index, case in enumerate(cases, start=1):
                m4b_trajectory, resumed = _load_or_run_m4b(
                    path=root / "progress" / variant / f"{case.task_input.task_id}.json",
                    runtime=m4b_runtime,
                    case=case,
                    config=config,
                    run_id=f"{variant}-{case.task_input.task_id}",
                )
                resume_hits += resumed
                trajectories.append(m4b_trajectory)
                evaluations.append(
                    evaluate_m4b_trajectory(m4b_trajectory, case.gold, variant=variant)
                )
                m4b_store.write(m4b_trajectory)
                print(
                    f"[{variant}] {index}/{len(cases)} {case.task_input.task_id} "
                    f"terminal={m4b_trajectory.terminal_reason} resumed={resumed}",
                    flush=True,
                )
            _write_json(
                root / "evaluations" / f"{variant}.json",
                [item.model_dump(mode="json") for item in evaluations],
            )
            recovery_cases: list[M4BRecoveryCaseRecord] = []
            for current_m4b, case, evaluation in zip(trajectories, cases, evaluations, strict=True):
                recovery_case = analyze_m4b_recovery_case(current_m4b, case.gold, evaluation)
                if recovery_case is not None:
                    recovery_cases.append(recovery_case)
            _write_json(
                root / "recovery_cases" / f"{variant}.json",
                [item.model_dump(mode="json") for item in recovery_cases],
            )
            experiment["variants"][variant] = {
                "metrics": summarize_m4b_evaluations(evaluations).model_dump(mode="json"),
                "performance": _performance(trajectories),
                "terminal_reason_counts": dict(
                    Counter(item.terminal_reason for item in trajectories)
                ),
                "resume_hits": resume_hits,
                "validation": _validate_trajectories(variant, trajectories),
                "policy_backend_version": m4b_policy.backend_version,
                "model_load_time_ms": m4b_policy.load_time_ms,
            }
            _write_json(root / "metrics" / f"{variant}.json", experiment["variants"][variant])

    experiment["experiment_wall_time_ms"] = (time.perf_counter_ns() - started) // 1_000_000
    experiment["process_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    experiment["process_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    _write_json(root / "experiment_summary.json", experiment)
    print(json.dumps(experiment, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
