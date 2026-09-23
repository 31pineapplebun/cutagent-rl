"""Run handoff-only and compact recovery on the frozen M4B.5 failure set."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

from cutagent_evaluation.m4b5_recovery import (
    M4B5_DATASET_VERSION,
    M4B5_EVALUATOR_VERSION,
    DeterministicFailureInjectingRegistry,
    FailureInjectionTrigger,
    M4B5FailureType,
    M4B5RecoveryEvaluation,
    M4B5ValidationCase,
    M4B5Variant,
    build_m4b5_recovery_set,
    evaluate_m4b5_trajectory,
    finalize_trigger,
    summarize_m4b5_evaluations,
    trajectory_has_private_injection_data,
)

from cutagent.agent.m4b_runtime import M4BAgentRuntime
from cutagent.agent.m4b_state_reducer import M4BStateReducer
from cutagent.agent.m4b_trajectory import M4BTrajectoryStore
from cutagent.models.qwen_policy_m4b import Qwen3VLPolicyBackendM4B
from cutagent.retrieval.adaptive import AdaptiveHybridRetriever
from cutagent.retrieval.encoders import BGETextEncoder, Siglip2VisualEncoder
from cutagent.retrieval.fusion import WeightedRankFusion
from cutagent.retrieval.index import RetrievalIndexBuilder
from cutagent.retrieval.protocols import MultimodalRetriever
from cutagent.retrieval.retrievers import DenseTextRetriever, SparseRetriever, VisualRetriever
from cutagent.schemas.m4b_agent import M4BAgentTrajectory, M4BRuntimeConfig
from cutagent.schemas.retrieval import AdaptiveRetrievalConfig, RetrievalChannel, RetrievalConfig
from cutagent.tools.factory import create_m3a_registry
from scripts.m4b_run_experiment import (
    _artifact_path,
    _keyframes,
    _load_gold,
    _load_prepared,
    _performance,
    _write_json,
)

VARIANTS: tuple[M4B5Variant, ...] = ("handoff_only", "compact_recovery")
M4B_SOURCE_GROUPS = frozenset({"m2b-heldout-source-07", "m2b-heldout-source-08"})


def _load_or_run(
    *,
    trajectory_path: Path,
    trigger_path: Path,
    case: M4B5ValidationCase,
    registry: Any,
    policy: Qwen3VLPolicyBackendM4B,
    artifact_root: Path,
    config: M4BRuntimeConfig,
    run_id: str,
) -> tuple[M4BAgentTrajectory, FailureInjectionTrigger, bool]:
    if trajectory_path.is_file() or trigger_path.is_file():
        if not trajectory_path.is_file() or not trigger_path.is_file():
            raise RuntimeError("M4B.5 resume requires both trajectory and private trigger record")
        trajectory = M4BAgentTrajectory.model_validate_json(
            trajectory_path.read_text(encoding="utf-8")
        )
        trigger = FailureInjectionTrigger.model_validate_json(
            trigger_path.read_text(encoding="utf-8")
        )
        if trajectory.task_input != case.task_input or trajectory.runtime_config != config:
            raise RuntimeError(f"incompatible M4B.5 trajectory resume: {trajectory_path}")
        if (
            trigger.injection_id != case.gold.injection.injection_id
            or trigger.failure_type != case.gold.injection.failure_type
        ):
            raise RuntimeError(f"incompatible M4B.5 trigger resume: {trigger_path}")
        return trajectory, trigger, True
    injected = DeterministicFailureInjectingRegistry(registry, case.gold.injection)
    runtime = M4BAgentRuntime(
        registry=injected,
        policy_model=policy,
        artifact_root=artifact_root,
    )
    trajectory = runtime.run(case.task_input, config=config, run_id=run_id)
    trigger = finalize_trigger(injected.private_trigger(), trajectory)
    _write_json(trajectory_path, trajectory.model_dump(mode="json"))
    _write_json(trigger_path, trigger.model_dump(mode="json"))
    return trajectory, trigger, False


def _validation(trajectories: list[M4BAgentTrajectory]) -> dict[str, int]:
    replay_count = sum(
        M4BStateReducer.replay(item.initial_state, item.events) == item.final_state
        for item in trajectories
    )
    leak_count = sum(trajectory_has_private_injection_data(item) for item in trajectories)
    return {
        "trajectory_count": len(trajectories),
        "deterministic_event_replay_count": replay_count,
        "private_injection_or_gold_leak_count": leak_count,
    }


def _select_cases(
    cases: tuple[M4B5ValidationCase, ...],
    *,
    task_id: str | None,
    failure_type: M4B5FailureType | None,
    limit: int | None,
) -> tuple[M4B5ValidationCase, ...]:
    selected = cases
    if task_id is not None:
        selected = tuple(item for item in selected if item.task_input.task_id == task_id)
        if not selected:
            raise ValueError(f"unknown M4B.5 task: {task_id}")
    if failure_type is not None:
        selected = tuple(
            item for item in selected if item.gold.injection.failure_type == failure_type
        )
        if not selected:
            raise ValueError(f"no M4B.5 task has failure type: {failure_type}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2b-root", type=Path, default=Path("artifacts/m2b"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m4b_5"))
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument(
        "--failure-type",
        choices=tuple(M4B5FailureType.__args__),  # type: ignore[attr-defined]
        default=None,
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    started_ns = time.perf_counter_ns()
    root = args.artifact_root.resolve()
    m2b_root = args.m2b_root.resolve()
    ingestions, perceptions = _load_prepared(m2b_root)
    source_gold = _load_gold(m2b_root)
    recovery_gold = tuple(
        item for item in source_gold if item.source_group_id not in M4B_SOURCE_GROUPS
    )
    if len(recovery_gold) != 6:
        raise RuntimeError("M4B.5 expects the six M4B-disjoint development source groups")
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
        for item in recovery_gold
    }
    all_cases = build_m4b5_recovery_set(
        recovery_gold,
        source_refs,
        excluded_source_groups=M4B_SOURCE_GROUPS,
    )
    cases = _select_cases(
        all_cases,
        task_id=args.task_id,
        failure_type=cast(M4B5FailureType | None, args.failure_type),
        limit=args.limit,
    )
    _write_json(
        root / "validation_tasks_public.json",
        [item.task_input.model_dump(mode="json") for item in cases],
    )
    _write_json(
        root / "private" / "validation_gold_and_injection.json",
        [item.gold.model_dump(mode="json") for item in cases],
    )
    _write_json(
        root / "private" / "source_group_policy.json",
        {
            "m4b_validation_source_groups": sorted(M4B_SOURCE_GROUPS),
            "m4b5_recovery_source_groups": sorted({item.gold.source_group_id for item in cases}),
            "overlap_with_m4b_validation": [],
            "note": "M4B.5 uses new tasks over prior M4A development media.",
        },
    )

    experiment: dict[str, Any] = {
        "schema_version": "1.0",
        "task_count": len(cases),
        "dataset_version": M4B5_DATASET_VERSION,
        "evaluator_version": M4B5_EVALUATOR_VERSION,
        "failure_injector_version": DeterministicFailureInjectingRegistry.version,
        "split": "recovery-specific validation",
        "used_for_prompt_tuning": False,
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "dtype": "bfloat16",
        "policy_view": "structured_state",
        "seed": 20_260_823,
        "runtime_configs": {
            variant: M4BRuntimeConfig(protocol_variant=variant).model_dump(mode="json")
            for variant in VARIANTS
        },
        "retrieval_index_id": memory.manifest.index_id,
        "native_video_verification": "disabled",
        "variants": {},
    }
    selected_variants = tuple(
        item for item in VARIANTS if args.variant is None or item == args.variant
    )
    policy = Qwen3VLPolicyBackendM4B(model_cache=args.model_cache.resolve())
    store = M4BTrajectoryStore(root / "trajectory_artifacts")
    for variant in selected_variants:
        config = M4BRuntimeConfig(protocol_variant=variant)
        trajectories: list[M4BAgentTrajectory] = []
        evaluations: list[M4B5RecoveryEvaluation] = []
        triggers: list[FailureInjectionTrigger] = []
        resume_hits = 0
        for index, case in enumerate(cases, start=1):
            trajectory, trigger, resumed = _load_or_run(
                trajectory_path=root / "progress" / variant / f"{case.task_input.task_id}.json",
                trigger_path=(
                    root / "private" / "progress" / variant / f"{case.task_input.task_id}.json"
                ),
                case=case,
                registry=registry,
                policy=policy,
                artifact_root=root / "runs" / variant,
                config=config,
                run_id=f"m4b5-{variant}-{case.task_input.task_id}",
            )
            resume_hits += resumed
            trajectories.append(trajectory)
            triggers.append(trigger)
            evaluations.append(
                evaluate_m4b5_trajectory(
                    trajectory,
                    case.gold,
                    trigger,
                    variant=variant,
                )
            )
            store.write(trajectory)
            print(
                f"[{variant}] {index}/{len(cases)} {case.task_input.task_id} "
                f"failure={case.gold.injection.failure_type} triggered={trigger.triggered} "
                f"terminal={trajectory.terminal_reason} resumed={resumed}",
                flush=True,
            )
        summary = summarize_m4b5_evaluations(evaluations)
        _write_json(
            root / "private" / "triggers" / f"{variant}.json",
            [item.model_dump(mode="json") for item in triggers],
        )
        _write_json(
            root / "evaluations" / f"{variant}.json",
            [item.model_dump(mode="json") for item in evaluations],
        )
        _write_json(root / "metrics" / f"{variant}.json", summary.model_dump(mode="json"))
        experiment["variants"][variant] = {
            "metrics": summary.model_dump(mode="json"),
            "performance": _performance(trajectories),
            "terminal_reason_counts": dict(Counter(item.terminal_reason for item in trajectories)),
            "resume_hits": resume_hits,
            "validation": _validation(trajectories),
            "policy_backend_version": policy.backend_version,
            "model_load_time_ms": policy.load_time_ms,
        }
    experiment["experiment_wall_time_ms"] = (time.perf_counter_ns() - started_ns) // 1_000_000
    experiment["process_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    experiment["process_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    _write_json(root / "experiment_summary.json", experiment)
    print(json.dumps(experiment, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
