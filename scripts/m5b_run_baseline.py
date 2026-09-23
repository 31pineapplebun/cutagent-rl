"""Run the frozen M4B handoff-only Agent on one access-controlled M5B split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

from cutagent_evaluation.m4b5_recovery import (
    DeterministicFailureInjectingRegistry,
    FailureInjectionConfig,
    finalize_trigger,
)
from cutagent_evaluation.m5a_evaluator import evaluate_trajectory, summarize_evaluations
from cutagent_evaluation.m5a_schemas import FailureInjectionGold
from cutagent_evaluation.m5b_baseline import (
    M5BRunDeclaration,
    M5BRunSummary,
    assert_split_access,
    config_sha256,
    load_cases,
    trajectory_has_private_fields,
)
from cutagent_evaluation.schemas import DatasetSplit

from cutagent.agent.m4b_runtime import M4BAgentRuntime
from cutagent.agent.m4b_state_reducer import M4BStateReducer
from cutagent.agent.m4b_trajectory import M4BTrajectoryStore
from cutagent.models.qwen_policy_m4b import Qwen3VLPolicyBackendM4B
from cutagent.retrieval.adaptive import AdaptiveHybridRetriever
from cutagent.retrieval.encoders import BGETextEncoder, Siglip2VisualEncoder
from cutagent.retrieval.fusion import WeightedRankFusion
from cutagent.retrieval.index import KeyframeAsset, RetrievalIndexBuilder
from cutagent.retrieval.protocols import MultimodalRetriever
from cutagent.retrieval.retrievers import DenseTextRetriever, SparseRetriever, VisualRetriever
from cutagent.schemas.m4b_agent import M4BAgentTrajectory, M4BRuntimeConfig
from cutagent.schemas.media import IngestionResult
from cutagent.schemas.perception import PerceptionResult
from cutagent.schemas.retrieval import AdaptiveRetrievalConfig, RetrievalChannel, RetrievalConfig
from cutagent.tools.factory import create_m3a_registry

DECLARED_METRICS = (
    "task_success_rate",
    "hard_constraint_satisfaction",
    "correct_final_artifact_rate",
    "correct_impossible_refusal_rate",
    "recall_at_1",
    "recall_at_5",
    "mrr",
    "temporal_iou",
    "tool_selection_accuracy",
    "tool_argument_validity",
    "structured_output_validity",
    "termination_loop_budget_recovery",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("M5B requires local file artifacts")
    value = unquote(parsed.path)
    if len(value) >= 3 and value[0] == "/" and value[2] == ":":
        value = value[1:]
    return Path(value).resolve(strict=True)


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
        raise RuntimeError("M5B prepared public media is incomplete")
    by_video = {item.video.video_id: item for item in ingestions}
    if set(by_video) != {item.world_state.video_id for item in perceptions}:
        raise RuntimeError("M5B ingestion/perception video identities differ")
    return ingestions, perceptions


def _keyframes(ingestions: tuple[IngestionResult, ...]) -> tuple[KeyframeAsset, ...]:
    values: dict[str, KeyframeAsset] = {}
    for ingestion in ingestions:
        for keyframe in ingestion.keyframes:
            values[keyframe.artifact.artifact_id] = KeyframeAsset(
                artifact=keyframe.artifact,
                path=_artifact_path(keyframe.artifact.uri),
            )
    return tuple(values[key] for key in sorted(values))


def _injection_config(gold: FailureInjectionGold, task_id: str) -> FailureInjectionConfig:
    failure_map = {
        "search_no_results": "search_no_results",
        "timeout": "tool_timeout",
        "invalid_arguments": "invalid_tool_arguments",
        "artifact_not_allowed": "artifact_not_allowed",
        "post_validation_failure": "post_execution_validation_failure",
        "corrupt_media": "corrupt_media",
        "incompatible_concat": "incompatible_concat_inputs",
        "invalid_subtitle": "invalid_subtitle_timing",
        "output_too_large": "output_size_limit",
        "repeated_editor": "repeated_editor_stagnation",
    }
    mode_map = {
        "search_no_results": "empty_search",
        "post_validation_failure": "post_execute_validation",
        "repeated_editor": "repeated_editor",
    }
    return FailureInjectionConfig(
        injection_id=f"m5b-injection-{task_id}",
        task_id=task_id,
        failure_type=cast(Any, failure_map[gold.category]),
        trigger_mode=cast(Any, mode_map.get(gold.category, "pre_execute_failure")),
        trigger_tool_names=(gold.inject_on_tool,),
        trigger_occurrence=gold.occurrence,
        expected_recovery_operations=("retry_current_node", "cannot_recover"),
    )


def _trajectory_manifest(root: Path, trajectories: list[M4BAgentTrajectory]) -> str:
    records = []
    for trajectory in trajectories:
        path = root / "progress" / f"{trajectory.task_input.task_id}.json"
        records.append(
            {
                "task_id": trajectory.task_input.task_id,
                "run_id": trajectory.run_id,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    _write_json(root / "trajectory_manifest.json", records)
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    values.sort()
    return values[round((len(values) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", required=True, choices=("dev", "validation", "locked_test", "adversarial_test")
    )
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path("artifacts/m5a/cutagentbench_v0.1")
    )
    parser.add_argument("--prepared-root", type=Path, default=Path("artifacts/m5b/prepared"))
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--adapter-hash", required=True)
    parser.add_argument("--access-reason", required=True)
    parser.add_argument(
        "--human-gate",
        type=Path,
        default=Path("artifacts/m5a/human_calibration/final/human_calibration.json"),
    )
    parser.add_argument("--access-record", type=Path, default=None)
    parser.add_argument(
        "--variant", choices=("handoff_only", "compact_recovery"), default="handoff_only"
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    split = DatasetSplit(args.split)
    runtime_config = M4BRuntimeConfig(protocol_variant=args.variant)
    retrieval_config = RetrievalConfig()
    identity = {
        "runtime_config": runtime_config.model_dump(mode="json"),
        "retrieval_config": retrieval_config.model_dump(mode="json"),
        "adaptive_config": AdaptiveRetrievalConfig(native_video_enabled=False).model_dump(
            mode="json"
        ),
        "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "prompt_version": "m4b-handoff-policy-v1",
    }
    declaration = M5BRunDeclaration(
        split=split,
        adapter_hash=args.adapter_hash,
        config_sha256=config_sha256(identity),
        declared_metrics=DECLARED_METRICS,
        access_reason=args.access_reason,
        output_directory=str(args.output_directory),
        resume=not args.no_resume,
        protocol_variant=args.variant,
    )
    assert_split_access(
        declaration,
        human_gate_path=args.human_gate,
        access_record_path=args.access_record,
    )
    root = args.output_directory.resolve()
    _write_json(root / "run_declaration.json", declaration.model_dump(mode="json"))
    cases = load_cases(
        args.benchmark_root.resolve(strict=True),
        declaration,
        human_gate_path=args.human_gate,
        access_record_path=args.access_record,
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        cases = cases[: args.limit]
    prepared = args.prepared_root.resolve() / split.value
    ingestions, perceptions = _load_prepared(prepared)
    available_video_ids = {item.video.video_id for item in ingestions}
    required_video_ids = {task.video_ref.artifact_id for task, _ in cases}
    if not required_video_ids.issubset(available_video_ids):
        raise RuntimeError("M5B prepared media does not cover selected public tasks")

    torch: Any = importlib.import_module("torch")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    text_encoder = BGETextEncoder(
        model_cache=args.model_cache.resolve(strict=True),
        batch_size=retrieval_config.text_batch_size,
        maximum_tokens=retrieval_config.text_encoder_max_tokens,
    )
    visual_encoder = Siglip2VisualEncoder(
        model_cache=args.model_cache.resolve(strict=True),
        batch_size=retrieval_config.visual_batch_size,
    )
    memory = RetrievalIndexBuilder(root / "index").build(
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
    for task, _ in cases:
        registry.artifact_store.import_file(
            _artifact_path(task.video_ref.uri),
            media_type=task.video_ref.media_type,
            artifact_id=task.video_ref.artifact_id,
        )
    policy = Qwen3VLPolicyBackendM4B(model_cache=args.model_cache.resolve(strict=True))
    store = M4BTrajectoryStore(root / "trajectory_artifacts")
    trajectories: list[M4BAgentTrajectory] = []
    evaluations = []
    resume_hits = 0
    started = time.perf_counter_ns()
    for index, (task, gold) in enumerate(cases, start=1):
        path = root / "progress" / f"{task.task_id}.json"
        if declaration.resume and path.is_file():
            trajectory = M4BAgentTrajectory.model_validate_json(path.read_text(encoding="utf-8"))
            if trajectory.task_input != task or trajectory.runtime_config != runtime_config:
                raise RuntimeError(f"incompatible trajectory resume: {path}")
            resumed = True
        else:
            execution_registry: Any = registry
            injector: DeterministicFailureInjectingRegistry | None = None
            if gold.failure_injection is not None:
                injector = DeterministicFailureInjectingRegistry(
                    registry,
                    _injection_config(gold.failure_injection, task.task_id),
                )
                execution_registry = injector
            runtime = M4BAgentRuntime(
                registry=execution_registry,
                policy_model=policy,
                artifact_root=root / "runs",
            )
            trajectory = runtime.run(
                task,
                config=runtime_config,
                run_id=f"m5b-{split.value}-{args.variant}-{task.task_id}",
            )
            _write_json(path, trajectory.model_dump(mode="json"))
            if injector is not None:
                trigger = finalize_trigger(injector.private_trigger(), trajectory)
                _write_json(
                    root / "private" / "failure_triggers" / f"{task.task_id}.json",
                    trigger.model_dump(mode="json"),
                )
            resumed = False
        resume_hits += resumed
        trajectories.append(trajectory)
        evaluations.append(evaluate_trajectory(trajectory, gold))
        store.write(trajectory)
        print(
            f"[{split.value}] {index}/{len(cases)} {task.task_id} "
            f"terminal={trajectory.terminal_reason} resumed={resumed}",
            flush=True,
        )
    metrics = summarize_evaluations(evaluations)
    replay_count = sum(
        M4BStateReducer.replay(item.initial_state, item.events) == item.final_state
        for item in trajectories
    )
    leak_count = sum(trajectory_has_private_fields(item) for item in trajectories)
    manifest_hash = _trajectory_manifest(root, trajectories)
    protected_count = int(split in {DatasetSplit.LOCKED_TEST, DatasetSplit.ADVERSARIAL_TEST})
    status = "OFFICIAL_PROTECTED_EVALUATION" if protected_count else "PROVISIONAL_VALIDATION_ONLY"
    summary = M5BRunSummary(
        declaration=declaration,
        status=cast(Any, status),
        metrics=metrics,
        task_count=len(trajectories),
        resume_hits=resume_hits,
        deterministic_replay_count=replay_count,
        private_trajectory_leak_count=leak_count,
        protected_access_count=protected_count,
        trajectory_manifest_sha256=manifest_hash,
    )
    _write_json(root / "evaluations.json", [item.model_dump(mode="json") for item in evaluations])
    performance = {
        "wall_time_ms": (time.perf_counter_ns() - started) // 1_000_000,
        "task_latency_p50_ms": _percentile([item.latency.total_ms for item in trajectories], 0.5),
        "task_latency_p95_ms": _percentile([item.latency.total_ms for item in trajectories], 0.95),
        "terminal_reason_counts": dict(Counter(item.terminal_reason for item in trajectories)),
        "policy_model_load_time_ms": policy.load_time_ms,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    _write_json(root / "metrics.json", summary.model_dump(mode="json"))
    _write_json(root / "performance.json", performance)
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True))
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
