"""Run the frozen 50-task M4A comparison using real local Qwen and M3A tools."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from cutagent_evaluation.m2b_dataset import HeldOutVideoGold
from cutagent_evaluation.m4a_agent import (
    M4ADevelopmentCase,
    M4ATrajectoryEvaluation,
    build_m4a_development_set,
    evaluate_trajectory,
    summarize_evaluations,
)

from cutagent.agent.runtime import AgentRuntime
from cutagent.agent.trajectory import TrajectoryStore
from cutagent.models.qwen_policy import Qwen3VLPolicyBackend
from cutagent.retrieval.adaptive import AdaptiveHybridRetriever
from cutagent.retrieval.encoders import BGETextEncoder, Siglip2VisualEncoder
from cutagent.retrieval.fusion import WeightedRankFusion
from cutagent.retrieval.index import KeyframeAsset, RetrievalIndexBuilder
from cutagent.retrieval.protocols import MultimodalRetriever
from cutagent.retrieval.retrievers import DenseTextRetriever, SparseRetriever, VisualRetriever
from cutagent.schemas.agent import AgentRuntimeConfig, AgentTrajectory
from cutagent.schemas.media import IngestionResult
from cutagent.schemas.perception import PerceptionResult
from cutagent.schemas.retrieval import (
    AdaptiveRetrievalConfig,
    RetrievalChannel,
    RetrievalConfig,
)
from cutagent.tools.factory import create_m3a_registry

VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("react_structured", "react", "structured_state"),
    ("hierarchical_structured", "hierarchical", "structured_state"),
    ("hierarchical_visual", "hierarchical", "multimodal_evidence"),
)


def _artifact_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("M4A experiment requires local file artifacts")
    text = unquote(parsed.path)
    if len(text) >= 3 and text[0] == "/" and text[2] == ":":
        text = text[1:]
    return Path(text).resolve(strict=True)


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


def _keyframes(ingestions: tuple[IngestionResult, ...]) -> tuple[KeyframeAsset, ...]:
    values: dict[str, KeyframeAsset] = {}
    for ingestion in ingestions:
        for keyframe in ingestion.keyframes:
            values[keyframe.artifact.artifact_id] = KeyframeAsset(
                artifact=keyframe.artifact,
                path=_artifact_path(keyframe.artifact.uri),
            )
    return tuple(values[key] for key in sorted(values))


def _load_gold(root: Path) -> tuple[HeldOutVideoGold, ...]:
    payload = json.loads((root / "heldout_source_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("M2B held-out source manifest must be a list")
    return tuple(HeldOutVideoGold.model_validate(item) for item in payload)


def _load_or_run(
    *,
    path: Path,
    runtime: AgentRuntime,
    case: M4ADevelopmentCase,
    config: AgentRuntimeConfig,
    run_id: str,
) -> tuple[AgentTrajectory, bool]:
    if path.is_file():
        trajectory = AgentTrajectory.model_validate_json(path.read_text(encoding="utf-8"))
        if (
            trajectory.task_input.task_id == case.task_input.task_id
            and trajectory.runtime_config == config
        ):
            return trajectory, True
        raise RuntimeError(f"incompatible resumable trajectory exists: {path}")
    trajectory = runtime.run(case.task_input, config=config, run_id=run_id)
    _write_json(path, trajectory.model_dump(mode="json"))
    return trajectory, False


def _performance(trajectories: list[AgentTrajectory]) -> dict[str, Any]:
    task_latencies = sorted(item.latency.total_ms for item in trajectories)
    policy_steps = [step for item in trajectories for step in item.policy_steps]
    tool_records = [record for item in trajectories for record in item.tool_records]

    def percentile(values: list[int], fraction: float) -> int:
        if not values:
            return 0
        return values[min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))]

    return {
        "task_latency_p50_ms": percentile(task_latencies, 0.50),
        "task_latency_p95_ms": percentile(task_latencies, 0.95),
        "mean_policy_latency_per_step_ms": (
            sum(step.stats.latency_ms for step in policy_steps) / max(len(policy_steps), 1)
        ),
        "policy_latency_total_ms": sum(item.latency.policy_ms for item in trajectories),
        "retrieval_latency_total_ms": sum(item.latency.retrieval_ms for item in trajectories),
        "editing_tool_latency_total_ms": sum(item.latency.editing_tool_ms for item in trajectories),
        "verification_latency_total_ms": sum(item.latency.verification_ms for item in trajectories),
        "qwen_peak_allocated_bytes": max(
            (step.stats.peak_allocated_bytes or 0 for step in policy_steps), default=0
        ),
        "qwen_peak_reserved_bytes": max(
            (step.stats.peak_reserved_bytes or 0 for step in policy_steps), default=0
        ),
        "tool_cache_hits": sum(record.trace.cache_hit for record in tool_records),
        "tool_cache_misses": sum(not record.trace.cache_hit for record in tool_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2b-root", type=Path, default=Path("artifacts/m2b"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m4a"))
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--variant", choices=tuple(item[0] for item in VARIANTS), default=None)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    root = args.artifact_root.resolve()
    m2b_root = args.m2b_root.resolve()
    ingestions, perceptions = _load_prepared(m2b_root)
    source_gold = _load_gold(m2b_root)
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
        for item in source_gold
    }
    cases = build_m4a_development_set(source_gold, source_refs)
    if args.task_id is not None:
        cases = tuple(case for case in cases if case.task_input.task_id == args.task_id)
        if not cases:
            raise ValueError(f"unknown M4A development task ID: {args.task_id}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        cases = cases[: args.limit]
    _write_json(
        root / "development_tasks_public.json",
        [case.task_input.model_dump(mode="json") for case in cases],
    )
    _write_json(
        root / "private" / "development_gold.json",
        [case.gold.model_dump(mode="json") for case in cases],
    )
    visual_paths = {asset.artifact.artifact_id: asset.path for asset in _keyframes(ingestions)}
    policy = Qwen3VLPolicyBackend(model_cache=args.model_cache.resolve())
    runtime = AgentRuntime(
        registry=registry,
        policy_model=policy,
        artifact_root=root / "runs",
        visual_artifacts=visual_paths,
    )
    experiment: dict[str, Any] = {
        "schema_version": "1.0",
        "task_count": len(cases),
        "development_only": True,
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "dtype": "bfloat16",
        "seed": 20_260_823,
        "retrieval_index_id": memory.manifest.index_id,
        "native_video_verification": "disabled",
        "variants": {},
    }
    store = TrajectoryStore(root / "trajectory_artifacts")
    selected_variants = tuple(
        item for item in VARIANTS if args.variant is None or item[0] == args.variant
    )
    for variant, baseline, view in selected_variants:
        config = AgentRuntimeConfig(
            baseline=baseline,  # type: ignore[arg-type]
            policy_view_mode=view,  # type: ignore[arg-type]
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
        trajectories: list[AgentTrajectory] = []
        evaluations: list[M4ATrajectoryEvaluation] = []
        resume_hits = 0
        for index, case in enumerate(cases, start=1):
            path = root / "progress" / variant / f"{case.task_input.task_id}.json"
            trajectory, resumed = _load_or_run(
                path=path,
                runtime=runtime,
                case=case,
                config=config,
                run_id=f"{variant}-{case.task_input.task_id}",
            )
            resume_hits += resumed
            trajectories.append(trajectory)
            evaluations.append(evaluate_trajectory(trajectory, case.gold))
            store.write(trajectory)
            print(
                f"[{variant}] {index}/{len(cases)} {case.task_input.task_id} "
                f"terminal={trajectory.terminal_reason} resumed={resumed}",
                flush=True,
            )
        summary = summarize_evaluations(evaluations)
        _write_json(
            root / "evaluations" / f"{variant}.json",
            [item.model_dump(mode="json") for item in evaluations],
        )
        variant_payload = {
            "metrics": summary.model_dump(mode="json"),
            "performance": _performance(trajectories),
            "terminal_reason_counts": dict(
                __import__("collections", fromlist=["Counter"]).Counter(
                    item.terminal_reason for item in trajectories
                )
            ),
            "resume_hits": resume_hits,
            "trajectory_ids": [item.trajectory_id for item in trajectories],
        }
        experiment["variants"][variant] = variant_payload
        _write_json(root / "metrics" / f"{variant}.json", variant_payload)
    experiment["policy_backend_version"] = policy.backend_version
    experiment["model_load_time_ms"] = policy.load_time_ms
    experiment["experiment_wall_time_ms"] = (time.perf_counter_ns() - started) // 1_000_000
    experiment["process_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    experiment["process_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    _write_json(root / "experiment_summary.json", experiment)
    print(json.dumps(experiment, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
