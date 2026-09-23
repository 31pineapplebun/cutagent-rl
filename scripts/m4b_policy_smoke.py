"""Run one M4B compact-recovery protocol smoke on the inspected M4A dev set."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from cutagent_evaluation.m4a_agent import build_m4a_development_set

from cutagent.agent.m4b_runtime import M4BAgentRuntime
from cutagent.agent.m4b_state_reducer import M4BStateReducer
from cutagent.models.qwen_policy_m4b import Qwen3VLPolicyBackendM4B
from cutagent.retrieval.adaptive import AdaptiveHybridRetriever
from cutagent.retrieval.encoders import BGETextEncoder, Siglip2VisualEncoder
from cutagent.retrieval.fusion import WeightedRankFusion
from cutagent.retrieval.index import RetrievalIndexBuilder
from cutagent.retrieval.protocols import MultimodalRetriever
from cutagent.retrieval.retrievers import DenseTextRetriever, SparseRetriever, VisualRetriever
from cutagent.schemas.m4b_agent import M4BRuntimeConfig
from cutagent.schemas.retrieval import AdaptiveRetrievalConfig, RetrievalChannel, RetrievalConfig
from cutagent.tools.factory import create_m3a_registry
from scripts.m4b_run_experiment import (
    M4A_SOURCE_GROUPS,
    _artifact_path,
    _keyframes,
    _load_gold,
    _load_prepared,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2b-root", type=Path, default=Path("artifacts/m2b"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m4b/smoke"))
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--task-id", default="m4a-dev-001")
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    m2b_root = args.m2b_root.resolve()
    ingestions, perceptions = _load_prepared(m2b_root)
    source_gold = tuple(
        item for item in _load_gold(m2b_root) if item.source_group_id in M4A_SOURCE_GROUPS
    )
    ingestion_by_hash = {item.video.source.sha256: item for item in ingestions}

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
    case = next((item for item in cases if item.task_input.task_id == args.task_id), None)
    if case is None:
        raise ValueError(f"unknown M4A development smoke task: {args.task_id}")
    torch: Any = importlib.import_module("torch")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    policy = Qwen3VLPolicyBackendM4B(model_cache=args.model_cache.resolve())
    runtime = M4BAgentRuntime(
        registry=registry,
        policy_model=policy,
        artifact_root=root / "runs",
    )
    config = M4BRuntimeConfig(protocol_variant="compact_recovery")
    trajectory = runtime.run(
        case.task_input,
        config=config,
        run_id=f"m4b-protocol-smoke-{case.task_input.task_id}",
    )
    replayed = M4BStateReducer.replay(trajectory.initial_state, trajectory.events)
    serialized = trajectory.model_dump_json().casefold()
    leak_count = sum(
        token in serialized for token in ("source_group_id", '"split"', "gold_id", "benchmarkgold")
    )
    smoke_pass = bool(
        trajectory.initial_plan is not None
        and len(trajectory.policy_steps) >= 2
        and trajectory.tool_records
        and replayed == trajectory.final_state
        and leak_count == 0
    )
    result = {
        "schema_version": "1.0",
        "smoke_pass": smoke_pass,
        "development_task_id": case.task_input.task_id,
        "heldout_validation_used": False,
        "model_id": trajectory.model.model_id,
        "model_revision": trajectory.model.revision,
        "dtype": trajectory.model.dtype,
        "planner_prompt_version": config.planner_template_version,
        "policy_prompt_version": config.prompt_template_version,
        "terminal_reason": trajectory.terminal_reason,
        "policy_steps": len(trajectory.policy_steps),
        "policy_failures": len(trajectory.policy_failures),
        "tool_calls": len(trajectory.tool_records),
        "recovery_events": len(trajectory.final_state.recovery_events),
        "replay_identical": replayed == trajectory.final_state,
        "private_label_leak_count": leak_count,
        "model_load_time_ms": policy.load_time_ms,
        "latency_ms": trajectory.latency.model_dump(mode="json"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    _write_json(root / "result.json", result)
    _write_json(root / "trajectory.json", trajectory.model_dump(mode="json"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if smoke_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
