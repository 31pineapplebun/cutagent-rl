"""Execute train-only oracle trajectories and build the M6 decision SFT dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

from cutagent_evaluation.m4b5_recovery import (
    DeterministicFailureInjectingRegistry,
    FailureInjectionConfig,
    finalize_trigger,
)
from cutagent_evaluation.m5a_evaluator import evaluate_trajectory
from cutagent_evaluation.m5a_schemas import CutAgentBenchGold, FailureInjectionGold
from cutagent_evaluation.m6_dataset import assert_executable_targets, records_from_trajectories
from cutagent_evaluation.m6_oracle import M6ExecutedOraclePolicy
from cutagent_evaluation.m6_verification import M6OracleOnlineVerifier
from cutagent_training.contracts import M6AgentSFTRecord, build_m6_manifest

from cutagent.agent.m4b_policy_context import M4BPolicyContextBuilder
from cutagent.agent.m4b_runtime import M4BAgentRuntime
from cutagent.agent.m4b_state_reducer import M4BStateReducer
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
from cutagent.schemas.task_input import TaskInput
from cutagent.tools.factory import create_m3a_registry


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("M6 requires local immutable artifacts")
    value = unquote(parsed.path)
    if len(value) >= 3 and value[0] == "/" and value[2] == ":":
        value = value[1:]
    return Path(value).resolve(strict=True)


def _load_list(path: Path) -> list[object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON list: {path}")
    return value


def _load_train_cases(root: Path) -> tuple[tuple[TaskInput, CutAgentBenchGold], ...]:
    public = tuple(
        TaskInput.model_validate(item) for item in _load_list(root / "public/train.json")
    )
    private = tuple(
        CutAgentBenchGold.model_validate(item)
        for item in _load_list(root / "private/train_gold.json")
    )
    if tuple(item.task_id for item in public) != tuple(item.task_id for item in private):
        raise ValueError("M6 train public tasks and private annotations differ")
    if any(item.split.value != "train" for item in private):
        raise ValueError("M6 dataset builder may read train annotations only")
    return tuple(zip(public, private, strict=True))


def _load_prepared(root: Path) -> tuple[tuple[IngestionResult, ...], tuple[PerceptionResult, ...]]:
    ingestions = tuple(
        IngestionResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((root / "ingestion").glob("*.json"))
    )
    perceptions = tuple(
        PerceptionResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((root / "perception").glob("*.json"))
    )
    if not ingestions or len(ingestions) != len(perceptions):
        raise RuntimeError("M6 train public media preparation is incomplete")
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


def _injection(gold: FailureInjectionGold, task_id: str) -> FailureInjectionConfig:
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
        injection_id=f"m6-injection-{task_id}",
        task_id=task_id,
        failure_type=cast(Any, failure_map[gold.category]),
        trigger_mode=cast(Any, mode_map.get(gold.category, "pre_execute_failure")),
        trigger_tool_names=(gold.inject_on_tool,),
        trigger_occurrence=gold.occurrence,
        expected_recovery_operations=("retry_current_node", "cannot_recover"),
    )


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump(mode="json")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _training_rows(records: tuple[M6AgentSFTRecord, ...], subsplit: str) -> list[object]:
    return [
        {"messages": [message.model_dump(mode="json") for message in record.messages]}
        for record in records
        if record.train_subsplit == subsplit
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path("artifacts/m5a/cutagentbench_v0.1")
    )
    parser.add_argument("--prepared-root", type=Path, default=Path("artifacts/m6/prepared/train"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/m6"))
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    benchmark_root = args.benchmark_root.resolve(strict=True)
    output = args.output_root.resolve()
    cases = _load_train_cases(benchmark_root)
    gold_by_task = {gold.task_id: gold for _, gold in cases}
    ingestions, perceptions = _load_prepared(args.prepared_root.resolve(strict=True))
    retrieval_config = RetrievalConfig()
    text_encoder = BGETextEncoder(
        model_cache=args.model_cache.resolve(strict=True),
        batch_size=retrieval_config.text_batch_size,
        maximum_tokens=retrieval_config.text_encoder_max_tokens,
    )
    visual_encoder = Siglip2VisualEncoder(
        model_cache=args.model_cache.resolve(strict=True),
        batch_size=retrieval_config.visual_batch_size,
    )
    memory = RetrievalIndexBuilder(output / "retrieval_index").build(
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
        root=output / "tool_runtime",
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
    oracle = M6ExecutedOraclePolicy(gold_by_task)
    trajectories: list[M4BAgentTrajectory] = []
    evaluations = []
    trigger_count = 0
    for index, (task, gold) in enumerate(cases, start=1):
        trajectory_path = output / "oracle_trajectories" / f"{task.task_id}.json"
        expected_variant = "compact_recovery" if gold.failure_injection else "handoff_only"
        if args.resume and trajectory_path.is_file():
            trajectory = M4BAgentTrajectory.model_validate_json(
                trajectory_path.read_text(encoding="utf-8")
            )
            if trajectory.task_input != task or trajectory.protocol_variant != expected_variant:
                raise RuntimeError("M6 resume checkpoint is incompatible")
            resumed = trajectory.terminal_reason in {"SUCCESS", "CANNOT_COMPLETE"}
        else:
            resumed = False
        if not resumed:
            execution_registry: Any = registry
            injector: DeterministicFailureInjectingRegistry | None = None
            if gold.failure_injection is not None:
                injector = DeterministicFailureInjectingRegistry(
                    registry,
                    _injection(gold.failure_injection, task.task_id),
                )
                execution_registry = injector
            runtime = M4BAgentRuntime(
                registry=execution_registry,
                policy_model=oracle,
                artifact_root=output / "oracle_runs",
                context_builder=M4BPolicyContextBuilder(history_limit=3),
                verifier=M6OracleOnlineVerifier(),
            )
            trajectory = runtime.run(
                task,
                config=M4BRuntimeConfig(
                    protocol_variant=cast(Any, expected_variant),
                    max_steps=16,
                    max_tool_calls=14,
                    max_edit_calls=10,
                ),
                run_id=f"m6-oracle-{task.task_id}",
            )
            _write(trajectory_path, trajectory.model_dump(mode="json"))
            if injector is not None:
                trigger = finalize_trigger(injector.private_trigger(), trajectory)
                _write(
                    output / "private/failure_triggers" / f"{task.task_id}.json",
                    trigger.model_dump(mode="json"),
                )
                trigger_count += trigger.triggered
        evaluation = evaluate_trajectory(trajectory, gold)
        trajectories.append(trajectory)
        evaluations.append(evaluation)
        print(
            f"[M6 oracle] {index}/{len(cases)} success={evaluation.task_success} "
            f"terminal={trajectory.terminal_reason} resumed={resumed}",
            flush=True,
        )
    successful = tuple(
        trajectory
        for trajectory, evaluation in zip(trajectories, evaluations, strict=True)
        if evaluation.task_success
    )
    groups = sorted({gold.source_group_id for _, gold in cases})
    holdout_groups = frozenset(groups[-2:])
    trigger_count = sum(
        json.loads(path.read_text(encoding="utf-8")).get("triggered") is True
        for path in (output / "private/failure_triggers").glob("*.json")
    )
    records = records_from_trajectories(
        successful,
        gold_by_task=gold_by_task,
        holdout_source_groups=holdout_groups,
        tool_manifest=registry.manifest(),
    )
    audit_count = assert_executable_targets(records, registry.manifest())
    manifest = build_m6_manifest(records, executable_audit_count=audit_count)
    replay_count = sum(
        M4BStateReducer.replay(item.initial_state, item.events) == item.final_state
        for item in trajectories
    )
    _write_jsonl(output / "dataset/m6_agent_sft.jsonl", list(records))
    _write_jsonl(output / "dataset/sft_train.jsonl", _training_rows(records, "train"))
    _write_jsonl(output / "dataset/sft_holdout.jsonl", _training_rows(records, "holdout"))
    _write(output / "dataset/manifest.json", manifest.model_dump(mode="json"))
    _write(
        output / "dataset/build_summary.json",
        {
            "task_count": len(cases),
            "successful_oracle_trajectory_count": len(successful),
            "failed_oracle_trajectory_count": len(cases) - len(successful),
            "event_replay_count": replay_count,
            "failure_injection_trigger_count": trigger_count,
            "holdout_source_groups": sorted(holdout_groups),
            "records_sha256": manifest.records_sha256,
            "protected_split_access_count": 0,
            "train_gold_used_only_by_oracle": True,
        },
    )
    _write(
        output / "dataset/evaluations.json",
        [item.model_dump(mode="json") for item in evaluations],
    )
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
