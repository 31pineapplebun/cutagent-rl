"""Prepare and evaluate the held-out M2B adaptive retrieval experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from cutagent_evaluation.m2b_dataset import (
    HeldOutVideoGold,
    assert_source_group_disjoint,
    build_heldout_validation_cases,
    generate_heldout_dataset,
)
from cutagent_evaluation.m2b_retrieval import (
    M2BRetrievalCaseGold,
    attribute_failures,
    evaluate_m2b_method,
    native_effect_summary,
    router_confusion,
)
from cutagent_evaluation.retrieval import QueryRetrievalMetric, summarize_metrics

from cutagent.core.artifacts import ArtifactRef
from cutagent.ingestion.pipeline import VideoIngestionPipeline
from cutagent.perception.pipeline import MultimodalPerceptionPipeline
from cutagent.retrieval.adaptive import AdaptiveHybridRetriever
from cutagent.retrieval.encoders import BGETextEncoder, Siglip2VisualEncoder
from cutagent.retrieval.fusion import WeightedRankFusion
from cutagent.retrieval.index import KeyframeAsset, RetrievalIndexBuilder
from cutagent.retrieval.native_video import (
    QwenNativeVideoVerifier,
    SceneClipAsset,
    SceneClipStore,
)
from cutagent.retrieval.protocols import FloatMatrix, MultimodalRetriever
from cutagent.retrieval.query import RoutingStrategy
from cutagent.retrieval.retrievers import DenseTextRetriever, SparseRetriever, VisualRetriever
from cutagent.retrieval.store import RetrievalMemory
from cutagent.schemas.media import (
    IngestionConfig,
    IngestionResult,
    KeyframeExtractionConfig,
)
from cutagent.schemas.perception import PerceptionConfig, PerceptionResult
from cutagent.schemas.retrieval import (
    AdaptiveRetrievalConfig,
    AdaptiveRetrievalResult,
    RetrievalChannel,
    RetrievalConfig,
    RetrievalResponse,
)


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_path(artifact: ArtifactRef) -> Path:
    parsed = urlparse(artifact.uri)
    if parsed.scheme != "file":
        raise ValueError("M2B requires local file artifacts")
    return Path(unquote(parsed.path)).resolve(strict=True)


def _latency(responses: Mapping[str, RetrievalResponse]) -> dict[str, float]:
    values = sorted(item.latency_ms for item in responses.values())
    p95 = values[min(len(values) - 1, max(0, int(0.95 * len(values)) - 1))]
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": p95,
        "max_ms": max(values),
    }


def _index_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class _MeasuredTextEncoder:
    def __init__(self, backend: BGETextEncoder) -> None:
        self.backend = backend
        self.elapsed_ms = 0.0
        self.items = 0

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    @property
    def revision(self) -> str:
        return self.backend.revision

    @property
    def embedding_dimension(self) -> int:
        return self.backend.embedding_dimension

    def encode(self, texts: tuple[str, ...]) -> FloatMatrix:
        started = time.perf_counter()
        result = self.backend.encode(texts)
        self.elapsed_ms += (time.perf_counter() - started) * 1000
        self.items += len(texts)
        return result


class _MeasuredVisualEncoder:
    def __init__(self, backend: Siglip2VisualEncoder) -> None:
        self.backend = backend
        self.elapsed_ms = 0.0
        self.items = 0

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    @property
    def revision(self) -> str:
        return self.backend.revision

    def encode_images(self, paths: tuple[Path, ...]) -> FloatMatrix:
        started = time.perf_counter()
        result = self.backend.encode_images(paths)
        self.elapsed_ms += (time.perf_counter() - started) * 1000
        self.items += len(paths)
        return result

    def encode_queries(self, texts: tuple[str, ...]) -> FloatMatrix:
        return self.backend.encode_queries(texts)


def _source_path(root: Path, record: HeldOutVideoGold) -> Path:
    suffix = int(record.source_group_id.rsplit("-", 1)[1])
    return root / "dataset" / f"heldout-{suffix:02d}" / "source.mp4"


def _prepare(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    root: Path = args.artifact_root.resolve()
    records_with_paths = generate_heldout_dataset(root / "dataset", ffmpeg=args.ffmpeg)
    records = tuple(record for _, record in records_with_paths)
    _json_write(
        root / "heldout_source_manifest.json", [item.model_dump(mode="json") for item in records]
    )
    ingestion_pipeline = VideoIngestionPipeline(
        cache_root=root / "ingestion_cache",
        ffmpeg_executable=args.ffmpeg,
        ffprobe_executable=args.ffprobe,
    )
    perception_pipeline = MultimodalPerceptionPipeline(
        cache_root=root / "perception_cache",
        model_cache=args.model_cache,
        ffmpeg_executable=args.ffmpeg,
    )
    ingestion_config = IngestionConfig(
        # The held-out generator has an empirically verified gap between its
        # strongest within-scene transition (0.065889) and weakest intended
        # scene boundary (0.099605).  Use a fixed threshold inside that gap.
        scene_threshold=0.08,
        minimum_scene_duration_ms=700,
        keyframes=KeyframeExtractionConfig(strategy="uniform", frames_per_scene=3),
    )
    perception_config = PerceptionConfig(
        visual_mode="keyframes",
        prompt_template_version="m1b-structured-v1.2",
        temporal_prompt_style="explicit_comparison",
        label_temporal_phases=True,
        maximum_repair_attempts=2,
        maximum_new_tokens=768,
        asr_enabled=True,
    )
    summaries: dict[str, dict[str, Any]] = {}
    for source, record in records_with_paths:
        ingestion_path = root / "ingestion" / f"{record.source_group_id}.json"
        ingestion: IngestionResult | None = None
        if ingestion_path.is_file():
            ingestion = IngestionResult.model_validate_json(
                ingestion_path.read_text(encoding="utf-8")
            )
        if ingestion is None or ingestion.config != ingestion_config:
            ingestion = ingestion_pipeline.ingest(source, config=ingestion_config)
            _json_write(ingestion_path, ingestion.model_dump(mode="json"))
        perception_path = root / "perception" / f"{record.source_group_id}.json"
        perception: PerceptionResult | None = None
        if perception_path.is_file():
            perception = PerceptionResult.model_validate_json(
                perception_path.read_text(encoding="utf-8")
            )
        if (
            perception is None
            or perception.config != perception_config
            or perception.world_state.video_id != ingestion.video.video_id
            or tuple(scene.segment_id for scene in perception.world_state.scenes)
            != tuple(scene.segment_id for scene in ingestion.scenes)
        ):
            perception = perception_pipeline.run(
                source,
                ingestion=ingestion,
                config=perception_config,
            )
            _json_write(perception_path, perception.model_dump(mode="json"))
        if ingestion.video.source.sha256 != record.source_sha256:
            raise RuntimeError("generated source hash differs from private provenance")
        expected_ranges = ((0, 3_000), (3_000, 6_000), (6_000, 9_000), (9_000, 12_000))
        actual_ranges = tuple(
            (scene.time_range.start_ms, scene.time_range.end_ms) for scene in ingestion.scenes
        )
        if actual_ranges != expected_ranges:
            raise RuntimeError(
                f"held-out multi-scene contract failed for {record.source_group_id}: "
                f"expected {expected_ranges}, got {actual_ranges}"
            )
        summaries[record.source_group_id] = {
            "video_id": ingestion.video.video_id,
            "duration_ms": ingestion.video.duration_ms,
            "resolution": [
                ingestion.video.video_stream.width,
                ingestion.video.video_stream.height,
            ],
            "scene_count": len(ingestion.scenes),
            "scene_ranges": [
                scene.time_range.model_dump(mode="json") for scene in ingestion.scenes
            ],
            "keyframes": len(ingestion.keyframes),
            "transcript_spans": len(perception.transcript_spans),
            "ocr_spans": len(perception.ocr_spans),
            "actions": sum(len(scene.actions) for scene in perception.world_state.scenes),
            "events": len(perception.temporal_events),
            "perception_cache": [item.model_dump(mode="json") for item in perception.cache_records],
            "performance": [item.model_dump(mode="json") for item in perception.performance],
        }
    payload = {
        "status": "completed",
        "source_groups": len(records),
        "license": "CC0-1.0",
        "generator_version": "m2b-heldout-multiscene-v1",
        "ingestion_config": ingestion_config.model_dump(mode="json"),
        "perception_config": perception_config.model_dump(mode="json"),
        "videos": summaries,
        "total_ms": (time.perf_counter() - started) * 1000,
    }
    _json_write(root / "prepare_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _load_prepared(
    root: Path,
) -> tuple[
    tuple[HeldOutVideoGold, ...],
    dict[str, IngestionResult],
    dict[str, PerceptionResult],
]:
    records = tuple(
        HeldOutVideoGold.model_validate(item)
        for item in _json_read(root / "heldout_source_manifest.json")
    )
    ingestions = {
        record.source_group_id: IngestionResult.model_validate_json(
            (root / "ingestion" / f"{record.source_group_id}.json").read_text(encoding="utf-8")
        )
        for record in records
    }
    perceptions = {
        record.source_group_id: PerceptionResult.model_validate_json(
            (root / "perception" / f"{record.source_group_id}.json").read_text(encoding="utf-8")
        )
        for record in records
    }
    return records, ingestions, perceptions


def _keyframes(ingestions: Mapping[str, IngestionResult]) -> tuple[KeyframeAsset, ...]:
    assets: dict[str, KeyframeAsset] = {}
    for ingestion in ingestions.values():
        for keyframe in ingestion.keyframes:
            assets[keyframe.artifact.artifact_id] = KeyframeAsset(
                artifact=keyframe.artifact,
                path=_artifact_path(keyframe.artifact),
            )
    return tuple(assets[key] for key in sorted(assets))


def _components(
    text: _MeasuredTextEncoder,
    visual: _MeasuredVisualEncoder,
) -> dict[RetrievalChannel, MultimodalRetriever]:
    return {
        "bm25_transcript": SparseRetriever("transcript"),
        "bm25_ocr": SparseRetriever("ocr"),
        "bm25_structured": SparseRetriever("structured_semantic"),
        "bm25_combined": SparseRetriever("combined"),
        "dense_text": DenseTextRetriever(text),
        "visual": VisualRetriever(visual),
    }


def _scene_clips(
    *,
    root: Path,
    records: tuple[HeldOutVideoGold, ...],
    ingestions: Mapping[str, IngestionResult],
    ffmpeg: str,
) -> tuple[dict[tuple[str, str], SceneClipAsset], dict[str, int]]:
    store = SceneClipStore(root / "scene_clip_cache", ffmpeg_executable=ffmpeg)
    assets: dict[tuple[str, str], SceneClipAsset] = {}
    hits = Counter[str]()
    for record in records:
        ingestion = ingestions[record.source_group_id]
        source = _source_path(root, record)
        for scene in ingestion.scenes:
            clip, hit = store.materialize(
                source_path=source,
                source_artifact=ingestion.video.source,
                video_id=ingestion.video.video_id,
                scene_id=scene.segment_id,
                time_range=scene.time_range,
            )
            assets[(clip.video_id, clip.scene_id)] = clip
            hits["hit" if hit else "miss"] += 1
    return assets, dict(hits)


def _breakdown(
    name: str,
    metrics: tuple[QueryRetrievalMetric, ...],
    gold: Mapping[str, M2BRetrievalCaseGold],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for query_type in ("speech", "ocr", "entity", "semantic", "action", "hard_negative"):
        subset = tuple(item for item in metrics if item.query_type == query_type)
        output[query_type] = summarize_metrics(f"{name}:{query_type}", subset).model_dump(
            mode="json"
        )
    for category in (
        "same_entity_different_action",
        "same_text_different_scene",
        "semantically_similar_transcript",
        "visually_similar_distractor",
    ):
        subset = tuple(
            item for item in metrics if gold[item.query_id].hard_negative_category == category
        )
        output[f"hard:{category}"] = summarize_metrics(
            f"{name}:hard:{category}", subset
        ).model_dump(mode="json")
    return output


def _run_variant(
    *,
    name: str,
    strategy: RoutingStrategy,
    evidence_reranking: bool,
    native_video: bool,
    disabled_channels: tuple[RetrievalChannel, ...],
    memory: RetrievalMemory,
    cases: tuple[Any, ...],
    text: _MeasuredTextEncoder,
    visual: _MeasuredVisualEncoder,
    scene_clips: Mapping[tuple[str, str], SceneClipAsset],
    model_cache: Path,
    root: Path,
) -> dict[str, AdaptiveRetrievalResult]:
    config = AdaptiveRetrievalConfig(disabled_channels=disabled_channels)
    fusion = WeightedRankFusion(_components(text, visual), rank_constant=config.rank_constant)
    verifier = (
        QwenNativeVideoVerifier(
            cache_root=root / "native_verification_cache",
            model_cache=model_cache,
            config=config,
            index_id=memory.manifest.index_id,
        )
        if native_video
        else None
    )
    retriever = AdaptiveHybridRetriever(
        config=config,
        fusion=fusion,
        strategy=strategy,
        evidence_reranking=evidence_reranking,
        native_video=native_video,
        verifier=verifier,
        scene_clips=scene_clips,
    )
    results = {
        case.query.query_id: retriever.search_with_trace(case.query, memory) for case in cases
    }
    _json_write(
        root / "responses" / f"{name}.json",
        {key: value.model_dump(mode="json") for key, value in results.items()},
    )
    return results


def _native_call_analysis(
    results: Mapping[str, AdaptiveRetrievalResult],
    gold: Mapping[str, M2BRetrievalCaseGold],
) -> dict[str, object]:
    outcomes: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    latencies: list[int] = []
    peak_allocated = peak_reserved = 0
    first_hits: Counter[str] = Counter()
    for query_id, result in results.items():
        relevant = {(item.video_id, item.scene_id) for item in gold[query_id].relevant_scenes}
        for call in result.native_verifications:
            statuses[call.status] += 1
            latencies.append(call.latency_ms)
            peak_allocated = max(peak_allocated, call.peak_allocated_bytes or 0)
            peak_reserved = max(peak_reserved, call.peak_reserved_bytes or 0)
            first_hits["hit" if call.cache_hit else "miss"] += 1
            identity = (call.video_id, call.scene_id)
            if call.status == "uncertain":
                outcomes["uncertain"] += 1
            elif call.status == "supports" and identity in relevant:
                outcomes["correct_support"] += 1
            elif call.status == "contradicts" and identity not in relevant:
                outcomes["correct_contradiction"] += 1
            elif call.status == "supports":
                outcomes["hallucinated_support"] += 1
            else:
                outcomes["false_contradiction"] += 1
    return {
        "status_counts": dict(statuses),
        "outcomes_against_private_gold": dict(outcomes),
        "cache": dict(first_hits),
        "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "latency_total_ms": sum(latencies),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def _evaluate(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    root: Path = args.artifact_root.resolve()
    records, ingestions, perceptions = _load_prepared(root)
    cases = build_heldout_validation_cases(records, perceptions)
    assert_source_group_disjoint(
        cases, {"m2a-generated-development-v1", "generated-motion-m1b5-v1"}
    )
    _json_write(
        root / "queries_public.json", [case.query.model_dump(mode="json") for case in cases]
    )
    _json_write(root / "private_gold.json", [case.gold.model_dump(mode="json") for case in cases])
    query_hash = hashlib.sha256((root / "queries_public.json").read_bytes()).hexdigest()

    retrieval_config = RetrievalConfig()
    torch = importlib.import_module("torch")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    text = _MeasuredTextEncoder(
        BGETextEncoder(
            model_cache=args.model_cache,
            batch_size=retrieval_config.text_batch_size,
            maximum_tokens=retrieval_config.text_encoder_max_tokens,
        )
    )
    visual = _MeasuredVisualEncoder(
        Siglip2VisualEncoder(
            model_cache=args.model_cache,
            batch_size=retrieval_config.visual_batch_size,
        )
    )
    builder = RetrievalIndexBuilder(root / "index")
    build_started = time.perf_counter()
    memory = builder.build(
        world_states=tuple(item.world_state for item in perceptions.values()),
        keyframe_assets=_keyframes(ingestions),
        config=retrieval_config,
        text_encoder=text,
        visual_encoder=visual,
    )
    build_ms = (time.perf_counter() - build_started) * 1000
    build_peak_allocated = torch.cuda.max_memory_allocated()
    build_peak_reserved = torch.cuda.max_memory_reserved()
    cached_memory = builder.build(
        world_states=tuple(item.world_state for item in perceptions.values()),
        keyframe_assets=_keyframes(ingestions),
        config=retrieval_config,
        text_encoder=text,
        visual_encoder=visual,
    )
    if not all(cached_memory.cache_hits.values()):
        raise RuntimeError("second M2B index build did not hit every component cache")
    scene_clips, clip_cache = _scene_clips(
        root=root,
        records=records,
        ingestions=ingestions,
        ffmpeg=args.ffmpeg,
    )

    variants: tuple[tuple[str, RoutingStrategy, bool, bool, tuple[RetrievalChannel, ...]], ...] = (
        ("uniform_rrf", "uniform", False, False, ()),
        ("static_weighted_rrf", "static_weighted", False, False, ()),
        ("query_aware_rrf", "query_aware", False, False, ()),
        ("query_aware_rerank", "query_aware", True, False, ()),
        ("query_aware_rerank_native", "query_aware", True, True, ()),
        (
            "ablation_without_bm25",
            "query_aware",
            True,
            False,
            ("bm25_transcript", "bm25_ocr", "bm25_structured", "bm25_combined"),
        ),
        ("ablation_without_dense", "query_aware", True, False, ("dense_text",)),
        ("ablation_without_visual", "query_aware", True, False, ("visual",)),
    )
    all_results: dict[str, dict[str, AdaptiveRetrievalResult]] = {}
    for name, strategy, rerank, native, disabled in variants:
        all_results[name] = _run_variant(
            name=name,
            strategy=strategy,
            evidence_reranking=rerank,
            native_video=native,
            disabled_channels=disabled,
            memory=memory,
            cases=cases,
            text=text,
            visual=visual,
            scene_clips=scene_clips,
            model_cache=args.model_cache,
            root=root,
        )

    gold = {case.gold.query_id: case.gold for case in cases}
    summaries: dict[str, dict[str, Any]] = {}
    breakdowns: dict[str, object] = {}
    latencies: dict[str, object] = {}
    for name, results in all_results.items():
        responses = {key: item.response for key, item in results.items()}
        summary, query_metrics = evaluate_m2b_method(name, responses, gold)
        summaries[name] = summary.model_dump(mode="json")
        breakdowns[name] = _breakdown(name, query_metrics, gold)
        latencies[name] = _latency(responses)
        _json_write(
            root / "metrics" / f"{name}.json",
            {
                "summary": summary.model_dump(mode="json"),
                "breakdown": breakdowns[name],
                "queries": [item.model_dump(mode="json") for item in query_metrics],
            },
        )

    before_native = {key: item.response for key, item in all_results["query_aware_rerank"].items()}
    native_results = all_results["query_aware_rerank_native"]
    effects = native_effect_summary(before_native, native_results, gold)
    call_analysis = _native_call_analysis(native_results, gold)
    failures = attribute_failures(
        uniform={key: item.response for key, item in all_results["uniform_rrf"].items()},
        routed={key: item.response for key, item in all_results["query_aware_rrf"].items()},
        reranked=before_native,
        native=native_results,
        gold=gold,
    )
    _json_write(
        root / "failure_attribution.json", [item.model_dump(mode="json") for item in failures]
    )

    repeat: dict[str, object] = {}
    for name, strategy, rerank, native, disabled in variants:
        repeated = _run_variant(
            name=f"repeat_{name}",
            strategy=strategy,
            evidence_reranking=rerank,
            native_video=native,
            disabled_channels=disabled,
            memory=memory,
            cases=cases,
            text=text,
            visual=visual,
            scene_clips=scene_clips,
            model_cache=args.model_cache,
            root=root,
        )
        mismatches = sum(
            [
                (item.video_id, item.scene_id)
                for item in all_results[name][query_id].response.candidates
            ]
            != [(item.video_id, item.scene_id) for item in repeated[query_id].response.candidates]
            for query_id in repeated
        )
        repeat[name] = {"ranking_mismatches": mismatches}
    if any(item["ranking_mismatches"] for item in repeat.values()):  # type: ignore[index]
        raise RuntimeError("M2B repeated ranking differs")

    router = router_confusion(all_results["query_aware_rrf"], gold)
    invoked = effects.invoked_queries
    extra_latency = {
        query_id: native_results[query_id].response.latency_ms - before_native[query_id].latency_ms
        for query_id in native_results
    }
    summary_payload = {
        "status": "completed",
        "heldout_query_count": len(cases),
        "heldout_query_sha256": query_hash,
        "source_groups": len(records),
        "multi_scene_videos": len(records),
        "corpus_scenes": len(memory.documents),
        "query_type_counts": dict(Counter(case.gold.query_type for case in cases)),
        "hard_negative_categories": dict(
            Counter(
                case.gold.hard_negative_category
                for case in cases
                if case.gold.hard_negative_category is not None
            )
        ),
        "models": {
            "text": retrieval_config.text_encoder.model_dump(mode="json"),
            "visual": retrieval_config.visual_encoder.model_dump(mode="json"),
            "native_video": PerceptionConfig().qwen.model_dump(mode="json"),
        },
        "index": {
            "manifest": memory.manifest.model_dump(mode="json"),
            "first_cache_hits": memory.cache_hits,
            "second_cache_hits": cached_memory.cache_hits,
            "build_ms": build_ms,
            "size_bytes": _index_size(root / "index"),
            "text_embedding_ms": text.elapsed_ms,
            "text_embedding_items": text.items,
            "text_embedding_items_per_second": (
                text.items / (text.elapsed_ms / 1000) if text.elapsed_ms else 0.0
            ),
            "visual_embedding_ms": visual.elapsed_ms,
            "visual_embedding_items": visual.items,
            "visual_embedding_items_per_second": (
                visual.items / (visual.elapsed_ms / 1000) if visual.elapsed_ms else 0.0
            ),
            "build_peak_allocated_bytes": build_peak_allocated,
            "build_peak_reserved_bytes": build_peak_reserved,
            "scene_clip_cache": clip_cache,
        },
        "metrics": summaries,
        "breakdowns": breakdowns,
        "latency": latencies,
        "router": router,
        "native_video": {
            "invocation_rate": invoked / len(cases),
            "average_calls_per_query": effects.qwen_calls / len(cases),
            "average_calls_per_invoked_query": effects.qwen_calls / max(invoked, 1),
            "corrected": effects.corrected,
            "no_effect": effects.no_effect,
            "harmed": effects.harmed,
            "extra_latency_mean_all_queries_ms": statistics.fmean(extra_latency.values()),
            "extra_latency_mean_invoked_ms": statistics.fmean(
                extra_latency[key]
                for key, result in native_results.items()
                if result.native_verifications
            )
            if invoked
            else 0.0,
            "quality_gain_recall1_per_call": (
                (
                    summaries["query_aware_rerank_native"]["recall_at_1"]
                    - summaries["query_aware_rerank"]["recall_at_1"]
                )
                / max(effects.qwen_calls / len(cases), 1e-9)
            ),
            "call_analysis": call_analysis,
        },
        "failure_attribution": dict(Counter(item.attribution for item in failures)),
        "reproducibility": repeat,
        "runtime": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "total_ms": (time.perf_counter() - started) * 1000,
        },
    }
    _json_write(root / "experiment_summary.json", summary_payload)
    _json_write(root / "index_manifest.json", memory.manifest.model_dump(mode="json"))
    print(json.dumps(summary_payload, ensure_ascii=False, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "evaluate"), required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m2b"))
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.phase == "prepare":
        return _prepare(args)
    return _evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
