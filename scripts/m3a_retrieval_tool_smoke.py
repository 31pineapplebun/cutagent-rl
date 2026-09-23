"""Run search_video -> trim_video -> validate_media with the real frozen M2B encoders."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

from cutagent.retrieval.adaptive import AdaptiveHybridRetriever
from cutagent.retrieval.encoders import BGETextEncoder, Siglip2VisualEncoder
from cutagent.retrieval.fusion import WeightedRankFusion
from cutagent.retrieval.index import KeyframeAsset, RetrievalIndexBuilder
from cutagent.retrieval.protocols import MultimodalRetriever
from cutagent.retrieval.retrievers import DenseTextRetriever, SparseRetriever, VisualRetriever
from cutagent.schemas.media import IngestionResult
from cutagent.schemas.perception import PerceptionResult
from cutagent.schemas.retrieval import (
    AdaptiveRetrievalConfig,
    RetrievalChannel,
    RetrievalConfig,
    RetrievalQuery,
)
from cutagent.schemas.tools import ToolExecutionContext
from cutagent.tools.factory import create_m3a_registry


def _artifact_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("M3A retrieval smoke requires local file artifacts")
    path_text = unquote(parsed.path)
    if len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":":
        path_text = path_text[1:]
    return Path(path_text).resolve(strict=True)


def _load_prepared(
    root: Path,
) -> tuple[tuple[IngestionResult, ...], tuple[PerceptionResult, ...]]:
    ingestion_paths = tuple(sorted((root / "ingestion").glob("*.json")))
    perception_paths = tuple(sorted((root / "perception").glob("*.json")))
    if not ingestion_paths or len(ingestion_paths) != len(perception_paths):
        raise RuntimeError("M2B prepared public ingestion/perception artifacts are incomplete")
    ingestions = tuple(
        IngestionResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in ingestion_paths
    )
    perceptions = tuple(
        PerceptionResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in perception_paths
    )
    if {item.video.video_id for item in ingestions} != {
        item.world_state.video_id for item in perceptions
    }:
        raise RuntimeError("M2B public ingestion/perception video identities differ")
    return ingestions, perceptions


def _keyframes(ingestions: tuple[IngestionResult, ...]) -> tuple[KeyframeAsset, ...]:
    assets: dict[str, KeyframeAsset] = {}
    for ingestion in ingestions:
        for keyframe in ingestion.keyframes:
            assets[keyframe.artifact.artifact_id] = KeyframeAsset(
                artifact=keyframe.artifact,
                path=_artifact_path(keyframe.artifact.uri),
            )
    return tuple(assets[key] for key in sorted(assets))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2b-root", type=Path, default=Path("artifacts/m2b"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m3a"))
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    started = time.perf_counter()
    m2b_root = args.m2b_root.resolve()
    artifact_root = args.artifact_root.resolve()
    ingestions, perceptions = _load_prepared(m2b_root)
    torch = importlib.import_module("torch")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    retrieval_config = RetrievalConfig()
    text = BGETextEncoder(
        model_cache=args.model_cache.resolve(),
        batch_size=retrieval_config.text_batch_size,
        maximum_tokens=retrieval_config.text_encoder_max_tokens,
    )
    visual = Siglip2VisualEncoder(
        model_cache=args.model_cache.resolve(),
        batch_size=retrieval_config.visual_batch_size,
    )
    memory = RetrievalIndexBuilder(m2b_root / "index").build(
        world_states=tuple(item.world_state for item in perceptions),
        keyframe_assets=_keyframes(ingestions),
        config=retrieval_config,
        text_encoder=text,
        visual_encoder=visual,
    )
    components: dict[RetrievalChannel, MultimodalRetriever] = {
        "bm25_transcript": SparseRetriever("transcript"),
        "bm25_ocr": SparseRetriever("ocr"),
        "bm25_structured": SparseRetriever("structured_semantic"),
        "bm25_combined": SparseRetriever("combined"),
        "dense_text": DenseTextRetriever(text),
        "visual": VisualRetriever(visual),
    }
    adaptive = AdaptiveHybridRetriever(
        config=AdaptiveRetrievalConfig(native_video_enabled=False),
        fusion=WeightedRankFusion(components),
        strategy="query_aware",
        evidence_reranking=True,
        native_video=False,
    )
    registry = create_m3a_registry(
        root=artifact_root / "retrieval_tool_runtime",
        retriever=adaptive,
        retrieval_memory=memory,
        ffmpeg_executable=args.ffmpeg,
        ffprobe_executable=args.ffprobe,
    )
    public_queries = [
        RetrievalQuery.model_validate(item)
        for item in json.loads((m2b_root / "queries_public.json").read_text(encoding="utf-8"))
    ]
    if not public_queries:
        raise RuntimeError("M2B public query set is empty")
    selected_query = next(
        (item for item in public_queries if "moving" in item.text.casefold()),
        public_queries[0],
    )
    ingestion_by_video = {item.video.video_id: item for item in ingestions}
    context = ToolExecutionContext(
        execution_id="m3a-real-retrieval-sequence",
        allowed_output_root_id="m3a-retrieval-smoke",
        allowed_artifact_ids=(),
        allowed_capabilities=(
            "retrieval.read",
            "media.inspect",
            "media.decode",
            "media.write",
        ),
        timeout_ms=180_000,
        maximum_output_bytes=2_000_000_000,
    )
    search = registry.execute(
        {
            "tool_name": "search_video",
            "tool_call_id": "real-search",
            "arguments": {
                "query": selected_query.text,
                "top_k": 5,
                "video_id": selected_query.video_id,
                "approximate_time_range": (
                    selected_query.approximate_time_range.model_dump(mode="json")
                    if selected_query.approximate_time_range is not None
                    else None
                ),
                "required_evidence_types": list(selected_query.required_evidence_types),
            },
        },
        context,
    )
    if search.observation.status != "success":
        raise RuntimeError(search.observation.model_dump_json(indent=2))
    response = cast(dict[str, Any], search.observation.details["response"])
    candidates = cast(list[dict[str, Any]], response["candidates"])
    if not candidates:
        raise RuntimeError("real M2B search returned no candidates")
    top = candidates[0]
    source_ingestion = ingestion_by_video[cast(str, top["video_id"])]
    source_path = _artifact_path(source_ingestion.video.source.uri)
    source_ref = registry.artifact_store.import_file(source_path, media_type="video/mp4")
    context = context.model_copy(update={"allowed_artifact_ids": (source_ref.artifact_id,)})
    time_range = cast(dict[str, int], top["time_range"])
    trim = registry.execute(
        {
            "tool_name": "trim_video",
            "tool_call_id": "real-search-trim",
            "arguments": {
                "input_artifact_id": source_ref.artifact_id,
                "time_range": {
                    "start_ms": time_range["start_ms"],
                    "end_ms": time_range["end_ms"],
                },
            },
        },
        context,
    )
    if trim.observation.status != "success":
        raise RuntimeError(trim.observation.model_dump_json(indent=2))
    output_id = trim.observation.artifacts[0].artifact_id
    validation = registry.execute(
        {
            "tool_name": "validate_media",
            "tool_call_id": "real-search-validate",
            "arguments": {"input_artifact_id": output_id},
        },
        context,
    )
    if validation.observation.status != "success":
        raise RuntimeError(validation.observation.model_dump_json(indent=2))
    plan = cast(dict[str, Any], search.observation.details["retrieval_plan"])
    if plan["native_video_policy"] != "disabled":
        raise RuntimeError("M3A retrieval unexpectedly enabled native-video verification")
    if trim.trace.output_artifact is None:
        raise RuntimeError("successful trim trace omitted its output artifact")
    report = {
        "query": selected_query.model_dump(mode="json"),
        "top_candidate": top,
        "retrieval_plan": plan,
        "search_status": search.observation.status,
        "trim_status": trim.observation.status,
        "validate_status": validation.observation.status,
        "output_artifact": trim.trace.output_artifact.model_dump(mode="json"),
        "index_id": memory.manifest.index_id,
        "index_cache_hits": memory.cache_hits,
        "text_encoder": {"model_id": text.model_id, "revision": text.revision},
        "visual_encoder": {"model_id": visual.model_id, "revision": visual.revision},
        "native_video_policy": "disabled",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "wall_time_ms": round((time.perf_counter() - started) * 1000),
    }
    _write_json(artifact_root / "retrieval_tool_smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
