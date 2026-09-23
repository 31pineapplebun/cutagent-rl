"""Run the frozen M2A retrieval baselines on real M1B/M1B.5 outputs."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from cutagent_evaluation.retrieval import evaluate_method, summarize_metrics
from cutagent_evaluation.retrieval_dataset import (
    build_frozen_development_cases,
    load_m1b_gold,
    load_motion_gold,
    load_perception_results,
)

from cutagent.core.artifacts import ArtifactRef
from cutagent.retrieval.encoders import BGETextEncoder, Siglip2VisualEncoder
from cutagent.retrieval.index import KeyframeAsset, RetrievalIndexBuilder
from cutagent.retrieval.protocols import FloatMatrix, MultimodalRetriever
from cutagent.retrieval.retrievers import (
    DenseTextRetriever,
    FusionRetriever,
    SparseRetriever,
    VisualRetriever,
)
from cutagent.schemas.media import KeyframeRef
from cutagent.schemas.perception import PerceptionResult
from cutagent.schemas.retrieval import RetrievalConfig, RetrievalResponse


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _artifact_path(artifact: ArtifactRef) -> Path:
    parsed = urlparse(artifact.uri)
    if parsed.scheme != "file":
        raise ValueError(f"M2A visual evidence must use a local file URI: {artifact.uri}")
    return Path(unquote(parsed.path))


def _discover_results(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return tuple(sorted(path for path in root.glob("*.json") if path.name != "summary.json"))


def _discover_keyframes(roots: tuple[Path, ...]) -> dict[str, KeyframeAsset]:
    assets: dict[str, KeyframeAsset] = {}
    for root in roots:
        for path in sorted(root.rglob("result.json")):
            if path.parent.parent.name != "keyframe_extract":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                continue
            for item in payload:
                keyframe = KeyframeRef.model_validate(item)
                asset = KeyframeAsset(
                    artifact=keyframe.artifact,
                    path=_artifact_path(keyframe.artifact),
                )
                existing = assets.get(keyframe.artifact.artifact_id)
                if existing is not None and existing.artifact.sha256 != asset.artifact.sha256:
                    raise ValueError(
                        f"conflicting keyframe artifact ID: {keyframe.artifact.artifact_id}"
                    )
                assets[keyframe.artifact.artifact_id] = asset
    return assets


def _required_keyframes(
    results: tuple[PerceptionResult, ...], assets: dict[str, KeyframeAsset]
) -> tuple[KeyframeAsset, ...]:
    required_ids = {
        reference.artifact_id
        for result in results
        for scene in result.world_state.scenes
        for reference in scene.keyframe_evidence
    }
    missing = sorted(required_ids - assets.keys())
    if missing:
        raise ValueError(f"missing {len(missing)} keyframe files; first={missing[0]}")
    return tuple(assets[artifact_id] for artifact_id in sorted(required_ids))


def _gpu_reset() -> Any:
    torch = importlib.import_module("torch")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return torch


def _latency_distribution(responses: dict[str, RetrievalResponse]) -> dict[str, float]:
    values = sorted(response.latency_ms for response in responses.values())
    p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": p95,
        "max_ms": max(values),
    }


def _index_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


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
        value = self.backend.encode(texts)
        self.elapsed_ms += (time.perf_counter() - started) * 1000
        self.items += len(texts)
        return value


class _MeasuredVisualEncoder:
    def __init__(self, backend: Siglip2VisualEncoder) -> None:
        self.backend = backend
        self.image_elapsed_ms = 0.0
        self.image_items = 0

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    @property
    def revision(self) -> str:
        return self.backend.revision

    def encode_images(self, paths: tuple[Path, ...]) -> FloatMatrix:
        started = time.perf_counter()
        value = self.backend.encode_images(paths)
        self.image_elapsed_ms += (time.perf_counter() - started) * 1000
        self.image_items += len(paths)
        return value

    def encode_queries(self, texts: tuple[str, ...]) -> FloatMatrix:
        return self.backend.encode_queries(texts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1b-results", type=Path, required=True)
    parser.add_argument("--motion-results", type=Path, required=True)
    parser.add_argument("--real-results", type=Path, required=True)
    parser.add_argument("--m1b-gold", type=Path, required=True)
    parser.add_argument("--motion-gold", type=Path, required=True)
    parser.add_argument("--ingestion-cache", action="append", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/m2a/experiment"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    generated_results = {
        **load_perception_results(_discover_results(args.m1b_results)),
        **load_perception_results(_discover_results(args.motion_results)),
    }
    real_results = load_perception_results(_discover_results(args.real_results))
    overlap = generated_results.keys() & real_results.keys()
    if overlap:
        raise ValueError(f"duplicate generated/real case IDs: {sorted(overlap)}")
    all_results_by_case = {**generated_results, **real_results}
    all_results = tuple(all_results_by_case[key] for key in sorted(all_results_by_case))
    frozen_cases = build_frozen_development_cases(
        m1b_gold=load_m1b_gold(args.m1b_gold),
        motion_gold=load_motion_gold(args.motion_gold),
        results=generated_results,
    )
    _json_write(
        args.output_root / "queries_public.json",
        [case.query.model_dump(mode="json") for case in frozen_cases],
    )
    _json_write(
        args.output_root / "private_gold.json",
        [case.gold.model_dump(mode="json") for case in frozen_cases],
    )

    keyframe_assets = _required_keyframes(
        all_results,
        _discover_keyframes(tuple(args.ingestion_cache)),
    )
    config = RetrievalConfig()
    torch = _gpu_reset()
    load_started = time.perf_counter()
    text_encoder = _MeasuredTextEncoder(
        BGETextEncoder(
            model_cache=args.model_cache,
            batch_size=config.text_batch_size,
            maximum_tokens=config.text_encoder_max_tokens,
        )
    )
    text_load_ms = (time.perf_counter() - load_started) * 1000
    load_started = time.perf_counter()
    visual_encoder = _MeasuredVisualEncoder(
        Siglip2VisualEncoder(
            model_cache=args.model_cache,
            batch_size=config.visual_batch_size,
        )
    )
    visual_load_ms = (time.perf_counter() - load_started) * 1000
    model_load_peak = torch.cuda.max_memory_allocated()

    builder = RetrievalIndexBuilder(args.index_root)
    torch.cuda.reset_peak_memory_stats()
    index_started = time.perf_counter()
    memory = builder.build(
        world_states=tuple(result.world_state for result in all_results),
        keyframe_assets=keyframe_assets,
        config=config,
        text_encoder=text_encoder,
        visual_encoder=visual_encoder,
    )
    index_build_ms = (time.perf_counter() - index_started) * 1000
    index_peak_allocated = torch.cuda.max_memory_allocated()
    index_peak_reserved = torch.cuda.max_memory_reserved()
    index_text_embedding_ms = text_encoder.elapsed_ms
    index_text_items = text_encoder.items
    index_visual_embedding_ms = visual_encoder.image_elapsed_ms
    index_visual_items = visual_encoder.image_items
    cached_memory = builder.build(
        world_states=tuple(result.world_state for result in all_results),
        keyframe_assets=keyframe_assets,
        config=config,
        text_encoder=text_encoder,
        visual_encoder=visual_encoder,
    )
    if cached_memory.documents != memory.documents:
        raise RuntimeError("cached index documents differ from first build")
    if not all(cached_memory.cache_hits.values()):
        raise RuntimeError("second identical index build did not hit every component cache")

    bm25_combined = SparseRetriever("combined")
    dense = DenseTextRetriever(text_encoder)
    visual = VisualRetriever(visual_encoder)
    methods: dict[str, MultimodalRetriever] = {
        "bm25_transcript": SparseRetriever("transcript"),
        "bm25_ocr": SparseRetriever("ocr"),
        "bm25_structured": SparseRetriever("structured_semantic"),
        "bm25_combined": bm25_combined,
        "dense_text": dense,
        "visual": visual,
        "bm25_dense_rrf": FusionRetriever({"bm25": bm25_combined, "dense": dense}),
        "dense_visual_rrf": FusionRetriever({"dense": dense, "visual": visual}),
        "all_rrf": FusionRetriever({"bm25": bm25_combined, "dense": dense, "visual": visual}),
    }
    gold_by_query = {case.gold.query_id: case.gold for case in frozen_cases}
    summaries: dict[str, object] = {}
    breakdowns: dict[str, object] = {}
    failures: dict[str, object] = {}
    latency: dict[str, object] = {}
    raw_root = args.output_root / "responses"
    for method_name, retriever in methods.items():
        responses = {
            case.query.query_id: retriever.search(case.query, memory) for case in frozen_cases
        }
        summary, query_metrics = evaluate_method(method_name, responses, gold_by_query)
        summaries[method_name] = summary.model_dump(mode="json")
        method_breakdown: dict[str, object] = {}
        for query_type in ("speech", "ocr", "entity", "semantic", "action", "hard_negative"):
            subset = tuple(item for item in query_metrics if item.query_type == query_type)
            method_breakdown[query_type] = summarize_metrics(
                f"{method_name}:{query_type}", subset
            ).model_dump(mode="json")
        normal = tuple(item for item in query_metrics if not item.hard_negative)
        hard = tuple(item for item in query_metrics if item.hard_negative)
        method_breakdown["normal"] = summarize_metrics(f"{method_name}:normal", normal).model_dump(
            mode="json"
        )
        method_breakdown["hard_negative_subset"] = summarize_metrics(
            f"{method_name}:hard_negative_subset", hard
        ).model_dump(mode="json")
        breakdowns[method_name] = method_breakdown
        failures[method_name] = dict(
            Counter(item.failure_type for item in query_metrics if item.failure_type is not None)
        )
        latency[method_name] = _latency_distribution(responses)
        _json_write(
            raw_root / f"{method_name}.json",
            {
                "responses": {
                    key: value.model_dump(mode="json") for key, value in responses.items()
                },
                "query_metrics": [item.model_dump(mode="json") for item in query_metrics],
            },
        )

    public_queries_json = (args.output_root / "queries_public.json").read_bytes()
    import hashlib

    query_set_sha256 = hashlib.sha256(public_queries_json).hexdigest()
    summary_payload = {
        "status": "completed",
        "frozen_query_count": len(frozen_cases),
        "frozen_query_sha256": query_set_sha256,
        "query_type_counts": dict(Counter(case.gold.query_type for case in frozen_cases)),
        "hard_negative_count": sum(case.gold.hard_negative for case in frozen_cases),
        "corpus": {
            "generated_perception_results": len(generated_results),
            "licensed_real_perception_results": len(real_results),
            "videos": len(all_results),
            "scenes": len(memory.documents),
            "keyframes": len(memory.visual_rows),
        },
        "models": {
            "text": config.text_encoder.model_dump(mode="json"),
            "visual": config.visual_encoder.model_dump(mode="json"),
        },
        "model_load": {
            "text_load_ms": text_load_ms,
            "visual_load_ms": visual_load_ms,
            "peak_allocated_bytes": model_load_peak,
        },
        "index": {
            "manifest": memory.manifest.model_dump(mode="json"),
            "first_build_cache_hits": memory.cache_hits,
            "second_build_cache_hits": cached_memory.cache_hits,
            "build_ms": index_build_ms,
            "size_bytes": _index_size(args.index_root),
            "peak_allocated_bytes": index_peak_allocated,
            "peak_reserved_bytes": index_peak_reserved,
            "text_embedding_ms": index_text_embedding_ms,
            "visual_embedding_ms": index_visual_embedding_ms,
            "text_documents_per_second": index_text_items
            / max(index_text_embedding_ms / 1000, 1e-9),
            "keyframes_per_second": index_visual_items
            / max(index_visual_embedding_ms / 1000, 1e-9),
        },
        "metrics": summaries,
        "breakdowns": breakdowns,
        "failures": failures,
        "latency": latency,
        "runtime": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "total_ms": (time.perf_counter() - started) * 1000,
        },
    }
    _json_write(args.output_root / "experiment_summary.json", summary_payload)
    _json_write(args.output_root / "index_manifest.json", memory.manifest.model_dump(mode="json"))
    print(json.dumps(summary_payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
