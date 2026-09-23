"""Run paired M1B perception on the frozen licensed real-video clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from cutagent_evaluation.licensed_media import LicensedMediaProvenance

from cutagent.ingestion.pipeline import VideoIngestionPipeline
from cutagent.perception.pipeline import MultimodalPerceptionPipeline
from cutagent.schemas.media import IngestionConfig, KeyframeExtractionConfig
from cutagent.schemas.perception import PerceptionConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _performance(result: Any, operation_prefix: str) -> dict[str, int | None]:
    records = [item for item in result.performance if item.operation.startswith(operation_prefix)]
    return {
        "latency_ms": sum(item.latency_ms for item in records),
        "peak_allocated_bytes": max(
            (item.peak_allocated_bytes or 0 for item in records), default=0
        ),
        "peak_reserved_bytes": max((item.peak_reserved_bytes or 0 for item in records), default=0),
        "frames": sum(item.frames for item in records),
        "input_tokens": sum(item.input_tokens or 0 for item in records),
    }


def _claim_view(result: Any) -> dict[str, object]:
    return {
        "transcripts": [span.model_dump(mode="json") for span in result.transcript_spans],
        "ocr": [span.model_dump(mode="json") for span in result.ocr_spans],
        "entities": [
            entity.model_dump(mode="json")
            for observation in result.visual_observations
            for entity in observation.entities
        ],
        "actions": [
            action.model_dump(mode="json")
            for observation in result.visual_observations
            for action in observation.actions
        ],
        "events": [event.model_dump(mode="json") for event in result.temporal_events],
        "summaries": [observation.scene_summary for observation in result.visual_observations],
        "uncertainties": [
            uncertainty
            for observation in result.visual_observations
            for uncertainty in observation.uncertainties
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("artifacts/m1b_5/real_media/provenance.json"),
    )
    parser.add_argument("--work-root", type=Path, default=Path("artifacts/m1b_5/real_validation"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/m1b_5/real_validation.json"))
    parser.add_argument("--maximum-new-tokens", type=int, default=768, choices=(384, 768, 1024))
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    model_cache = args.model_cache.resolve()
    if model_cache == repository_root or repository_root in model_cache.parents:
        parser.error("--model-cache must be outside the Git repository")
    provenance_path = args.provenance.resolve(strict=True)
    provenance_root = provenance_path.parent
    provenance = LicensedMediaProvenance.model_validate_json(
        provenance_path.read_text(encoding="utf-8")
    )
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    ingestion_pipeline = VideoIngestionPipeline(cache_root=work_root / "ingestion-cache")
    perception_pipeline = MultimodalPerceptionPipeline(
        cache_root=work_root / "perception-cache",
        model_cache=model_cache,
    )
    rows: list[dict[str, object]] = []
    started = time.perf_counter_ns()
    for source_record in provenance.sources:
        for clip_record in source_record.clips:
            clip_path = provenance_root / clip_record.relative_path
            if _sha256(clip_path) != clip_record.sha256:
                raise RuntimeError(f"derived clip hash mismatch: {clip_record.clip_id}")
            ingestion = ingestion_pipeline.ingest(
                clip_path,
                config=IngestionConfig(
                    scene_threshold=0.99,
                    minimum_scene_duration_ms=300,
                    keyframes=KeyframeExtractionConfig(strategy="uniform", frames_per_scene=3),
                ),
            )
            for mode in ("keyframes", "native_video"):
                config = PerceptionConfig(
                    visual_mode=mode,
                    asr_enabled=mode == "keyframes",
                    native_video_fps=2.0,
                    maximum_new_tokens=args.maximum_new_tokens,
                )
                row: dict[str, object]
                try:
                    first_started = time.perf_counter_ns()
                    first = perception_pipeline.run(
                        clip_path,
                        ingestion=ingestion,
                        config=config,
                    )
                    first_wall_ms = (time.perf_counter_ns() - first_started) // 1_000_000
                    second_started = time.perf_counter_ns()
                    second = perception_pipeline.run(
                        clip_path,
                        ingestion=ingestion,
                        config=config,
                    )
                    second_wall_ms = (time.perf_counter_ns() - second_started) // 1_000_000
                    heavy_records = [
                        record
                        for record in second.cache_records
                        if record.operation in {"asr", "vlm_perception"}
                    ]
                    row = {
                        "source_id": source_record.source_id,
                        "clip_id": clip_record.clip_id,
                        "mode": mode,
                        "status": "passed",
                        "source_license": source_record.license,
                        "source_sha256": source_record.sha256,
                        "clip_sha256": clip_record.sha256,
                        "source_time_range_ms": [
                            clip_record.start_ms,
                            clip_record.end_ms,
                        ],
                        "duration_ms": ingestion.video.duration_ms,
                        "scene_count": len(ingestion.scenes),
                        "first_wall_ms": first_wall_ms,
                        "second_wall_ms": second_wall_ms,
                        "second_run_all_heavy_cache_hits": bool(heavy_records)
                        and all(record.hit for record in heavy_records),
                        "world_state_replay_identical": first.world_state == second.world_state,
                        "qwen": _performance(first, "qwen_"),
                        "whisper": _performance(first, "whisper"),
                        "claim_counts": {
                            "transcript_spans": len(first.transcript_spans),
                            "ocr_spans": len(first.ocr_spans),
                            "entities": sum(
                                len(item.entities) for item in first.visual_observations
                            ),
                            "actions": sum(len(item.actions) for item in first.visual_observations),
                            "events": len(first.temporal_events),
                        },
                        "claims": _claim_view(first),
                    }
                    result_path = work_root / "results" / mode / f"{clip_record.clip_id}.json"
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(first.model_dump_json(indent=2), encoding="utf-8")
                except Exception as error:
                    row = {
                        "source_id": source_record.source_id,
                        "clip_id": clip_record.clip_id,
                        "mode": mode,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                rows.append(row)
                row_path = work_root / "rows" / mode / f"{clip_record.clip_id}.json"
                row_path.parent.mkdir(parents=True, exist_ok=True)
                row_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"real clip={clip_record.clip_id} mode={mode} status={row['status']}")
    passed = [row for row in rows if row.get("status") == "passed"]
    report = {
        "schema_version": "1.0.0",
        "status": "completed",
        "dataset_id": provenance.dataset_id,
        "source_count": len(provenance.sources),
        "clip_count": provenance.clip_count,
        "paired_attempt_count": len(rows),
        "structured_output_validity": len(passed) / len(rows),
        "models": {
            "qwen": PerceptionConfig().qwen.model_dump(mode="json"),
            "whisper": PerceptionConfig().whisper.model_dump(mode="json"),
        },
        "configs": {
            "keyframes": PerceptionConfig(
                visual_mode="keyframes",
                asr_enabled=True,
                maximum_new_tokens=args.maximum_new_tokens,
            ).model_dump(mode="json"),
            "native_video": PerceptionConfig(
                visual_mode="native_video",
                asr_enabled=False,
                maximum_new_tokens=args.maximum_new_tokens,
            ).model_dump(mode="json"),
        },
        "rows": rows,
        "processing_time_ms": (time.perf_counter_ns() - started) // 1_000_000,
        "quantitative_ground_truth_available": False,
        "manual_review_required": True,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"M1B.5 real validation completed sources={len(provenance.sources)} "
        f"clips={provenance.clip_count} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
