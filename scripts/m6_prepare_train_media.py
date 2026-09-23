"""Prepare authentic M1A/M1B public evidence for CutAgentBench train sources."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from cutagent.ingestion.pipeline import VideoIngestionPipeline
from cutagent.perception.pipeline import MultimodalPerceptionPipeline
from cutagent.schemas.media import IngestionConfig, IngestionResult, KeyframeExtractionConfig
from cutagent.schemas.perception import PerceptionConfig, PerceptionResult
from cutagent.schemas.task_input import TaskInput


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("M6 train preparation requires immutable local media")
    value = unquote(parsed.path)
    if len(value) >= 3 and value[0] == "/" and value[2] == ":":
        value = value[1:]
    return Path(value).resolve(strict=True)


def _canonical_ingestion(value: IngestionResult, task: TaskInput) -> IngestionResult:
    payload = value.model_dump(mode="json")
    payload["video"]["video_id"] = task.video_ref.artifact_id
    payload["video"]["source"] = task.video_ref.model_dump(mode="json")
    for scene in payload["scenes"]:
        scene["parent_video_id"] = task.video_ref.artifact_id
    for keyframe in payload["keyframes"]:
        keyframe["parent_video_id"] = task.video_ref.artifact_id
    return IngestionResult.model_validate(payload)


def _canonical_perception(value: PerceptionResult, task: TaskInput) -> PerceptionResult:
    payload = value.model_dump(mode="json")
    payload["source_video_id"] = task.video_ref.artifact_id
    payload["world_state"]["video_id"] = task.video_ref.artifact_id
    return PerceptionResult.model_validate(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path("artifacts/m5a/cutagentbench_v0.1")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m6"))
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    benchmark_root = args.benchmark_root.resolve(strict=True)
    raw = json.loads((benchmark_root / "public/train.json").read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("train public manifest must be a list")
    tasks = tuple(TaskInput.model_validate(item) for item in raw)
    by_video: dict[str, TaskInput] = {}
    for task in tasks:
        by_video.setdefault(task.video_ref.artifact_id, task)
    selected = tuple(by_video[key] for key in sorted(by_video))
    ingestion_config = IngestionConfig(
        scene_threshold=0.07,
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
    root = args.artifact_root.resolve()
    output = root / "prepared/train"
    ingestion_pipeline = VideoIngestionPipeline(
        cache_root=root / "media_cache/ingestion",
        ffmpeg_executable=args.ffmpeg,
        ffprobe_executable=args.ffprobe,
    )
    perception_pipeline = MultimodalPerceptionPipeline(
        cache_root=root / "media_cache/perception",
        model_cache=args.model_cache.resolve(strict=True),
        ffmpeg_executable=args.ffmpeg,
    )
    started = time.perf_counter_ns()
    records: list[dict[str, object]] = []
    for index, task in enumerate(selected, start=1):
        ingestion_path = output / "ingestion" / f"{task.video_ref.artifact_id}.json"
        perception_path = output / "perception" / f"{task.video_ref.artifact_id}.json"
        resumed = ingestion_path.is_file() and perception_path.is_file()
        if resumed:
            ingestion = IngestionResult.model_validate_json(
                ingestion_path.read_text(encoding="utf-8")
            )
            perception = PerceptionResult.model_validate_json(
                perception_path.read_text(encoding="utf-8")
            )
            resumed = (
                ingestion.config == ingestion_config and perception.config == perception_config
            )
        if not resumed:
            source = _path(task.video_ref.uri)
            raw_ingestion = ingestion_pipeline.ingest(source, config=ingestion_config)
            ingestion = _canonical_ingestion(raw_ingestion, task)
            perception = _canonical_perception(
                perception_pipeline.run(
                    source,
                    ingestion=raw_ingestion,
                    config=perception_config,
                ),
                task,
            )
            _write(ingestion_path, ingestion.model_dump(mode="json"))
            _write(perception_path, perception.model_dump(mode="json"))
        records.append(
            {
                "video_id": task.video_ref.artifact_id,
                "scene_count": len(ingestion.scenes),
                "keyframe_count": len(ingestion.keyframes),
                "transcript_span_count": len(perception.transcript_spans),
                "ocr_span_count": len(perception.ocr_spans),
                "resumed": resumed,
            }
        )
        print(f"[M6 train] {index}/{len(selected)} resumed={resumed}", flush=True)
    manifest = {
        "schema_version": "1.0",
        "split": "train",
        "protected_gold_deserialized": False,
        "video_count": len(records),
        "processing_time_ms": (time.perf_counter_ns() - started) // 1_000_000,
        "records": records,
    }
    _write(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
