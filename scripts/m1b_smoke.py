"""Run one real M1A -> M1B pipeline twice and prove heavy-model cache hits."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from cutagent.ingestion.pipeline import VideoIngestionPipeline
from cutagent.perception.pipeline import MultimodalPerceptionPipeline
from cutagent.schemas.media import IngestionConfig, KeyframeExtractionConfig
from cutagent.schemas.perception import PerceptionConfig

Image: Any = importlib.import_module("PIL.Image")
ImageDraw: Any = importlib.import_module("PIL.ImageDraw")
ImageFont: Any = importlib.import_module("PIL.ImageFont")


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)


def _font(size: int) -> Any:
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def generate_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    font = _font(30)
    colors = ("red", "green", "blue")
    for index, color in enumerate(colors):
        image = Image.new("RGB", (384, 256), "white")
        draw = ImageDraw.Draw(image)
        left = 30 + index * 110
        draw.rectangle((left, 90, left + 70, 160), fill=color)
        label = ("START", "MIDDLE", "结束")[index]
        draw.text((20, 20), label, font=font, fill="black")
        image.save(root / f"frame_{index + 1:02d}.png")
    speech = root / "speech.wav"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "flite=text='The red square moves to the right.'",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(speech),
        ]
    )
    output = root / "m1b_fixture.mp4"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "1",
            "-i",
            str(root / "frame_%02d.png"),
            "-i",
            str(speech),
            "-filter_complex",
            "[1:a]apad=pad_dur=3[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-t",
            "3",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(output),
        ]
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("artifacts/m1b/server_smoke"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/m1b/server_smoke.json"))
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    model_cache = args.model_cache.resolve()
    if model_cache == repository_root or repository_root in model_cache.parents:
        parser.error("--model-cache must be outside the Git repository")
    root = args.work_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = generate_fixture(root / "fixture")
    ingestion = VideoIngestionPipeline(cache_root=root / "ingestion-cache").ingest(
        source,
        config=IngestionConfig(
            scene_threshold=0.1,
            minimum_scene_duration_ms=300,
            keyframes=KeyframeExtractionConfig(strategy="midpoint"),
        ),
    )
    pipeline = MultimodalPerceptionPipeline(
        cache_root=root / "perception-cache",
        model_cache=model_cache,
    )
    config = PerceptionConfig(visual_mode="keyframes", asr_language="en")
    started = time.perf_counter_ns()
    first = pipeline.run(source, ingestion=ingestion, config=config)
    first_elapsed_ms = (time.perf_counter_ns() - started) // 1_000_000
    started = time.perf_counter_ns()
    second = pipeline.run(source, ingestion=ingestion, config=config)
    second_elapsed_ms = (time.perf_counter_ns() - started) // 1_000_000
    heavy_first = [
        record for record in first.cache_records if record.operation in {"asr", "vlm_perception"}
    ]
    heavy_second = [
        record for record in second.cache_records if record.operation in {"asr", "vlm_perception"}
    ]
    passed = (
        bool(heavy_first)
        and all(not record.hit for record in heavy_first)
        and all(record.hit for record in heavy_second)
        and first.world_state == second.world_state
    )
    (root / "perception_result_first.json").write_text(
        first.model_dump_json(indent=2), encoding="utf-8"
    )
    report = {
        "schema_version": "1.0.0",
        "status": "passed" if passed else "failed",
        "source": {
            "artifact_id": ingestion.video.source.artifact_id,
            "sha256": ingestion.video.source.sha256,
            "duration_ms": ingestion.video.duration_ms,
            "resolution": [
                ingestion.video.video_stream.width,
                ingestion.video.video_stream.height,
            ],
            "scene_count": len(ingestion.scenes),
        },
        "counts": {
            "transcript_spans": len(first.transcript_spans),
            "ocr_spans": len(first.ocr_spans),
            "entities": sum(len(scene.entities) for scene in first.world_state.scenes),
            "actions": sum(len(scene.actions) for scene in first.world_state.scenes),
            "events": len(first.temporal_events),
        },
        "world_state_example": first.world_state.scenes[0].model_dump(mode="json"),
        "performance": [item.model_dump(mode="json") for item in first.performance],
        "first_run_ms": first_elapsed_ms,
        "second_run_ms": second_elapsed_ms,
        "first_heavy_cache": [item.model_dump(mode="json") for item in heavy_first],
        "second_heavy_cache": [item.model_dump(mode="json") for item in heavy_second],
        "world_state_identical": first.world_state == second.world_state,
        "produced_artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "sha256": artifact.sha256,
                "media_type": artifact.media_type,
            }
            for artifact in first.provenance_artifacts
        ],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"M1B pipeline smoke status={report['status']} scenes={len(ingestion.scenes)} "
        f"output={args.output.resolve()}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
