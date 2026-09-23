"""Real FFmpeg M1A smoke test using user media or a generated licensed fixture."""

import argparse
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from cutagent.ingestion.pipeline import VideoIngestionPipeline
from cutagent.schemas.media import IngestionConfig, KeyframeExtractionConfig


def _generate_fixture(path: Path, ffmpeg: str) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=320x180:r=10:d=1",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=320x180:r=10:d=1",
        "-f",
        "lavfi",
        "-i",
        "color=c=green:s=320x180:r=10:d=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=16000:duration=3",
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map",
        "[v]",
        "-map",
        "3:a",
        "-c:v",
        "mpeg4",
        "-q:v",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        "-y",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"synthetic fixture generation failed: {completed.stderr}")


def run_smoke(
    *,
    input_path: Path | None,
    work_root: Path,
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, Any]:
    run_id = f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_directory = (work_root / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    if input_path is None:
        source_path = run_directory / "generated-multiscene.mp4"
        _generate_fixture(source_path, ffmpeg)
        provenance = {
            "kind": "generated_fixture",
            "description": "FFmpeg lavfi red/blue/green scenes with generated sine audio",
            "license": "generated locally; no third-party copyrighted media",
            "real_video_validation": "pending_for_m1b",
        }
    else:
        source_path = input_path.resolve(strict=True)
        provenance = {
            "kind": "user_supplied",
            "description": "caller-provided media; provenance must be managed by caller",
            "license": "not asserted by smoke test",
            "real_video_validation": "performed_on_user_supplied_media",
        }

    config = IngestionConfig(
        scene_threshold=0.1,
        minimum_scene_duration_ms=300,
        keyframes=KeyframeExtractionConfig(strategy="midpoint", frames_per_scene=1),
    )
    pipeline = VideoIngestionPipeline(
        cache_root=run_directory / "cache",
        ffmpeg_executable=ffmpeg,
        ffprobe_executable=ffprobe,
    )
    started = time.perf_counter_ns()
    first = pipeline.ingest(source_path, config=config)
    second = pipeline.ingest(source_path, config=config)
    total_processing_time_ms = (time.perf_counter_ns() - started) // 1_000_000
    first_miss = all(not record.hit for record in first.cache_records)
    second_hit = all(record.hit for record in second.cache_records)
    replay_identical = (
        first.video == second.video
        and first.normalization == second.normalization
        and first.scenes == second.scenes
        and first.keyframes == second.keyframes
    )
    status = "passed" if first_miss and second_hit and replay_identical else "failed"
    return {
        "schema_version": "1.0",
        "status": status,
        "run_id": run_id,
        "provenance": provenance,
        "source": {
            "artifact": first.video.source.model_dump(mode="json"),
            "duration_ms": first.video.duration_ms,
            "resolution": {
                "width": first.video.video_stream.width,
                "height": first.video.video_stream.height,
            },
            "container_formats": list(first.video.container_formats),
            "video_codec": first.video.video_stream.codec_name,
            "time_base": first.video.video_stream.time_base.model_dump(mode="json"),
            "source_start_time_ms": first.video.video_stream.source_start_time_ms,
            "average_frame_rate": (
                first.video.video_stream.average_frame_rate.model_dump(mode="json")
                if first.video.video_stream.average_frame_rate is not None
                else None
            ),
            "real_frame_rate": (
                first.video.video_stream.real_frame_rate.model_dump(mode="json")
                if first.video.video_stream.real_frame_rate is not None
                else None
            ),
            "variable_frame_rate": first.video.video_stream.variable_frame_rate,
            "audio_present": bool(first.video.audio_streams),
            "raw_ffprobe_artifact": first.video.raw_ffprobe.model_dump(mode="json"),
        },
        "scene_count": len(first.scenes),
        "scenes": [
            {
                "segment_id": scene.segment_id,
                "start_ms": scene.time_range.start_ms,
                "end_ms": scene.time_range.end_ms,
                "source_timestamps": scene.source_timestamps.model_dump(mode="json"),
            }
            for scene in first.scenes
        ],
        "keyframe_count": len(first.keyframes),
        "keyframes": [keyframe.model_dump(mode="json") for keyframe in first.keyframes],
        "first_run": {
            "processing_time_ms": first.processing_time_ms,
            "cache_records": [record.model_dump(mode="json") for record in first.cache_records],
            "all_cache_misses": first_miss,
        },
        "second_run": {
            "processing_time_ms": second.processing_time_ms,
            "cache_records": [record.model_dump(mode="json") for record in second.cache_records],
            "all_cache_hits": second_hit,
        },
        "replay_identical": replay_identical,
        "total_processing_time_ms": total_processing_time_ms,
        "tool_versions": first.tool_versions,
    }


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=Path("artifacts/m1a"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/m1a/smoke.json"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    ffmpeg = shutil.which(args.ffmpeg)
    ffprobe = shutil.which(args.ffprobe)
    if ffmpeg is None or ffprobe is None:
        print("M1A smoke status=failed error=FFmpeg/ffprobe not found")
        return 1
    try:
        report = run_smoke(
            input_path=args.input,
            work_root=args.work_root,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
    except Exception as error:
        print(f"M1A smoke status=failed error={type(error).__name__}: {error}")
        return 1
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"M1A smoke status={report['status']} scenes={report['scene_count']} "
        f"keyframes={report['keyframe_count']} output={output}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
