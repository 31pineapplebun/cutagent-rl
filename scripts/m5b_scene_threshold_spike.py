"""Measure M1A scene-split stability on public M5 media without private Gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlparse

from cutagent.ingestion.pipeline import VideoIngestionPipeline
from cutagent.schemas.media import IngestionConfig, KeyframeExtractionConfig
from cutagent.schemas.task_input import TaskInput


def _source_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("scene spike requires local file media")
    value = unquote(parsed.path)
    if len(value) >= 3 and value[0] == "/" and value[2] == ":":
        value = value[1:]
    return Path(value).resolve(strict=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "validation"), default="dev")
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path("artifacts/m5a/cutagentbench_v0.1")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m5b/scene_spike"))
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(0.04, 0.05, 0.06, 0.07, 0.08)
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    public_path = args.benchmark_root.resolve(strict=True) / "public" / f"{args.split}.json"
    payload = json.loads(public_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("public task manifest must be a list")
    tasks = tuple(TaskInput.model_validate(item) for item in payload)
    by_video = {task.video_ref.artifact_id: task for task in tasks}
    root = args.artifact_root.resolve()
    pipeline = VideoIngestionPipeline(
        cache_root=root / "cache",
        ffmpeg_executable=args.ffmpeg,
        ffprobe_executable=args.ffprobe,
    )
    results: list[dict[str, object]] = []
    for threshold in args.thresholds:
        config = IngestionConfig(
            scene_threshold=threshold,
            minimum_scene_duration_ms=700,
            keyframes=KeyframeExtractionConfig(strategy="midpoint"),
        )
        for task in tuple(by_video[key] for key in sorted(by_video)):
            ingestion = pipeline.ingest(_source_path(task.video_ref.uri), config=config)
            results.append(
                {
                    "threshold": threshold,
                    "video_id": task.video_ref.artifact_id,
                    "scene_count": len(ingestion.scenes),
                    "ranges": [
                        scene.time_range.model_dump(mode="json") for scene in ingestion.scenes
                    ],
                }
            )
    summary = {
        "schema_version": "1.0",
        "split": args.split,
        "private_gold_deserialized": False,
        "video_count": len(by_video),
        "thresholds": list(args.thresholds),
        "scene_count_distribution": {
            str(threshold): {
                str(count): sum(
                    item["threshold"] == threshold and item["scene_count"] == count
                    for item in results
                )
                for count in sorted(
                    {
                        cast(int, item["scene_count"])
                        for item in results
                        if item["threshold"] == threshold
                    }
                )
            }
            for threshold in args.thresholds
        },
        "records": results,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{args.split}_threshold_spike.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["scene_count_distribution"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
