"""Prepare real M1A/M1B public media evidence for an allowed M5B split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

from cutagent.ingestion.pipeline import VideoIngestionPipeline
from cutagent.perception.pipeline import MultimodalPerceptionPipeline
from cutagent.schemas.media import IngestionConfig, IngestionResult, KeyframeExtractionConfig
from cutagent.schemas.perception import PerceptionConfig, PerceptionResult
from cutagent.schemas.task_input import TaskInput

try:
    from scripts.m10_protected_preparation import (
        ALL_PREPARATION_SPLITS,
        ProtectedPublicPreparationAccessRecord,
        authorization_sha256,
        validate_preparation_request,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from m10_protected_preparation import (  # type: ignore[no-redef,import-not-found]
        ALL_PREPARATION_SPLITS,
        ProtectedPublicPreparationAccessRecord,
        authorization_sha256,
        validate_preparation_request,
    )

ALLOWED_SPLITS = ALL_PREPARATION_SPLITS


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("M5B requires local immutable file artifacts")
    value = unquote(parsed.path)
    if len(value) >= 3 and value[0] == "/" and value[2] == ":":
        value = value[1:]
    return Path(value).resolve(strict=True)


def _load_public_tasks(benchmark_root: Path, split: str) -> tuple[TaskInput, ...]:
    payload = json.loads((benchmark_root / "public" / f"{split}.json").read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("public split manifest must be a JSON list")
    return tuple(TaskInput.model_validate(item) for item in payload)


def _canonicalize_ingestion(
    ingestion: IngestionResult,
    task: TaskInput,
) -> IngestionResult:
    if ingestion.video.source.sha256 != task.video_ref.sha256:
        raise ValueError("ingested source differs from public TaskInput artifact")
    payload = ingestion.model_dump(mode="json")
    payload["video"]["video_id"] = task.video_ref.artifact_id
    payload["video"]["source"] = task.video_ref.model_dump(mode="json")
    for scene in payload["scenes"]:
        scene["parent_video_id"] = task.video_ref.artifact_id
    for keyframe in payload["keyframes"]:
        keyframe["parent_video_id"] = task.video_ref.artifact_id
    return IngestionResult.model_validate(payload)


def _canonicalize_perception(
    perception: PerceptionResult,
    task: TaskInput,
) -> PerceptionResult:
    payload = perception.model_dump(mode="json")
    payload["source_video_id"] = task.video_ref.artifact_id
    payload["world_state"]["video_id"] = task.video_ref.artifact_id
    return PerceptionResult.model_validate(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=ALLOWED_SPLITS)
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path("artifacts/m5a/cutagentbench_v0.1")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m5b"))
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--protected-authorization", type=Path, default=None)
    parser.add_argument("--human-gate", type=Path, default=None)
    parser.add_argument("--protected-plan", type=Path, default=None)
    parser.add_argument("--preparation-access-record", type=Path, default=None)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    benchmark_root = args.benchmark_root.resolve(strict=True)
    model_cache = args.model_cache.resolve(strict=True)
    authorization = validate_preparation_request(
        split=args.split,
        benchmark_root=benchmark_root,
        authorization_path=(
            args.protected_authorization.resolve(strict=True)
            if args.protected_authorization is not None
            else None
        ),
        human_gate_path=(
            args.human_gate.resolve(strict=True) if args.human_gate is not None else None
        ),
        protected_plan_path=(
            args.protected_plan.resolve(strict=True) if args.protected_plan is not None else None
        ),
    )
    if authorization is None and args.preparation_access_record is not None:
        raise ValueError("development preparation must not write a protected access record")
    if authorization is not None and args.preparation_access_record is None:
        raise PermissionError("protected preparation requires a public-preparation access record")
    output_root = args.artifact_root.resolve() / "prepared" / args.split
    tasks = _load_public_tasks(benchmark_root, args.split)
    access_record_path: Path | None = None
    access_started_at: datetime | None = None
    if authorization is not None:
        access_record_path = args.preparation_access_record.resolve()
        access_started_at = datetime.now(UTC)
        _write_json(
            access_record_path,
            ProtectedPublicPreparationAccessRecord(
                status="started",
                benchmark_manifest_sha256=authorization.benchmark_manifest_sha256,
                split=cast(Any, args.split),
                finalization_run_id=authorization.finalization_run_id,
                authorization_sha256=authorization_sha256(authorization),
                public_task_manifest_sha256=authorization.public_task_manifest_sha256,
                task_count=len(tasks),
                started_at=access_started_at,
            ).model_dump(mode="json"),
        )
    by_artifact: dict[str, TaskInput] = {}
    for task in tasks:
        previous = by_artifact.setdefault(task.video_ref.artifact_id, task)
        if previous.video_ref != task.video_ref:
            raise ValueError("one public artifact ID maps to inconsistent media")
    selected = tuple(by_artifact[key] for key in sorted(by_artifact))
    if args.limit_videos is not None:
        if args.limit_videos <= 0:
            raise ValueError("--limit-videos must be positive")
        selected = selected[: args.limit_videos]
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
    ingestion_pipeline = VideoIngestionPipeline(
        cache_root=args.artifact_root.resolve() / "media_cache" / "ingestion",
        ffmpeg_executable=args.ffmpeg,
        ffprobe_executable=args.ffprobe,
    )
    perception_pipeline = MultimodalPerceptionPipeline(
        cache_root=args.artifact_root.resolve() / "media_cache" / "perception",
        model_cache=model_cache,
        ffmpeg_executable=args.ffmpeg,
    )
    records: list[dict[str, object]] = []
    for index, task in enumerate(selected, start=1):
        ingestion_path = output_root / "ingestion" / f"{task.video_ref.artifact_id}.json"
        perception_path = output_root / "perception" / f"{task.video_ref.artifact_id}.json"
        prepared_exists = ingestion_path.is_file() and perception_path.is_file()
        resumed = False
        if prepared_exists:
            ingestion = IngestionResult.model_validate_json(
                ingestion_path.read_text(encoding="utf-8")
            )
            perception = PerceptionResult.model_validate_json(
                perception_path.read_text(encoding="utf-8")
            )
            resumed = (
                ingestion.config == ingestion_config
                and perception.config == perception_config
                and ingestion.video.video_id == task.video_ref.artifact_id
                and perception.world_state.video_id == task.video_ref.artifact_id
            )
        if not resumed:
            source = _artifact_path(task.video_ref.uri)
            raw_ingestion = ingestion_pipeline.ingest(source, config=ingestion_config)
            ingestion = _canonicalize_ingestion(raw_ingestion, task)
            raw_perception = perception_pipeline.run(
                source,
                ingestion=raw_ingestion,
                config=perception_config,
            )
            perception = _canonicalize_perception(raw_perception, task)
            _write_json(ingestion_path, ingestion.model_dump(mode="json"))
            _write_json(perception_path, perception.model_dump(mode="json"))
        records.append(
            {
                "video_id": task.video_ref.artifact_id,
                "source_sha256": task.video_ref.sha256,
                "duration_ms": ingestion.video.duration_ms,
                "scene_count": len(ingestion.scenes),
                "scene_ranges": [
                    item.time_range.model_dump(mode="json") for item in ingestion.scenes
                ],
                "keyframe_count": len(ingestion.keyframes),
                "transcript_span_count": len(perception.transcript_spans),
                "ocr_span_count": len(perception.ocr_spans),
                "action_count": sum(len(item.actions) for item in perception.world_state.scenes),
                "resumed": resumed,
            }
        )
        print(
            f"[{args.split}] {index}/{len(selected)} {task.video_ref.artifact_id} "
            f"scenes={len(ingestion.scenes)} resumed={resumed}",
            flush=True,
        )
    identity = {
        "split": args.split,
        "benchmark_version": "cutagentbench-v0.1",
        "ingestion_config": ingestion_config.model_dump(mode="json"),
        "perception_config": perception_config.model_dump(mode="json"),
        "video_ids": [item["video_id"] for item in records],
    }
    manifest = {
        "schema_version": "1.0",
        "status": "completed",
        "split": args.split,
        "benchmark_version": "cutagentbench-v0.1",
        "task_count": len(tasks),
        "prepared_video_count": len(records),
        "config_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "processing_time_ms": (time.perf_counter_ns() - started) // 1_000_000,
        "records": records,
        "protected_gold_deserialized": False,
    }
    _write_json(output_root / "manifest.json", manifest)
    if authorization is not None:
        if access_record_path is None or access_started_at is None:
            raise RuntimeError("protected preparation access accounting was not initialized")
        _write_json(
            access_record_path,
            ProtectedPublicPreparationAccessRecord(
                status="completed",
                benchmark_manifest_sha256=authorization.benchmark_manifest_sha256,
                split=cast(Any, args.split),
                finalization_run_id=authorization.finalization_run_id,
                authorization_sha256=authorization_sha256(authorization),
                public_task_manifest_sha256=authorization.public_task_manifest_sha256,
                task_count=len(tasks),
                prepared_video_count=len(records),
                started_at=access_started_at,
                completed_at=datetime.now(UTC),
            ).model_dump(mode="json"),
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
