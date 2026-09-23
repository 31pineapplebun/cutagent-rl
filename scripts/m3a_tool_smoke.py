"""Smoke every M3A media tool and deterministic event replay on real FFmpeg media."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cutagent_evaluation.m3a_tools import (
    build_media_registry,
    context_for,
    generate_media_fixture,
)

from cutagent.agent.state_reducer import StateReducer
from cutagent.schemas.state import AgentState, ExecutionBudget
from cutagent.schemas.task_input import TaskInput
from cutagent.schemas.tools import ToolExecutionRecord
from cutagent.tools.events import tool_record_to_event


def _output_id(record: ToolExecutionRecord) -> str:
    if record.observation.status != "success" or not record.observation.artifacts:
        raise RuntimeError(record.observation.model_dump_json(indent=2))
    return record.observation.artifacts[0].artifact_id


def _execute(
    records: list[ToolExecutionRecord],
    registry: Any,
    context: Any,
    call: dict[str, Any],
    *,
    require_output: bool = False,
) -> str | None:
    record = registry.execute(call, context)
    records.append(record)
    if record.observation.status != "success":
        raise RuntimeError(record.observation.model_dump_json(indent=2))
    return _output_id(record) if require_output else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m3a"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    ffmpeg = shutil.which(args.ffmpeg)
    ffprobe = shutil.which(args.ffprobe)
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("FFmpeg and ffprobe are required")
    started = time.perf_counter()
    root = args.artifact_root.resolve()
    fixture_root = root / "tool_smoke_fixtures"
    red_path = generate_media_fixture(fixture_root / "red.mp4", ffmpeg=ffmpeg, color="red")
    blue_path = generate_media_fixture(fixture_root / "blue.mp4", ffmpeg=ffmpeg, color="blue")
    registry = build_media_registry(root / "tool_smoke_runtime", ffmpeg=ffmpeg, ffprobe=ffprobe)
    red = registry.artifact_store.import_file(red_path, media_type="video/mp4")
    blue = registry.artifact_store.import_file(blue_path, media_type="video/mp4")
    context = context_for(
        execution_id="m3a-tool-smoke",
        artifact_ids=(red.artifact_id, blue.artifact_id),
    )
    records: list[ToolExecutionRecord] = []
    _execute(
        records,
        registry,
        context,
        {
            "tool_name": "inspect_media",
            "tool_call_id": "smoke-inspect",
            "arguments": {"input_artifact_id": red.artifact_id},
        },
    )
    trim_red = _execute(
        records,
        registry,
        context,
        {
            "tool_name": "trim_video",
            "tool_call_id": "smoke-trim-red",
            "arguments": {
                "input_artifact_id": red.artifact_id,
                "time_range": {"start_ms": 0, "end_ms": 1000},
            },
        },
        require_output=True,
    )
    trim_blue = _execute(
        records,
        registry,
        context,
        {
            "tool_name": "trim_video",
            "tool_call_id": "smoke-trim-blue",
            "arguments": {
                "input_artifact_id": blue.artifact_id,
                "time_range": {"start_ms": 500, "end_ms": 1500},
            },
        },
        require_output=True,
    )
    concatenated = _execute(
        records,
        registry,
        context,
        {
            "tool_name": "concat_videos",
            "tool_call_id": "smoke-concat",
            "arguments": {"input_artifact_ids": [trim_red, trim_blue]},
        },
        require_output=True,
    )
    sped = _execute(
        records,
        registry,
        context,
        {
            "tool_name": "change_speed",
            "tool_call_id": "smoke-speed",
            "arguments": {"input_artifact_id": concatenated, "speed_factor": 2.0},
        },
        require_output=True,
    )
    reframed = _execute(
        records,
        registry,
        context,
        {
            "tool_name": "reframe_video",
            "tool_call_id": "smoke-reframe",
            "arguments": {
                "input_artifact_id": sped,
                "width": 90,
                "height": 160,
                "fit": "crop",
            },
        },
        require_output=True,
    )
    normalized = _execute(
        records,
        registry,
        context,
        {
            "tool_name": "normalize_audio",
            "tool_call_id": "smoke-audio",
            "arguments": {"input_artifact_id": reframed, "target_lufs": -16.0},
        },
        require_output=True,
    )
    subtitled = _execute(
        records,
        registry,
        context,
        {
            "tool_name": "add_subtitles",
            "tool_call_id": "smoke-subtitle",
            "arguments": {
                "input_artifact_id": normalized,
                "cues": [
                    {
                        "cue_id": "smoke-cue",
                        "time_range": {"start_ms": 100, "end_ms": 800},
                        "text": "M3A SAFE TOOL",
                    }
                ],
            },
        },
        require_output=True,
    )
    validation = registry.execute(
        {
            "tool_name": "validate_media",
            "tool_call_id": "smoke-validate",
            "arguments": {"input_artifact_id": subtitled, "require_audio": True},
        },
        context,
    )
    records.append(validation)
    if validation.observation.status != "success":
        raise RuntimeError(validation.observation.model_dump_json(indent=2))
    cached = registry.execute(
        {
            "tool_name": "trim_video",
            "tool_call_id": "smoke-trim-red-cached",
            "arguments": {
                "input_artifact_id": red.artifact_id,
                "time_range": {"start_ms": 0, "end_ms": 1000},
            },
        },
        context,
    )
    records.append(cached)
    if cached.observation.status != "success" or not cached.trace.cache_hit:
        raise RuntimeError("M3A smoke cache-repeat did not hit")
    failure = registry.execute(
        {"tool_name": "unknown", "tool_call_id": "smoke-failure", "arguments": {}},
        context,
    )
    if failure.observation.error_code != "unknown_tool":
        raise RuntimeError("M3A smoke failure boundary did not classify unknown tool")

    initial = AgentState.initial(
        TaskInput(
            task_id="m3a-smoke-task",
            video_ref=red,
            instruction="exercise typed media tools without an Agent policy",
        ),
        ExecutionBudget(max_steps=4, max_tool_calls=4, max_wall_time_ms=60_000),
    )
    timestamp = datetime(2026, 8, 23, tzinfo=UTC)
    success_event = tool_record_to_event(validation, initial, created_at=timestamp)
    state_one = StateReducer.apply(initial, success_event)
    failure_event = tool_record_to_event(failure, state_one, created_at=timestamp)
    final = StateReducer.apply(state_one, failure_event)
    replay = StateReducer.apply(StateReducer.apply(initial, success_event), failure_event)
    if replay != final:
        raise RuntimeError("M3A event replay was not deterministic")
    output = {
        "schema_version": "1.0",
        "status": "passed",
        "tool_names": [item.name for item in registry.manifest().tools],
        "successful_calls": len(records),
        "cache_hit": cached.trace.cache_hit,
        "failure_error_code": failure.observation.error_code,
        "event_replay_identical": replay == final,
        "final_state_version": final.state_version,
        "output_artifact": validation.trace.parent_artifacts[0].model_dump(mode="json"),
        "wall_time_ms": round((time.perf_counter() - started) * 1000),
    }
    output_path = root / "tool_smoke.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
