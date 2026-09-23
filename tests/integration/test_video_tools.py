"""Real FFmpeg integration tests for all M3A media tools and cache semantics."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from cutagent.schemas.tools import ToolExecutionContext, ToolExecutionRecord
from cutagent.tools.executor import FFmpegExecutor
from cutagent.tools.validation import MediaValidator, media_contract_equivalence
from tests.media_fixtures import require_ffmpeg
from tests.tool_fixtures import ToolRuntime, build_tool_runtime, generate_tool_media


def _success(
    runtime: ToolRuntime,
    call: Mapping[str, Any],
    context: ToolExecutionContext | None = None,
) -> ToolExecutionRecord:
    record = runtime.registry.execute(call, context or runtime.context)
    assert record.observation.status == "success", record.model_dump_json(indent=2)
    return record


def _frame_digest(path: Path, at_seconds: float) -> str:
    ffmpeg, _ = require_ffmpeg()
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{at_seconds:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    return hashlib.sha256(completed.stdout).hexdigest()


def test_readonly_trim_validate_and_cache(tmp_path: Path) -> None:
    source_path = generate_tool_media(tmp_path / "source.mp4")
    runtime = build_tool_runtime(tmp_path / "runtime", source_path)
    inspection = _success(
        runtime,
        {
            "tool_name": "inspect_media",
            "tool_call_id": "inspect-001",
            "arguments": {"input_artifact_id": runtime.source.artifact_id},
        },
    )
    assert inspection.observation.details["duration_ms"] == 3000
    assert "uri" not in inspection.observation.model_dump_json()

    call = {
        "tool_name": "trim_video",
        "tool_call_id": "trim-001",
        "arguments": {
            "input_artifact_id": runtime.source.artifact_id,
            "time_range": {"start_ms": 500, "end_ms": 2000},
        },
    }
    first = _success(runtime, call)
    assert first.trace.cache_hit is False
    output_id = first.observation.artifacts[0].artifact_id
    validation = _success(
        runtime,
        {
            "tool_name": "validate_media",
            "tool_call_id": "validate-trim",
            "arguments": {"input_artifact_id": output_id, "require_audio": True},
        },
    )
    assert validation.observation.details["fully_decoded"] is True
    isolated = runtime.registry.execute(
        {
            "tool_name": "validate_media",
            "tool_call_id": "validate-cross-execution",
            "arguments": {"input_artifact_id": output_id},
        },
        runtime.context.model_copy(update={"execution_id": "isolated-execution"}),
    )
    assert isolated.observation.error_code == "artifact_not_allowed"

    second = _success(
        runtime,
        {**call, "tool_call_id": "trim-002"},
    )
    assert second.trace.cache_hit is True
    assert second.observation.artifacts[0].sha256 == first.observation.artifacts[0].sha256
    changed = _success(
        runtime,
        {
            **call,
            "tool_call_id": "trim-003",
            "arguments": {
                "input_artifact_id": runtime.source.artifact_id,
                "time_range": {"start_ms": 750, "end_ms": 1500},
            },
        },
    )
    assert changed.trace.cache_hit is False
    assert changed.trace.cache_key != first.trace.cache_key


def test_media_contract_equivalence_is_distinct_from_byte_identity(tmp_path: Path) -> None:
    ffmpeg, ffprobe = require_ffmpeg()
    left_path = generate_tool_media(tmp_path / "left.mp4", color="red")
    right_path = generate_tool_media(tmp_path / "right.mp4", color="blue")
    runtime = build_tool_runtime(tmp_path / "runtime", left_path)
    right = runtime.store.import_file(right_path, media_type="video/mp4")
    validator = MediaValidator(
        artifact_store=runtime.store,
        executor=FFmpegExecutor(ffmpeg),
        ffprobe_executable=ffprobe,
    )
    _, left_object = runtime.store.get(runtime.source.artifact_id)
    _, right_object = runtime.store.get(right.artifact_id)
    left_media = validator.probe(
        runtime.source,
        left_object,
        context=runtime.context,
        decode_entire_video=True,
        require_audio=True,
    )
    context = runtime.context.model_copy(
        update={"allowed_artifact_ids": (runtime.source.artifact_id, right.artifact_id)}
    )
    right_media = validator.probe(
        right,
        right_object,
        context=context,
        decode_entire_video=True,
        require_audio=True,
    )
    assert runtime.source.sha256 != right.sha256
    assert media_contract_equivalence(left_media.asset, right_media.asset).passed


def test_concat_speed_reframe_audio_and_subtitle_tools(tmp_path: Path) -> None:
    red = generate_tool_media(tmp_path / "red.mp4", color="red", duration_seconds=2)
    blue = generate_tool_media(tmp_path / "blue.mp4", color="blue", duration_seconds=2)
    runtime = build_tool_runtime(tmp_path / "runtime", red)
    blue_ref = runtime.store.import_file(blue, media_type="video/mp4")
    context = runtime.context.model_copy(
        update={
            "allowed_artifact_ids": tuple(
                sorted((runtime.source.artifact_id, blue_ref.artifact_id))
            )
        }
    )

    concatenated = _success(
        runtime,
        {
            "tool_name": "concat_videos",
            "tool_call_id": "concat-001",
            "arguments": {"input_artifact_ids": [runtime.source.artifact_id, blue_ref.artifact_id]},
        },
        context,
    )
    assert abs(cast(int, concatenated.observation.details["duration_ms"]) - 4000) <= 200
    assert concatenated.observation.details["audio_sample_rates_hz"] == [48000]
    assert concatenated.observation.details["audio_channel_counts"] == [2]

    sped = _success(
        runtime,
        {
            "tool_name": "change_speed",
            "tool_call_id": "speed-001",
            "arguments": {
                "input_artifact_id": concatenated.observation.artifacts[0].artifact_id,
                "speed_factor": 2.0,
            },
        },
        context,
    )
    assert abs(cast(int, sped.observation.details["duration_ms"]) - 2000) <= 200
    assert any(
        item.check_name == "audio_video_duration_sync" and item.passed
        for item in sped.trace.validation_results
    )

    reframed = _success(
        runtime,
        {
            "tool_name": "reframe_video",
            "tool_call_id": "reframe-001",
            "arguments": {
                "input_artifact_id": sped.observation.artifacts[0].artifact_id,
                "width": 90,
                "height": 160,
                "fit": "crop",
            },
        },
        context,
    )
    assert (reframed.observation.details["width"], reframed.observation.details["height"]) == (
        90,
        160,
    )

    normalized = _success(
        runtime,
        {
            "tool_name": "normalize_audio",
            "tool_call_id": "audio-001",
            "arguments": {
                "input_artifact_id": reframed.observation.artifacts[0].artifact_id,
                "target_lufs": -16,
            },
        },
        context,
    )
    assert normalized.observation.details["has_audio"] is True

    subtitled = _success(
        runtime,
        {
            "tool_name": "add_subtitles",
            "tool_call_id": "subtitle-001",
            "arguments": {
                "input_artifact_id": normalized.observation.artifacts[0].artifact_id,
                "cues": [
                    {
                        "cue_id": "cue-001",
                        "time_range": {"start_ms": 200, "end_ms": 1200},
                        "text": "VISIBLE M3A",
                    }
                ],
            },
        },
        context,
    )
    _, normalized_path = runtime.store.get(normalized.observation.artifacts[0].artifact_id)
    _, subtitle_path = runtime.store.get(subtitled.observation.artifacts[0].artifact_id)
    assert _frame_digest(normalized_path, 0.7) != _frame_digest(subtitle_path, 0.7)


def test_slow_down_duration_ratio(tmp_path: Path) -> None:
    source_path = generate_tool_media(tmp_path / "source.mp4", duration_seconds=1)
    runtime = build_tool_runtime(tmp_path / "runtime", source_path)
    slowed = _success(
        runtime,
        {
            "tool_name": "change_speed",
            "tool_call_id": "slow-001",
            "arguments": {
                "input_artifact_id": runtime.source.artifact_id,
                "speed_factor": 0.5,
            },
        },
    )
    assert abs(cast(int, slowed.observation.details["duration_ms"]) - 2000) <= 200
