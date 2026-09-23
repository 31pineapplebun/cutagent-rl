"""Controlled M3A failure injection through the public registry boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from cutagent.schemas.tools import ToolExecutionContext
from tests.tool_fixtures import ToolRuntime, build_tool_runtime, generate_tool_media


def _failure(
    runtime: ToolRuntime,
    call: dict[str, object],
    expected: str,
    *,
    context: ToolExecutionContext | None = None,
) -> None:
    record = runtime.registry.execute(call, context or runtime.context)
    assert record.observation.status != "success"
    assert record.observation.error_code == expected
    assert record.trace.error_category == expected


@pytest.fixture
def failure_runtime(tmp_path: Path) -> ToolRuntime:
    source = generate_tool_media(tmp_path / "source.mp4", duration_seconds=2)
    return build_tool_runtime(tmp_path / "runtime", source)


@pytest.mark.parametrize(
    ("call", "expected"),
    (
        (
            {"tool_name": "not_registered", "tool_call_id": "f-unknown", "arguments": {}},
            "unknown_tool",
        ),
        (
            {
                "tool_name": "trim_video",
                "tool_call_id": "f-negative",
                "arguments": {
                    "input_artifact_id": "input-any",
                    "time_range": {"start_ms": -1, "end_ms": 100},
                },
            },
            "invalid_call",
        ),
        (
            {
                "tool_name": "trim_video",
                "tool_call_id": "f-order",
                "arguments": {
                    "input_artifact_id": "input-any",
                    "time_range": {"start_ms": 100, "end_ms": 100},
                },
            },
            "invalid_call",
        ),
        (
            {
                "tool_name": "trim_video",
                "tool_call_id": "f-path",
                "arguments": {
                    "input_artifact_id": "../../etc/passwd",
                    "time_range": {"start_ms": 0, "end_ms": 100},
                },
            },
            "invalid_call",
        ),
        (
            {
                "tool_name": "trim_video",
                "tool_call_id": "f-absolute",
                "arguments": {
                    "input_artifact_id": "C:/Windows/win.ini",
                    "time_range": {"start_ms": 0, "end_ms": 100},
                },
            },
            "invalid_call",
        ),
    ),
)
def test_schema_and_registry_failures_are_structured(
    failure_runtime: ToolRuntime,
    call: dict[str, object],
    expected: str,
) -> None:
    _failure(failure_runtime, call, expected)


def test_interval_outside_media_is_rejected(failure_runtime: ToolRuntime) -> None:
    _failure(
        failure_runtime,
        {
            "tool_name": "trim_video",
            "tool_call_id": "f-outside",
            "arguments": {
                "input_artifact_id": failure_runtime.source.artifact_id,
                "time_range": {"start_ms": 1000, "end_ms": 3000},
            },
        },
        "invalid_interval",
    )


def test_nonexistent_and_corrupt_artifacts_are_structured(
    failure_runtime: ToolRuntime,
    tmp_path: Path,
) -> None:
    missing_context = failure_runtime.context.model_copy(
        update={"allowed_artifact_ids": ("missing-artifact",)}
    )
    _failure(
        failure_runtime,
        {
            "tool_name": "inspect_media",
            "tool_call_id": "f-missing",
            "arguments": {"input_artifact_id": "missing-artifact"},
        },
        "artifact_not_found",
        context=missing_context,
    )
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a media file")
    corrupt_ref = failure_runtime.store.import_file(corrupt, media_type="video/mp4")
    corrupt_context = failure_runtime.context.model_copy(
        update={"allowed_artifact_ids": (corrupt_ref.artifact_id,)}
    )
    _failure(
        failure_runtime,
        {
            "tool_name": "inspect_media",
            "tool_call_id": "f-corrupt",
            "arguments": {"input_artifact_id": corrupt_ref.artifact_id},
        },
        "corrupt_media",
        context=corrupt_context,
    )


def test_incompatible_concat_and_invalid_subtitle(
    failure_runtime: ToolRuntime, tmp_path: Path
) -> None:
    other_path = generate_tool_media(
        tmp_path / "other.mp4", width=320, height=180, duration_seconds=1
    )
    other = failure_runtime.store.import_file(other_path, media_type="video/mp4")
    context = failure_runtime.context.model_copy(
        update={
            "allowed_artifact_ids": (
                failure_runtime.source.artifact_id,
                other.artifact_id,
            )
        }
    )
    _failure(
        failure_runtime,
        {
            "tool_name": "concat_videos",
            "tool_call_id": "f-concat",
            "arguments": {
                "input_artifact_ids": [
                    failure_runtime.source.artifact_id,
                    other.artifact_id,
                ]
            },
        },
        "incompatible_media",
        context=context,
    )
    _failure(
        failure_runtime,
        {
            "tool_name": "add_subtitles",
            "tool_call_id": "f-subtitle",
            "arguments": {
                "input_artifact_id": failure_runtime.source.artifact_id,
                "cues": [
                    {
                        "cue_id": "cue-outside",
                        "time_range": {"start_ms": 1500, "end_ms": 2500},
                        "text": "outside",
                    }
                ],
            },
        },
        "invalid_subtitle",
    )


def test_timeout_and_output_size_limit_are_structured(tmp_path: Path) -> None:
    source = generate_tool_media(tmp_path / "source.mp4", duration_seconds=2)
    timeout_runtime = build_tool_runtime(tmp_path / "timeout-runtime", source, timeout_ms=1)
    _failure(
        timeout_runtime,
        {
            "tool_name": "trim_video",
            "tool_call_id": "f-timeout",
            "arguments": {
                "input_artifact_id": timeout_runtime.source.artifact_id,
                "time_range": {"start_ms": 0, "end_ms": 1000},
            },
        },
        "timeout",
    )
    small_runtime = build_tool_runtime(tmp_path / "small-runtime", source, maximum_output_bytes=32)
    _failure(
        small_runtime,
        {
            "tool_name": "trim_video",
            "tool_call_id": "f-size",
            "arguments": {
                "input_artifact_id": small_runtime.source.artifact_id,
                "time_range": {"start_ms": 0, "end_ms": 1000},
            },
        },
        "output_too_large",
    )
