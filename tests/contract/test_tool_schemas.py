"""Contract tests for the public M3A tool-call boundary."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cutagent.schemas.media import TimeRange
from cutagent.schemas.tools import (
    TOOL_CALL_ADAPTER,
    AddSubtitlesArgs,
    ReframeVideoArgs,
    SubtitleCue,
    ToolExecutionContext,
    ToolTrace,
    TrimVideoCall,
)


def test_discriminated_call_accepts_only_typed_arguments() -> None:
    call = TOOL_CALL_ADAPTER.validate_python(
        {
            "tool_name": "trim_video",
            "tool_call_id": "trim-001",
            "arguments": {
                "input_artifact_id": "input-001",
                "time_range": {"start_ms": 0, "end_ms": 1000},
            },
        }
    )
    assert isinstance(call, TrimVideoCall)
    assert call.arguments.time_range == TimeRange(start_ms=0, end_ms=1000)


@pytest.mark.parametrize(
    "extra",
    (
        {"command": "ffmpeg -i /etc/passwd out.mp4"},
        {"ffmpeg_args": ["-i", "anything"]},
        {"shell": True},
        {"output_path": "../../escape.mp4"},
    ),
)
def test_raw_commands_and_paths_are_forbidden(extra: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TOOL_CALL_ADAPTER.validate_python(
            {
                "tool_name": "trim_video",
                "tool_call_id": "trim-unsafe",
                "arguments": {
                    "input_artifact_id": "input-001",
                    "time_range": {"start_ms": 0, "end_ms": 1000},
                    **extra,
                },
            }
        )


def test_context_uses_opaque_namespace_not_host_path() -> None:
    with pytest.raises(ValidationError):
        ToolExecutionContext(
            execution_id="exec-001",
            allowed_output_root_id="../../host",
            allowed_artifact_ids=(),
            allowed_capabilities=("media.inspect",),
        )


def test_subtitle_and_reframe_contracts_are_strict() -> None:
    with pytest.raises(ValidationError, match="ordered and non-overlapping"):
        AddSubtitlesArgs(
            input_artifact_id="input-001",
            cues=(
                SubtitleCue(
                    cue_id="cue-a",
                    time_range=TimeRange(start_ms=500, end_ms=1000),
                    text="A",
                ),
                SubtitleCue(
                    cue_id="cue-b",
                    time_range=TimeRange(start_ms=900, end_ms=1200),
                    text="B",
                ),
            ),
        )
    with pytest.raises(ValidationError, match="even"):
        ReframeVideoArgs(input_artifact_id="input-001", width=721, height=1280)


def test_success_trace_cannot_carry_an_error_category() -> None:
    with pytest.raises(ValidationError):
        ToolTrace(
            trace_id="trace-001",
            execution_id="exec-001",
            tool_call_id="call-001",
            tool_name="trim_video",
            tool_version="v1",
            normalized_arguments={},
            parent_artifacts=(),
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            ended_at=datetime(2026, 1, 1, tzinfo=UTC),
            latency_ms=0,
            validation_results=(),
            status="success",
            error_category="internal_error",
            cache_hit=False,
        )
