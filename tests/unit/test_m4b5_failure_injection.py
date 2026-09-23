"""Deterministic environment-private failure injection contracts for M4B.5."""

from __future__ import annotations

from pathlib import Path

import pytest
from cutagent_evaluation.m4b5_recovery import (
    DeterministicFailureInjectingRegistry,
    FailureInjectionConfig,
    M4B5FailureType,
)

from cutagent.schemas.media import TimeRange
from cutagent.schemas.retrieval import RetrievalResponse
from cutagent.schemas.tools import (
    SearchVideoArgs,
    SearchVideoCall,
    TrimVideoArgs,
    TrimVideoCall,
)
from tests.tool_fixtures import build_tool_runtime, generate_tool_media


def _config(
    failure_type: M4B5FailureType,
    *,
    mode: str = "pre_execute_failure",
    tools: tuple[str, ...] = ("trim_video",),
    occurrence: int = 1,
) -> FailureInjectionConfig:
    return FailureInjectionConfig.model_validate(
        {
            "injection_id": f"failure-{failure_type}",
            "task_id": "m4b5-test-task",
            "failure_type": failure_type,
            "trigger_mode": mode,
            "trigger_tool_names": tools,
            "trigger_occurrence": occurrence,
            "expected_recovery_operations": ["retry_current_node"],
        }
    )


def _trim(source_id: str, call_id: str) -> TrimVideoCall:
    return TrimVideoCall(
        tool_call_id=call_id,
        arguments=TrimVideoArgs(
            input_artifact_id=source_id,
            time_range=TimeRange(start_ms=0, end_ms=1000),
        ),
    )


@pytest.mark.parametrize(
    ("failure_type", "status", "error_code"),
    (
        ("tool_timeout", "timeout", "timeout"),
        ("invalid_tool_arguments", "invalid", "invalid_call"),
        ("artifact_not_allowed", "invalid", "artifact_not_allowed"),
        ("corrupt_media", "error", "corrupt_media"),
        ("incompatible_concat_inputs", "invalid", "incompatible_media"),
        ("invalid_subtitle_timing", "invalid", "invalid_subtitle"),
        ("output_size_limit", "error", "output_too_large"),
    ),
)
def test_pre_execution_failures_are_structured_and_one_shot(
    tmp_path: Path,
    failure_type: M4B5FailureType,
    status: str,
    error_code: str,
) -> None:
    source = generate_tool_media(tmp_path / "source.mp4")
    runtime = build_tool_runtime(tmp_path / "tools", source)
    registry = DeterministicFailureInjectingRegistry(runtime.registry, _config(failure_type))

    first = registry.execute(_trim(runtime.source.artifact_id, "first"), runtime.context)
    second = registry.execute(_trim(runtime.source.artifact_id, "second"), runtime.context)

    assert first.observation.status == status
    assert first.observation.error_code == error_code
    assert second.observation.status == "success"
    assert registry.private_trigger().failure_type == failure_type
    assert registry.private_trigger().eligible_call_count == 1


def test_empty_search_is_successful_output_with_failed_online_semantics(tmp_path: Path) -> None:
    source = generate_tool_media(tmp_path / "source.mp4")
    runtime = build_tool_runtime(tmp_path / "tools", source)
    registry = DeterministicFailureInjectingRegistry(
        runtime.registry,
        _config(
            "search_no_results",
            mode="empty_search",
            tools=("search_video",),
        ),
    )
    record = registry.execute(
        SearchVideoCall(
            tool_call_id="search-empty",
            arguments=SearchVideoArgs(query="observable query"),
        ),
        runtime.context,
    )
    response = record.observation.details["response"]
    assert record.observation.status == "success"
    assert record.observation.error_code is None
    assert isinstance(response, dict)
    assert response["candidates"] == []
    assert RetrievalResponse.model_validate(response).candidates == ()


def test_post_execution_validation_failure_preserves_failed_check(tmp_path: Path) -> None:
    source = generate_tool_media(tmp_path / "source.mp4")
    runtime = build_tool_runtime(tmp_path / "tools", source)
    registry = DeterministicFailureInjectingRegistry(
        runtime.registry,
        _config("post_execution_validation_failure", mode="post_execute_validation"),
    )
    record = registry.execute(_trim(runtime.source.artifact_id, "post-fail"), runtime.context)
    assert record.observation.status == "success"
    assert record.trace.output_artifact is not None
    assert any(
        item.check_name == "injected_post_execution_integrity" and not item.passed
        for item in record.trace.validation_results
    )


def test_repeated_editor_injection_requires_equivalent_second_call(tmp_path: Path) -> None:
    source = generate_tool_media(tmp_path / "source.mp4")
    runtime = build_tool_runtime(tmp_path / "tools", source)
    registry = DeterministicFailureInjectingRegistry(
        runtime.registry,
        _config(
            "repeated_editor_stagnation",
            mode="repeated_editor",
            occurrence=2,
        ),
    )
    first = registry.execute(_trim(runtime.source.artifact_id, "repeat-1"), runtime.context)
    second = registry.execute(_trim(runtime.source.artifact_id, "repeat-2"), runtime.context)
    third = registry.execute(_trim(runtime.source.artifact_id, "repeat-3"), runtime.context)
    assert first.observation.status == "success"
    assert second.observation.error_code == "output_validation_failed"
    assert third.observation.status == "success"
    assert registry.private_trigger().eligible_call_count == 2


def test_private_configuration_does_not_enter_public_observation(tmp_path: Path) -> None:
    source = generate_tool_media(tmp_path / "source.mp4")
    runtime = build_tool_runtime(tmp_path / "tools", source)
    config = _config("tool_timeout")
    registry = DeterministicFailureInjectingRegistry(runtime.registry, config)
    record = registry.execute(
        _trim(runtime.source.artifact_id, "private-boundary"), runtime.context
    )
    serialized = record.model_dump_json().casefold()
    assert config.injection_id.casefold() not in serialized
    assert "expected_recovery_operations" not in serialized
    assert "environment_private" not in serialized
