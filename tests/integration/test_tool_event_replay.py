"""Successful and failed M3A observations replay through the M0 reducer."""

from datetime import UTC, datetime
from pathlib import Path

from cutagent.agent.state_reducer import StateReducer
from cutagent.schemas.event import AgentEventEnvelope
from cutagent.schemas.state import AgentState, ExecutionBudget
from cutagent.schemas.task_input import TaskInput
from cutagent.tools.events import tool_record_to_event
from tests.tool_fixtures import build_tool_runtime, generate_tool_media


def test_success_and_failure_observations_replay_deterministically(tmp_path: Path) -> None:
    source_path = generate_tool_media(tmp_path / "source.mp4", duration_seconds=2)
    runtime = build_tool_runtime(tmp_path / "runtime", source_path)
    success = runtime.registry.execute(
        {
            "tool_name": "trim_video",
            "tool_call_id": "sequence-trim",
            "arguments": {
                "input_artifact_id": runtime.source.artifact_id,
                "time_range": {"start_ms": 0, "end_ms": 1000},
            },
        },
        runtime.context,
    )
    failure = runtime.registry.execute(
        {"tool_name": "unknown", "tool_call_id": "sequence-failure", "arguments": {}},
        runtime.context,
    )
    initial = AgentState.initial(
        TaskInput(
            task_id="m3a-event-task",
            video_ref=runtime.source,
            instruction="exercise deterministic tool observations",
        ),
        ExecutionBudget(
            max_steps=4,
            max_tool_calls=4,
            max_wall_time_ms=60_000,
        ),
    )
    timestamp = datetime(2026, 8, 23, tzinfo=UTC)
    first = tool_record_to_event(
        success,
        initial,
        created_at=timestamp,
    )
    state_one = StateReducer.apply(initial, first)
    second = tool_record_to_event(
        failure,
        state_one,
        created_at=timestamp,
    )
    final = StateReducer.apply(state_one, second)

    replay = initial
    for event in (first, second):
        replay = StateReducer.apply(replay, AgentEventEnvelope.model_validate(event.model_dump()))
    assert replay == final
    assert [item.status for item in final.tool_observations] == ["success", "invalid"]
