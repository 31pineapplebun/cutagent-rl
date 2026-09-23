"""Reducer invariants and deterministic replay."""

from datetime import UTC, datetime, timedelta

import pytest

from cutagent.agent.state_reducer import StateReducer
from cutagent.core.errors import (
    DuplicateEventError,
    EventSequenceError,
    InvalidEventError,
    ParentStateVersionError,
    TaskMismatchError,
    TerminalStateError,
)
from cutagent.schemas.event import (
    AgentEventEnvelope,
    BudgetUpdate,
    PlanPatch,
    PlanStep,
    RuntimeCheckResult,
    TerminalEvent,
    ToolObservation,
    VerificationResult,
)
from cutagent.schemas.state import AgentState

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def envelope(
    sequence_no: int,
    event: object,
    emitted_by: str,
    *,
    event_id: str | None = None,
    task_id: str = "task-001",
    parent_state_version: int | None = None,
) -> AgentEventEnvelope:
    event_payload = event.model_dump(mode="python")  # type: ignore[attr-defined]
    return AgentEventEnvelope.model_validate(
        {
            "event_id": event_id or f"event-{sequence_no}",
            "task_id": task_id,
            "sequence_no": sequence_no,
            "event": event_payload,
            "emitted_by": emitted_by,
            "created_at": NOW + timedelta(seconds=sequence_no),
            "parent_state_version": (
                sequence_no - 1 if parent_state_version is None else parent_state_version
            ),
        }
    )


def sequence() -> tuple[AgentEventEnvelope, ...]:
    return (
        envelope(
            1,
            PlanPatch(
                revision=1,
                reason="test plan",
                steps=(PlanStep(step_id="step-1", description="test step"),),
            ),
            "planner",
        ),
        envelope(
            2,
            ToolObservation(
                call_id="call-1",
                tool_name="synthetic_tool",
                status="success",
                public_summary="synthetic observation",
                details={"private_path": "/internal/result"},
            ),
            "tool",
        ),
        envelope(
            3,
            VerificationResult(
                verification_id="verification-1",
                status="passed",
                checks=(
                    RuntimeCheckResult(
                        check_name="schema_check",
                        passed=True,
                        summary="schema is valid",
                    ),
                ),
            ),
            "verifier",
        ),
        envelope(
            4,
            BudgetUpdate(
                used_steps_delta=1,
                used_tool_calls_delta=1,
                used_wall_time_ms_delta=10,
            ),
            "budget",
        ),
        envelope(5, TerminalEvent(status="succeeded", reason="done"), "runtime"),
    )


def test_reducer_replay_is_deterministic(initial_state: AgentState) -> None:
    events = sequence()
    first = StateReducer.replay(initial_state, events)
    second = StateReducer.replay(initial_state, events)

    assert first.model_dump_json() == second.model_dump_json()
    assert first.state_version == len(events)
    assert first.last_sequence_no == len(events)
    assert len(first.processed_event_ids) == len(events)


def test_duplicate_event_is_rejected(initial_state: AgentState) -> None:
    first_event = sequence()[0]
    state = StateReducer.apply(initial_state, first_event)
    duplicate = envelope(
        2,
        ToolObservation(
            call_id="call-duplicate",
            tool_name="synthetic_tool",
            status="success",
            public_summary="duplicate",
        ),
        "tool",
        event_id=first_event.event_id,
        parent_state_version=state.state_version,
    )
    with pytest.raises(DuplicateEventError):
        StateReducer.apply(state, duplicate)


def test_skipped_sequence_is_rejected(initial_state: AgentState) -> None:
    skipped = envelope(
        2,
        ToolObservation(
            call_id="call-2",
            tool_name="synthetic_tool",
            status="success",
            public_summary="skipped",
        ),
        "tool",
        parent_state_version=0,
    )
    with pytest.raises(EventSequenceError):
        StateReducer.apply(initial_state, skipped)


def test_parent_version_is_checked(initial_state: AgentState) -> None:
    wrong_parent = envelope(
        1,
        PlanPatch(revision=1, reason="test", steps=()),
        "planner",
        parent_state_version=1,
    )
    with pytest.raises(ParentStateVersionError):
        StateReducer.apply(initial_state, wrong_parent)


def test_task_id_is_checked(initial_state: AgentState) -> None:
    wrong_task = envelope(
        1,
        PlanPatch(revision=1, reason="test", steps=()),
        "planner",
        task_id="task-other",
    )
    with pytest.raises(TaskMismatchError):
        StateReducer.apply(initial_state, wrong_task)


def test_plan_revision_cannot_skip(initial_state: AgentState) -> None:
    skipped_revision = envelope(
        1,
        PlanPatch(revision=2, reason="skip", steps=()),
        "planner",
    )
    with pytest.raises(InvalidEventError):
        StateReducer.apply(initial_state, skipped_revision)


def test_budget_cannot_be_exceeded(initial_state: AgentState) -> None:
    excess = envelope(
        1,
        BudgetUpdate(used_steps_delta=5),
        "budget",
    )
    with pytest.raises(InvalidEventError):
        StateReducer.apply(initial_state, excess)


def test_events_after_terminal_are_rejected(initial_state: AgentState) -> None:
    terminal = envelope(1, TerminalEvent(status="failed", reason="stop"), "runtime")
    state = StateReducer.apply(initial_state, terminal)
    later = envelope(
        2,
        BudgetUpdate(used_steps_delta=1),
        "budget",
        parent_state_version=1,
    )
    with pytest.raises(TerminalStateError):
        StateReducer.apply(state, later)
