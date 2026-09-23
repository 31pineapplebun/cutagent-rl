"""Deterministic event-based state reducer."""

from collections.abc import Iterable

from pydantic import ValidationError

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
    TerminalEvent,
    ToolObservation,
    VerificationResult,
)
from cutagent.schemas.state import AgentState, ExecutionBudget


class StateReducer:
    """Apply immutable Agent events with strict ordering and version checks."""

    @staticmethod
    def apply(state: AgentState, envelope: AgentEventEnvelope) -> AgentState:
        if envelope.task_id != state.task_input.task_id:
            raise TaskMismatchError(
                f"event task {envelope.task_id!r} does not match {state.task_input.task_id!r}"
            )
        if envelope.event_id in state.processed_event_ids:
            raise DuplicateEventError(f"event {envelope.event_id!r} has already been applied")
        expected_sequence = state.last_sequence_no + 1
        if envelope.sequence_no != expected_sequence:
            raise EventSequenceError(
                f"expected sequence {expected_sequence}, got {envelope.sequence_no}"
            )
        if envelope.parent_state_version != state.state_version:
            raise ParentStateVersionError(
                f"expected parent state version {state.state_version}, "
                f"got {envelope.parent_state_version}"
            )
        if state.terminal_event is not None:
            raise TerminalStateError("cannot apply an event after terminal state")

        update: dict[str, object] = {}
        event = envelope.event

        if isinstance(event, ToolObservation):
            update["tool_observations"] = (*state.tool_observations, event)
        elif isinstance(event, PlanPatch):
            expected_revision = state.plan_revision + 1
            if event.revision != expected_revision:
                raise InvalidEventError(
                    f"expected plan revision {expected_revision}, got {event.revision}"
                )
            update["plan_revision"] = event.revision
            update["plan_steps"] = event.steps
        elif isinstance(event, VerificationResult):
            update["verification_results"] = (*state.verification_results, event)
        elif isinstance(event, BudgetUpdate):
            try:
                update["execution_budget"] = ExecutionBudget(
                    max_steps=state.execution_budget.max_steps,
                    max_tool_calls=state.execution_budget.max_tool_calls,
                    max_search_calls=state.execution_budget.max_search_calls,
                    max_edit_calls=state.execution_budget.max_edit_calls,
                    max_repeated_identical_actions=(
                        state.execution_budget.max_repeated_identical_actions
                    ),
                    max_structured_output_repairs=(
                        state.execution_budget.max_structured_output_repairs
                    ),
                    max_model_tokens=state.execution_budget.max_model_tokens,
                    max_wall_time_ms=state.execution_budget.max_wall_time_ms,
                    used_steps=state.execution_budget.used_steps + event.used_steps_delta,
                    used_tool_calls=(
                        state.execution_budget.used_tool_calls + event.used_tool_calls_delta
                    ),
                    used_search_calls=(
                        state.execution_budget.used_search_calls + event.used_search_calls_delta
                    ),
                    used_edit_calls=(
                        state.execution_budget.used_edit_calls + event.used_edit_calls_delta
                    ),
                    used_structured_output_repairs=(
                        state.execution_budget.used_structured_output_repairs
                        + event.used_structured_output_repairs_delta
                    ),
                    used_model_tokens=(
                        state.execution_budget.used_model_tokens + event.used_model_tokens_delta
                    ),
                    used_wall_time_ms=(
                        state.execution_budget.used_wall_time_ms + event.used_wall_time_ms_delta
                    ),
                )
            except ValidationError as exc:
                raise InvalidEventError("budget update exceeds the execution budget") from exc
        elif isinstance(event, TerminalEvent):
            update["terminal_event"] = event
        else:  # pragma: no cover - the discriminated union prevents this
            raise InvalidEventError(f"unsupported event type: {type(event).__name__}")

        update.update(
            processed_event_ids=(*state.processed_event_ids, envelope.event_id),
            last_sequence_no=envelope.sequence_no,
            state_version=state.state_version + 1,
        )
        candidate = {**state.model_dump(mode="python"), **update}
        return AgentState.model_validate(candidate)

    @classmethod
    def replay(
        cls,
        initial_state: AgentState,
        events: Iterable[AgentEventEnvelope],
    ) -> AgentState:
        state = initial_state
        for event in events:
            state = cls.apply(state, event)
        return state
