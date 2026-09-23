"""Deterministic reducer for the versioned M4B public Agent state."""

from __future__ import annotations

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
    BudgetUpdate,
    PlanPatch,
    TerminalEvent,
    ToolObservation,
    VerificationResult,
)
from cutagent.schemas.m4b_agent import (
    M4BAgentState,
    M4BEventEnvelope,
    ObservableMediaMetadata,
    RecoveryOperationEvent,
    SanitizedSystemErrorEvent,
    StepOutcomeEvent,
    WorkingArtifactState,
    WorkingArtifactValidationEvent,
    WorkingMediaArtifact,
)
from cutagent.schemas.state import ExecutionBudget

_EDIT_TOOLS = frozenset(
    {
        "trim_video",
        "concat_videos",
        "change_speed",
        "add_subtitles",
        "reframe_video",
        "normalize_audio",
    }
)


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _media_metadata(details: dict[str, object]) -> ObservableMediaMetadata | None:
    duration = _nonnegative_int(details.get("duration_ms"))
    width = _positive_int(details.get("width"))
    height = _positive_int(details.get("height"))
    has_audio = details.get("has_audio")
    fully_decoded = details.get("fully_decoded")
    if not isinstance(has_audio, bool):
        has_audio = None
    if not isinstance(fully_decoded, bool):
        fully_decoded = None
    if width is None or height is None:
        width = height = None
    if all(value is None for value in (duration, width, height, has_audio, fully_decoded)):
        return None
    return ObservableMediaMetadata(
        duration_ms=duration,
        width=width,
        height=height,
        has_audio=has_audio,
        fully_decoded=fully_decoded,
    )


def _budget_after(budget: ExecutionBudget, event: BudgetUpdate) -> ExecutionBudget:
    try:
        return ExecutionBudget(
            max_steps=budget.max_steps,
            max_tool_calls=budget.max_tool_calls,
            max_search_calls=budget.max_search_calls,
            max_edit_calls=budget.max_edit_calls,
            max_repeated_identical_actions=budget.max_repeated_identical_actions,
            max_structured_output_repairs=budget.max_structured_output_repairs,
            max_model_tokens=budget.max_model_tokens,
            max_wall_time_ms=budget.max_wall_time_ms,
            used_steps=budget.used_steps + event.used_steps_delta,
            used_tool_calls=budget.used_tool_calls + event.used_tool_calls_delta,
            used_search_calls=budget.used_search_calls + event.used_search_calls_delta,
            used_edit_calls=budget.used_edit_calls + event.used_edit_calls_delta,
            used_structured_output_repairs=(
                budget.used_structured_output_repairs + event.used_structured_output_repairs_delta
            ),
            used_model_tokens=budget.used_model_tokens + event.used_model_tokens_delta,
            used_wall_time_ms=budget.used_wall_time_ms + event.used_wall_time_ms_delta,
        )
    except ValidationError as error:
        raise InvalidEventError("budget update exceeds the M4B execution budget") from error


class M4BStateReducer:
    """Apply base and M4B events with one sequence/version increment per envelope."""

    version = "m4b-state-reducer-v1"

    @staticmethod
    def apply(state: M4BAgentState, envelope: M4BEventEnvelope) -> M4BAgentState:
        if envelope.task_id != state.task_input.task_id:
            raise TaskMismatchError("M4B event task does not match public TaskInput")
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
            raise TerminalStateError("cannot apply an M4B event after terminal state")

        update: dict[str, object] = {}
        event = envelope.event
        if isinstance(event, ToolObservation):
            update["tool_observations"] = (*state.tool_observations, event)
            if event.tool_name in _EDIT_TOOLS and event.status == "success":
                if len(event.artifacts) != 1:
                    raise InvalidEventError(
                        "successful deterministic editor must expose exactly one artifact"
                    )
                reference = event.artifacts[0]
                generated = WorkingMediaArtifact(
                    artifact_id=reference.artifact_id,
                    media_type=reference.media_type,
                    producing_tool=event.tool_name,
                    metadata=_media_metadata(dict(event.details)),
                    validation_status="tool_validated",
                )
                update["working_artifacts"] = WorkingArtifactState(
                    original_input_artifact=(state.working_artifacts.original_input_artifact),
                    current_working_artifact=generated,
                    latest_generated_artifact=generated,
                    final_candidate_artifact=None,
                )
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
            update["execution_budget"] = _budget_after(state.execution_budget, event)
        elif isinstance(event, WorkingArtifactValidationEvent):
            current = state.working_artifacts.current_working_artifact
            if event.artifact_id != current.artifact_id:
                raise InvalidEventError(
                    "working-artifact validation must target the current artifact"
                )
            validation_status = {
                "passed": "independently_validated",
                "failed": "failed",
                "inconclusive": "inconclusive",
            }[event.status]
            refreshed = current.model_copy(
                update={
                    "metadata": event.observable_metadata or current.metadata,
                    "validation_status": validation_status,
                }
            )
            update["working_artifacts"] = WorkingArtifactState(
                original_input_artifact=state.working_artifacts.original_input_artifact,
                current_working_artifact=refreshed,
                latest_generated_artifact=state.working_artifacts.latest_generated_artifact,
                final_candidate_artifact=refreshed if event.status == "passed" else None,
            )
        elif isinstance(event, StepOutcomeEvent):
            update["step_outcomes"] = (*state.step_outcomes, event.summary)
        elif isinstance(event, RecoveryOperationEvent):
            update["recovery_events"] = (*state.recovery_events, event)
        elif isinstance(event, SanitizedSystemErrorEvent):
            update["system_diagnostics"] = (*state.system_diagnostics, event)
        elif isinstance(event, TerminalEvent):
            update["terminal_event"] = event
        else:  # pragma: no cover - discriminated schema prevents this
            raise InvalidEventError(f"unsupported M4B event: {type(event).__name__}")

        update.update(
            processed_event_ids=(*state.processed_event_ids, envelope.event_id),
            last_sequence_no=envelope.sequence_no,
            state_version=state.state_version + 1,
        )
        return M4BAgentState.model_validate({**state.model_dump(mode="python"), **update})

    @classmethod
    def replay(
        cls,
        initial_state: M4BAgentState,
        events: Iterable[M4BEventEnvelope],
    ) -> M4BAgentState:
        state = initial_state
        for event in events:
            state = cls.apply(state, event)
        return state
