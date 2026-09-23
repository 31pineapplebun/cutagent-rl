"""Bridge tool observations into the existing deterministic event reducer."""

from __future__ import annotations

from datetime import datetime

from cutagent.schemas.event import AgentEventEnvelope
from cutagent.schemas.state import AgentState
from cutagent.schemas.tools import ToolExecutionRecord


def tool_record_to_event(
    record: ToolExecutionRecord,
    state: AgentState,
    *,
    created_at: datetime,
) -> AgentEventEnvelope:
    return AgentEventEnvelope(
        event_id=f"event-{record.observation.call_id}",
        task_id=state.task_input.task_id,
        sequence_no=state.last_sequence_no + 1,
        event=record.observation,
        emitted_by="tool",
        created_at=created_at,
        parent_state_version=state.state_version,
    )
