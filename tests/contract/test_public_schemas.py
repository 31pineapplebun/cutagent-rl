"""Public Pydantic schema contract tests."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.event import AgentEventEnvelope
from cutagent.schemas.task_input import DurationConstraint, TaskInput


def test_all_public_models_forbid_extra_fields(artifact_ref: ArtifactRef) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ArtifactRef.model_validate({**artifact_ref.model_dump(), "filesystem_path": "/secret"})

    with pytest.raises(ValidationError, match="extra_forbidden"):
        TaskInput.model_validate(
            {
                "task_id": "task-extra",
                "video_ref": artifact_ref.model_dump(),
                "instruction": "test",
                "split": "locked_test",
            }
        )


def test_duration_constraint_rejects_reversed_range() -> None:
    with pytest.raises(ValidationError, match="min_ms cannot exceed max_ms"):
        DurationConstraint(min_ms=2_000, max_ms=1_000)


def test_event_union_rejects_unknown_discriminator() -> None:
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        AgentEventEnvelope.model_validate(
            {
                "event_id": "event-1",
                "task_id": "task-1",
                "sequence_no": 1,
                "event": {"event_type": "benchmark_gold"},
                "emitted_by": "runtime",
                "created_at": datetime.now(UTC),
                "parent_state_version": 0,
            }
        )


def test_event_timestamp_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        AgentEventEnvelope.model_validate(
            {
                "event_id": "event-1",
                "task_id": "task-1",
                "sequence_no": 1,
                "event": {
                    "event_type": "terminal",
                    "status": "failed",
                    "reason": "test",
                },
                "emitted_by": "runtime",
                "created_at": datetime(2026, 1, 1),
                "parent_state_version": 0,
            }
        )
