from __future__ import annotations

import hashlib

import pytest
from cutagent_training.contracts import M6AgentSFTRecord, TrainingMessage, canonical_json
from pydantic import JsonValue, ValidationError


def _record(prompt: str = "public observable context") -> M6AgentSFTRecord:
    target: dict[str, JsonValue] = {
        "decision_type": "cannot_complete",
        "reason": "evidence is unavailable",
        "missing_evidence_or_capability": ["requested evidence"],
        "attempted_action_ids": [],
    }
    return M6AgentSFTRecord(
        record_id="m6-record-1",
        trajectory_id="trajectory-1",
        task_id="task-1",
        source_group_id="train-source-1",
        train_subsplit="train",
        operation="decide",
        policy_context_sha256=hashlib.sha256(b"context").hexdigest(),
        public_state_sha256=hashlib.sha256(b"state").hexdigest(),
        target_source="correct_cannot_complete",
        messages=(
            TrainingMessage(role="user", content=prompt),
            TrainingMessage(role="assistant", content=canonical_json(target)),
        ),
        target=target,
    )


def test_m6_record_accepts_typed_policy_decision() -> None:
    assert _record().target["decision_type"] == "cannot_complete"


@pytest.mark.parametrize(
    "leak",
    ("BenchmarkGold", '"split":"train"', "source_group_id", '"sha256":"abc"', "file://x"),
)
def test_m6_record_rejects_private_or_filesystem_model_input(leak: str) -> None:
    with pytest.raises(ValidationError):
        _record(f"public context {leak}")


def test_m6_record_rejects_tool_observation_target() -> None:
    record = _record()
    payload = record.model_dump(mode="python")
    payload["target"] = {
        "event_type": "tool_observation",
        "call_id": "call-1",
        "tool_name": "trim_video",
        "status": "success",
        "public_summary": "ok",
    }
    payload["messages"][1]["content"] = canonical_json(payload["target"])
    with pytest.raises(ValidationError):
        M6AgentSFTRecord.model_validate(payload)
