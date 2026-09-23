from __future__ import annotations

import hashlib
from typing import cast

import pytest
from cutagent_training.m7_contracts import (
    M7DecisionPreference,
    environment_snapshot_sha256,
)
from pydantic import JsonValue, ValidationError


def _decision(reason: str) -> dict[str, JsonValue]:
    return {
        "schema_version": "1.0",
        "decision_type": "cannot_complete",
        "reason": reason,
        "missing_evidence_or_capability": ["observable item"],
        "attempted_action_ids": [],
    }


def _pair(prompt: str = "public observable context") -> M7DecisionPreference:
    state = hashlib.sha256(b"state").hexdigest()
    context = hashlib.sha256(b"context").hexdigest()
    return M7DecisionPreference(
        pair_id="m7-pair-1",
        source_record_id="m6-record-1",
        task_id="task-1",
        source_group_id="train-source-1",
        train_subsplit="train",
        operation="decide",
        policy_context_sha256=context,
        public_state_sha256=state,
        environment_snapshot_sha256=environment_snapshot_sha256(
            task_id="task-1", public_state_sha256=state, policy_context_sha256=context
        ),
        prompt=prompt,
        chosen=_decision("correct observable refusal"),
        rejected=_decision("premature refusal"),
        preference_reason="premature_refusal",
        label_source="executed_oracle_vs_controlled_failure",
        rejected_evaluation="schema_valid_failure_aware_counterfactual",
        score_margin=1.0,
        confidence="high",
    )


def test_pair_has_same_snapshot_and_swift_format() -> None:
    pair = _pair()
    row = pair.swift_row()
    messages = cast(list[dict[str, JsonValue]], row["messages"])
    assert messages[0]["role"] == "user"
    assert row["rejected_response"] == pair.rejected_text


def test_pair_rejects_snapshot_mismatch() -> None:
    payload = _pair().model_dump(mode="python")
    payload["environment_snapshot_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    with pytest.raises(ValidationError):
        M7DecisionPreference.model_validate(payload)


def test_pair_rejects_private_prompt() -> None:
    with pytest.raises(ValidationError):
        _pair('public {"split":"train"}')


def test_pair_rejects_identical_behavior() -> None:
    payload = _pair().model_dump(mode="python")
    payload["rejected"] = payload["chosen"]
    with pytest.raises(ValidationError):
        M7DecisionPreference.model_validate(payload)
