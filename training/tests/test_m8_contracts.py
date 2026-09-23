from __future__ import annotations

import hashlib

import pytest
from cutagent_training.m8_contracts import (
    M8AgentCandidate,
    M8AgentCandidateInput,
    M8RMInput,
    M8RMLabel,
)
from pydantic import JsonValue, ValidationError


def _input(context: dict[str, JsonValue] | None = None) -> M8RMInput:
    return M8RMInput(
        input_id="m8-input-1",
        pair_id="m7-pair-1",
        task_id="task-1",
        environment_snapshot_id=hashlib.sha256(b"snapshot").hexdigest(),
        observable_context=context or {"task": {"instruction": "trim the red scene"}},
        candidate_a={"decision_type": "tool", "tool_call": {"tool_name": "trim_video"}},
        candidate_b={"decision_type": "cannot_complete", "reason": "premature"},
    )


def test_public_input_renders_one_candidate_without_labels() -> None:
    record = _input()
    rendered = record.render_candidate("a")
    assert "trim_video" in rendered
    assert "cannot_complete" not in rendered
    assert "preferred_candidate" not in rendered
    assert "failure_class" not in rendered


@pytest.mark.parametrize(
    "context",
    [
        {"split": "train"},
        {"source_group_id": "private-source"},
        {"uri": "file:///etc/passwd"},
        {"failure_class": "wrong_tool"},
    ],
)
def test_public_input_rejects_private_or_label_markers(
    context: dict[str, JsonValue],
) -> None:
    with pytest.raises(ValidationError):
        _input(context)


def test_private_label_requires_nonfailure_preferred_candidate() -> None:
    with pytest.raises(ValidationError):
        M8RMLabel(
            label_id="m8-label-1",
            input_id="m8-input-1",
            pair_id="m7-pair-1",
            source_group_id="train-source-1",
            train_subsplit="fit",
            preferred_candidate="a",
            failure_class_a="premature_finish",
            failure_class_b="none",
            label_source="executed_oracle_vs_controlled_failure",
            confidence="high",
        )


def test_agent_candidate_input_rejects_split_and_duplicate_ids() -> None:
    candidate = M8AgentCandidate(candidate_id="candidate-1", decision={"decision_type": "tool"})
    with pytest.raises(ValidationError):
        M8AgentCandidateInput(
            case_id="case-1",
            task_id="task-1",
            observable_context={"split": "validation"},
            candidates=(candidate, candidate),
        )
