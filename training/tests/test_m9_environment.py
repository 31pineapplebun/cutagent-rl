from __future__ import annotations

import json

import pytest
from cutagent_training.m9_contracts import M9EnvironmentInput, M9EnvironmentLabel
from cutagent_training.m9_environment import ControlledAgentEnvironment
from pydantic import ValidationError


def _target() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "decision_type": "tool",
        "tool_call": {
            "schema_version": "1.0",
            "tool_call_id": "call-1",
            "tool_name": "validate_media",
            "arguments": {
                "schema_version": "1.0",
                "input_artifact_id": "artifact-1",
                "decode_entire_video": True,
                "require_audio": False,
            },
        },
        "rationale": "Validate the current output.",
        "expected_observation": "A structured validation result.",
        "success_condition": "The media is valid.",
    }


def _input() -> M9EnvironmentInput:
    return M9EnvironmentInput(
        sample_id="sample-1",
        environment_snapshot_sha256="a" * 64,
        curriculum_stage="one_tool_step",
        action_contract="policy_decision",
        prompt="Observable task and state only.",
    )


def _label() -> M9EnvironmentLabel:
    return M9EnvironmentLabel(
        sample_id="sample-1",
        source_group_id="source-1",
        train_subsplit="fit",
        environment_snapshot_sha256="a" * 64,
        action_contract="policy_decision",
        expected_action=_target(),
        cached_observation={
            "schema_version": "1.0",
            "status": "success",
            "summary": "cached verified result",
        },
    )


def test_environment_reset_and_exact_verified_replay() -> None:
    environment = ControlledAgentEnvironment({"sample-1": _label()})
    assert environment.reset(_input()) == "Observable task and state only."
    observation, reward, exact, valid = environment.step(json.dumps(_target()))
    assert observation["status"] == "success"
    assert reward.total_environment_reward > 0
    assert exact is True
    assert valid is True


def test_environment_rejects_bad_action_and_repeat() -> None:
    environment = ControlledAgentEnvironment({"sample-1": _label()})
    environment.reset(_input())
    observation, reward, exact, valid = environment.step("not JSON")
    assert observation["status"] == "invalid"
    assert reward.total_environment_reward < 0
    assert exact is False
    assert valid is False
    _, repeated, _, _ = environment.step(json.dumps(_target()))
    assert repeated.repetition_penalty == -1.0


def test_group_resets_share_snapshot_but_state_is_independent() -> None:
    environments = [ControlledAgentEnvironment({"sample-1": _label()}) for _ in range(4)]
    for environment in environments:
        environment.reset(_input())
    results = [environment.step(json.dumps(_target())) for environment in environments]
    assert all(result[2] is True for result in results)


def test_public_input_forbids_hidden_reward_and_gold() -> None:
    for leaked in ("BenchmarkGold", '"split"', "expected_action", "failure_injection"):
        with pytest.raises(ValidationError):
            M9EnvironmentInput(
                sample_id="sample-1",
                environment_snapshot_sha256="a" * 64,
                curriculum_stage="single_decision",
                action_contract="policy_decision",
                prompt=f"observable {leaked}",
            )


def test_snapshot_mismatch_is_rejected() -> None:
    bad = _input().model_copy(update={"environment_snapshot_sha256": "b" * 64})
    environment = ControlledAgentEnvironment({"sample-1": _label()})
    with pytest.raises(ValueError, match="snapshot"):
        environment.reset(bad)
