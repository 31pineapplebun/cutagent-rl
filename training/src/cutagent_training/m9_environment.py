"""Resettable, cached-verified short-horizon Agent environment for M9."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import cast

from pydantic import JsonValue, ValidationError

from cutagent.schemas.agent import POLICY_DECISION_ADAPTER
from cutagent.schemas.m4b_agent import RECOVERY_DECISION_ADAPTER
from cutagent_training.m9_contracts import (
    M9EnvironmentInput,
    M9EnvironmentLabel,
    M9RewardBreakdown,
    canonical_json,
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_UNSAFE = re.compile(r"(?:shell|ffmpeg\s+-|powershell|cmd\.exe|/etc/|/root/|\.\.[\\/])", re.I)


def completion_text(value: object) -> str:
    """Normalize TRL plain or conversational completion values."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                content = item.get("content")
                if isinstance(content, str):
                    texts.append(content)
        return "\n".join(texts)
    return str(value)


def extract_json_object(text: str) -> dict[str, JsonValue] | None:
    stripped = text.strip()
    candidates = [stripped]
    match = _JSON_OBJECT.search(stripped)
    if match is not None and match.group(0) != stripped:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            return cast(dict[str, JsonValue], value)
    return None


def _decision_kind(action: Mapping[str, object]) -> object:
    return action.get("decision_type", action.get("recovery_type"))


def _tool_or_operation(action: Mapping[str, object]) -> object:
    tool_call = action.get("tool_call")
    if isinstance(tool_call, Mapping):
        return tool_call.get("tool_name")
    return _decision_kind(action)


def _arguments(action: Mapping[str, object]) -> Mapping[str, object]:
    tool_call = action.get("tool_call")
    if isinstance(tool_call, Mapping):
        arguments = tool_call.get("arguments")
        if isinstance(arguments, Mapping):
            return arguments
    return {}


def _argument_similarity(actual: Mapping[str, object], expected: Mapping[str, object]) -> float:
    if not expected:
        return 1.0 if not actual else 0.0
    keys = set(actual) | set(expected)
    if not keys:
        return 1.0
    return sum(actual.get(key) == expected.get(key) for key in keys) / len(keys)


class ControlledAgentEnvironment:
    """One-action resettable environment using verified cached M6 operations."""

    def __init__(self, labels: Mapping[str, M9EnvironmentLabel], *, rm_coefficient: float = 0.0):
        if rm_coefficient < 0 or rm_coefficient > 0.1:
            raise ValueError("M9 RM coefficient must remain in [0, 0.1]")
        self._labels = dict(labels)
        self._rm_coefficient = rm_coefficient
        self._current: M9EnvironmentInput | None = None
        self._acted = False

    def reset(self, public_input: M9EnvironmentInput) -> str:
        label = self._labels.get(public_input.sample_id)
        if label is None:
            raise KeyError("unknown M9 environment sample")
        if label.environment_snapshot_sha256 != public_input.environment_snapshot_sha256:
            raise ValueError("public/private environment snapshot mismatch")
        if label.action_contract != public_input.action_contract:
            raise ValueError("public/private action contract mismatch")
        self._current = public_input
        self._acted = False
        return public_input.prompt

    def step(self, raw_action: str) -> tuple[dict[str, JsonValue], M9RewardBreakdown, bool, bool]:
        if self._current is None:
            raise RuntimeError("environment must be reset before step")
        if self._acted:
            return self._repeated_action(raw_action)
        self._acted = True
        label = self._labels[self._current.sample_id]
        parsed = extract_json_object(raw_action)
        unsafe = bool(_UNSAFE.search(raw_action))
        valid = False
        if parsed is not None:
            try:
                if label.action_contract == "policy_decision":
                    POLICY_DECISION_ADAPTER.validate_python(parsed)
                else:
                    RECOVERY_DECISION_ADAPTER.validate_python(parsed)
                valid = True
            except ValidationError:
                valid = False

        expected = label.expected_action
        exact = valid and canonical_json(parsed) == canonical_json(expected)
        kind_match = parsed is not None and _decision_kind(parsed) == _decision_kind(expected)
        operation_match = parsed is not None and _tool_or_operation(parsed) == _tool_or_operation(
            expected
        )
        argument_similarity = (
            _argument_similarity(_arguments(parsed), _arguments(expected))
            if parsed is not None and operation_match
            else 0.0
        )
        premature = (
            parsed is not None
            and _decision_kind(parsed)
            in {
                "finish",
                "cannot_complete",
                "cannot_recover",
            }
            and not kind_match
        )
        malformed_penalty = 0.0 if valid else -1.0
        invalid_penalty = 0.0 if valid else -0.25
        premature_penalty = -0.75 if premature else 0.0
        unsafe_penalty = -2.0 if unsafe else 0.0
        structured = 0.25 if valid else 0.0
        kind_reward = 0.4 if kind_match else 0.0
        operation_reward = 0.6 if operation_match else 0.0
        argument_reward = 0.5 * argument_similarity
        progress_reward = 1.25 if exact else (0.25 if valid and operation_match else 0.0)
        # M8 was a negative result. The primary M9 experiment explicitly disables it.
        rm_reward = 0.0 * self._rm_coefficient
        total = sum(
            (
                structured,
                kind_reward,
                operation_reward,
                argument_reward,
                progress_reward,
                rm_reward,
                malformed_penalty,
                invalid_penalty,
                premature_penalty,
                unsafe_penalty,
            )
        )
        breakdown = M9RewardBreakdown(
            structured_validity_reward=structured,
            decision_kind_reward=kind_reward,
            tool_or_operation_reward=operation_reward,
            argument_reward=argument_reward,
            observable_progress_reward=progress_reward,
            rm_reward=rm_reward,
            malformed_penalty=malformed_penalty,
            invalid_action_penalty=invalid_penalty,
            premature_terminal_penalty=premature_penalty,
            repetition_penalty=0.0,
            unsafe_action_penalty=unsafe_penalty,
            total_environment_reward=total,
        )
        if exact:
            observation = dict(label.cached_observation)
        else:
            observation = {
                "schema_version": "1.0",
                "status": "invalid" if not valid else "error",
                "summary": (
                    "structured action rejected by the cached verified environment"
                    if valid
                    else "action did not satisfy the public structured contract"
                ),
            }
        return observation, breakdown, exact, valid

    def _repeated_action(
        self, raw_action: str
    ) -> tuple[dict[str, JsonValue], M9RewardBreakdown, bool, bool]:
        parsed = extract_json_object(raw_action)
        breakdown = M9RewardBreakdown(
            structured_validity_reward=0.0,
            decision_kind_reward=0.0,
            tool_or_operation_reward=0.0,
            argument_reward=0.0,
            observable_progress_reward=0.0,
            rm_reward=0.0,
            malformed_penalty=0.0,
            invalid_action_penalty=0.0,
            premature_terminal_penalty=0.0,
            repetition_penalty=-1.0,
            unsafe_action_penalty=-2.0 if _UNSAFE.search(raw_action) else 0.0,
            total_environment_reward=-3.0 if _UNSAFE.search(raw_action) else -1.0,
        )
        return (
            {"schema_version": "1.0", "status": "error", "summary": "step budget exhausted"},
            breakdown,
            False,
            parsed is not None,
        )


def parsed_action_dict(raw_action: str) -> dict[str, JsonValue] | None:
    return extract_json_object(raw_action)
