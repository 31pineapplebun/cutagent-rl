"""Leakage-auditable contracts for the controlled M9 GRPO environment."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from cutagent.schemas.agent import POLICY_DECISION_ADAPTER
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.m4b_agent import RECOVERY_DECISION_ADAPTER

M9_DATASET_VERSION: Literal["m9-controlled-agent-v1"] = "m9-controlled-agent-v1"
M9_ENVIRONMENT_VERSION: Literal["m9-cached-verified-environment-v1"] = (
    "m9-cached-verified-environment-v1"
)
M9_REWARD_VERSION: Literal["m9-objective-reward-v1"] = "m9-objective-reward-v1"

CurriculumStage = Literal[
    "single_decision",
    "one_tool_step",
    "two_step_sequence",
    "observable_recovery",
]
ActionContract = Literal["policy_decision", "recovery_decision"]


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class M9EnvironmentInput(SchemaModel):
    """Public reset state passed to the policy; contains no reward or expected action."""

    sample_id: Identifier
    dataset_version: Literal["m9-controlled-agent-v1"] = M9_DATASET_VERSION
    environment_version: Literal["m9-cached-verified-environment-v1"] = M9_ENVIRONMENT_VERSION
    environment_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    curriculum_stage: CurriculumStage
    action_contract: ActionContract
    prompt: NonEmptyStr
    maximum_steps: int = Field(default=1, ge=1, le=2)

    @field_validator("prompt")
    @classmethod
    def public_prompt_is_clean(cls, value: str) -> str:
        forbidden = (
            "benchmarkgold",
            "benchmark_gold",
            '"split"',
            "source_group_id",
            "ground_truth",
            "expected_action",
            "failure_injection",
            "reward_breakdown",
            "evaluator_metadata",
            '"sha256"',
            "file://",
        )
        folded = value.casefold()
        if any(token in folded for token in forbidden):
            raise ValueError("M9 policy prompt contains evaluator-private or hidden reward data")
        return value


class M9EnvironmentLabel(SchemaModel):
    """Train-side hidden environment state, physically separated from model inputs."""

    sample_id: Identifier
    source_group_id: Identifier
    train_subsplit: Literal["fit", "holdout"]
    environment_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    action_contract: ActionContract
    expected_action: dict[str, JsonValue]
    verified_evidence_ids: tuple[Identifier, ...] = ()
    cached_observation: dict[str, JsonValue]

    @model_validator(mode="after")
    def action_matches_contract(self) -> M9EnvironmentLabel:
        if self.action_contract == "policy_decision":
            POLICY_DECISION_ADAPTER.validate_python(self.expected_action)
        else:
            RECOVERY_DECISION_ADAPTER.validate_python(self.expected_action)
        return self


class M9RewardBreakdown(SchemaModel):
    reward_version: Literal["m9-objective-reward-v1"] = M9_REWARD_VERSION
    structured_validity_reward: float
    decision_kind_reward: float
    tool_or_operation_reward: float
    argument_reward: float
    observable_progress_reward: float
    rm_reward: float
    malformed_penalty: float = Field(le=0)
    invalid_action_penalty: float = Field(le=0)
    premature_terminal_penalty: float = Field(le=0)
    repetition_penalty: float = Field(le=0)
    unsafe_action_penalty: float = Field(le=0)
    total_environment_reward: float

    @model_validator(mode="after")
    def total_is_exact_sum(self) -> M9RewardBreakdown:
        components = (
            self.structured_validity_reward,
            self.decision_kind_reward,
            self.tool_or_operation_reward,
            self.argument_reward,
            self.observable_progress_reward,
            self.rm_reward,
            self.malformed_penalty,
            self.invalid_action_penalty,
            self.premature_terminal_penalty,
            self.repetition_penalty,
            self.unsafe_action_penalty,
        )
        if abs(sum(components) - self.total_environment_reward) > 1e-8:
            raise ValueError("M9 total environment reward differs from component sum")
        return self


class M9RolloutTrace(SchemaModel):
    rollout_id: Identifier
    group_id: Identifier
    sample_id: Identifier
    environment_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    group_index: int = Field(ge=0, lt=4)
    raw_completion: str
    parsed_action: dict[str, JsonValue] | None = None
    public_observation: dict[str, JsonValue]
    breakdown: M9RewardBreakdown
    exact_action_match: bool
    structured_valid: bool
    completion_length: int = Field(ge=0)
    protected_access_count: Literal[0] = 0


class M9DatasetManifest(SchemaModel):
    dataset_version: Literal["m9-controlled-agent-v1"] = M9_DATASET_VERSION
    public_input_count: int = Field(gt=0)
    private_label_count: int = Field(gt=0)
    fit_count: int = Field(gt=0)
    holdout_count: int = Field(gt=0)
    fit_source_groups: int = Field(gt=0)
    holdout_source_groups: int = Field(gt=0)
    source_group_overlap_count: Literal[0] = 0
    curriculum_counts: dict[str, int]
    public_prompt_leak_count: Literal[0] = 0
    protected_access_count: Literal[0] = 0
    public_inputs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    private_labels_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def content_sha256(rows: list[SchemaModel]) -> str:
    payload = "\n".join(item.model_dump_json() for item in rows).encode()
    return hashlib.sha256(payload).hexdigest()
