"""Decision-level, same-state preference contracts for M7."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from cutagent.schemas.agent import POLICY_DECISION_ADAPTER, ReplanDecision
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.m4b_agent import RECOVERY_DECISION_ADAPTER
from cutagent_training.contracts import TrainingMessage, canonical_json

M7_DATASET_VERSION: Literal["m7-decision-preference-v1"] = "m7-decision-preference-v1"

PreferenceReason = Literal[
    "wrong_tool_or_arguments",
    "premature_finish",
    "premature_refusal",
    "repeated_action",
    "invalid_recovery",
]


def environment_snapshot_sha256(
    *, task_id: str, public_state_sha256: str, policy_context_sha256: str
) -> str:
    value = f"{task_id}:{public_state_sha256}:{policy_context_sha256}".encode()
    return hashlib.sha256(value).hexdigest()


class M7DecisionPreference(SchemaModel):
    """One chosen/rejected decision pair from an identical observable snapshot."""

    pair_id: Identifier
    dataset_version: Literal["m7-decision-preference-v1"] = M7_DATASET_VERSION
    source_record_id: Identifier
    task_id: Identifier
    source_group_id: Identifier
    train_subsplit: Literal["train", "holdout"]
    operation: Literal["decide", "replan", "recover"]
    policy_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt: NonEmptyStr
    chosen: dict[str, JsonValue]
    rejected: dict[str, JsonValue]
    preference_reason: PreferenceReason
    label_source: Literal["executed_oracle_vs_controlled_failure"]
    rejected_evaluation: Literal["schema_valid_failure_aware_counterfactual"]
    score_margin: float = Field(gt=0, le=1)
    confidence: Literal["high", "medium"]

    @model_validator(mode="after")
    def validate_pair(self) -> M7DecisionPreference:
        expected_snapshot = environment_snapshot_sha256(
            task_id=self.task_id,
            public_state_sha256=self.public_state_sha256,
            policy_context_sha256=self.policy_context_sha256,
        )
        if self.environment_snapshot_sha256 != expected_snapshot:
            raise ValueError("M7 environment snapshot hash is inconsistent")
        if canonical_json(self.chosen) == canonical_json(self.rejected):
            raise ValueError("M7 chosen and rejected decisions must differ behaviorally")
        if self.operation == "recover":
            RECOVERY_DECISION_ADAPTER.validate_python(self.chosen)
            RECOVERY_DECISION_ADAPTER.validate_python(self.rejected)
        else:
            chosen = POLICY_DECISION_ADAPTER.validate_python(self.chosen)
            rejected = POLICY_DECISION_ADAPTER.validate_python(self.rejected)
            if self.operation == "replan" and not (
                isinstance(chosen, ReplanDecision) and isinstance(rejected, ReplanDecision)
            ):
                raise ValueError("M7 replan pairs require two ReplanDecision values")
            if self.operation == "decide" and (
                isinstance(chosen, ReplanDecision) or isinstance(rejected, ReplanDecision)
            ):
                raise ValueError("M7 ordinary decisions cannot contain ReplanDecision")
        forbidden = (
            "benchmarkgold",
            "benchmark_gold",
            '"split"',
            "source_group_id",
            "ground_truth",
            "evaluator_metadata",
            '"sha256"',
            '"uri"',
            "file://",
        )
        if any(value in self.prompt.casefold() for value in forbidden):
            raise ValueError("M7 policy prompt contains private or filesystem data")
        return self

    @property
    def chosen_text(self) -> str:
        return canonical_json(self.chosen)

    @property
    def rejected_text(self) -> str:
        return canonical_json(self.rejected)

    @property
    def model_pair_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json([self.prompt, self.chosen, self.rejected]).encode()
        ).hexdigest()

    def swift_row(self) -> dict[str, JsonValue]:
        return {
            "messages": [
                TrainingMessage(role="user", content=self.prompt).model_dump(mode="json"),
                TrainingMessage(role="assistant", content=self.chosen_text).model_dump(mode="json"),
            ],
            "rejected_response": self.rejected_text,
        }


class M7PreferenceManifest(SchemaModel):
    dataset_version: Literal["m7-decision-preference-v1"] = M7_DATASET_VERSION
    pair_count: int = Field(gt=0)
    task_count: int = Field(gt=0)
    source_group_count: int = Field(gt=0)
    train_count: int = Field(ge=0)
    holdout_count: int = Field(ge=0)
    preference_reason_counts: dict[str, int]
    chosen_mean_characters: float = Field(gt=0)
    rejected_mean_characters: float = Field(gt=0)
    chosen_shorter_fraction: float = Field(ge=0, le=1)
    exact_duplicate_pair_count: Literal[0] = 0
    same_snapshot_compliance: float = Field(default=1.0, ge=1.0, le=1.0)
    policy_input_leak_count: Literal[0] = 0
    invalid_decision_count: Literal[0] = 0
    pairs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_range_note: NonEmptyStr


def build_m7_manifest(
    pairs: tuple[M7DecisionPreference, ...],
) -> M7PreferenceManifest:
    if not pairs:
        raise ValueError("M7 preference dataset cannot be empty")
    fingerprints = Counter(item.model_pair_sha256 for item in pairs)
    duplicate_count = sum(count - 1 for count in fingerprints.values())
    if duplicate_count:
        raise ValueError("M7 exact preference pairs must be unique")
    chosen_lengths = [len(item.chosen_text) for item in pairs]
    rejected_lengths = [len(item.rejected_text) for item in pairs]
    payload = "\n".join(item.model_dump_json() for item in pairs).encode()
    return M7PreferenceManifest(
        pair_count=len(pairs),
        task_count=len({item.task_id for item in pairs}),
        source_group_count=len({item.source_group_id for item in pairs}),
        train_count=sum(item.train_subsplit == "train" for item in pairs),
        holdout_count=sum(item.train_subsplit == "holdout" for item in pairs),
        preference_reason_counts=dict(Counter(item.preference_reason for item in pairs)),
        chosen_mean_characters=sum(chosen_lengths) / len(chosen_lengths),
        rejected_mean_characters=sum(rejected_lengths) / len(rejected_lengths),
        chosen_shorter_fraction=sum(
            chosen < rejected
            for chosen, rejected in zip(chosen_lengths, rejected_lengths, strict=True)
        )
        / len(pairs),
        pairs_sha256=hashlib.sha256(payload).hexdigest(),
        target_range_note=(
            "The 2k-5k research target was not padded; pairs are limited to unique, "
            "same-snapshot controlled behavioral counterfactuals from verified M6 records."
        ),
    )
