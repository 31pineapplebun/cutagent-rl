"""Physically separated public-input and private-label contracts for M8 RM v1."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent_training.contracts import canonical_json

M8_DATASET_VERSION: Literal["m8-reward-model-v1"] = "m8-reward-model-v1"

CandidateSide = Literal["a", "b"]
FailureClass = Literal[
    "none",
    "wrong_tool_or_arguments",
    "premature_finish",
    "premature_refusal",
    "repeated_action",
    "invalid_recovery",
]

FAILURE_CLASSES: tuple[FailureClass, ...] = (
    "none",
    "wrong_tool_or_arguments",
    "premature_finish",
    "premature_refusal",
    "repeated_action",
    "invalid_recovery",
)

_FORBIDDEN_INPUT_MARKERS = (
    "benchmarkgold",
    "benchmark_gold",
    '"split"',
    "source_group_id",
    "ground_truth",
    "evaluator_metadata",
    '"sha256"',
    '"uri"',
    "file://",
    "preferred_candidate",
    "failure_class",
    "label_source",
)


class M8RMInput(SchemaModel):
    """One public pair input; it intentionally contains no target label or split."""

    input_id: Identifier
    dataset_version: Literal["m8-reward-model-v1"] = M8_DATASET_VERSION
    pair_id: Identifier
    task_id: Identifier
    environment_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    observable_context: dict[str, JsonValue]
    candidate_a: dict[str, JsonValue]
    candidate_b: dict[str, JsonValue]
    input_template_version: Literal["m8-rm-structured-state-v1"] = "m8-rm-structured-state-v1"

    @model_validator(mode="after")
    def reject_private_or_label_fields(self) -> M8RMInput:
        text = canonical_json(self.model_dump(mode="json", exclude={"input_id"})).casefold()
        if any(marker in text for marker in _FORBIDDEN_INPUT_MARKERS):
            raise ValueError("M8 RM input contains a private, label, hash, or filesystem marker")
        if canonical_json(self.candidate_a) == canonical_json(self.candidate_b):
            raise ValueError("RM pair candidates must differ")
        return self

    def render_candidate(self, side: CandidateSide) -> str:
        candidate = self.candidate_a if side == "a" else self.candidate_b
        payload = {
            "contract": "Score this candidate only from observable CutAgent state.",
            "observable_context": self.observable_context,
            "candidate_decision": candidate,
        }
        return canonical_json(payload)


class M8RMLabel(SchemaModel):
    """Evaluator/train-side label stored in a file separate from ``M8RMInput``."""

    label_id: Identifier
    dataset_version: Literal["m8-reward-model-v1"] = M8_DATASET_VERSION
    input_id: Identifier
    pair_id: Identifier
    source_group_id: Identifier
    train_subsplit: Literal["fit", "holdout"]
    preferred_candidate: CandidateSide
    failure_class_a: FailureClass
    failure_class_b: FailureClass
    label_source: Literal["executed_oracle_vs_controlled_failure"]
    confidence: Literal["high", "medium"]

    @model_validator(mode="after")
    def validate_preferred_is_nonfailure(self) -> M8RMLabel:
        preferred = (
            self.failure_class_a if self.preferred_candidate == "a" else self.failure_class_b
        )
        rejected = self.failure_class_b if self.preferred_candidate == "a" else self.failure_class_a
        if preferred != "none" or rejected == "none":
            raise ValueError("preferred RM candidate must be the nonfailure candidate")
        return self


class M8RMDatasetManifest(SchemaModel):
    dataset_version: Literal["m8-reward-model-v1"] = M8_DATASET_VERSION
    pair_count: int = Field(gt=0)
    fit_count: int = Field(gt=0)
    holdout_count: int = Field(gt=0)
    task_count: int = Field(gt=0)
    source_group_count: int = Field(gt=0)
    fit_source_groups: int = Field(gt=0)
    holdout_source_groups: int = Field(gt=0)
    source_group_overlap_count: Literal[0] = 0
    preferred_side_counts: dict[str, int]
    failure_class_counts: dict[str, int]
    public_input_leak_count: Literal[0] = 0
    physical_input_label_separation: Literal[True] = True
    aligned_input_label_count: int = Field(gt=0)
    public_inputs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    private_labels_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_preferences_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_input_description: NonEmptyStr

    @model_validator(mode="after")
    def validate_counts(self) -> M8RMDatasetManifest:
        if self.fit_count + self.holdout_count != self.pair_count:
            raise ValueError("M8 fit and holdout counts do not sum to pair count")
        if self.aligned_input_label_count != self.pair_count:
            raise ValueError("every M8 public input requires exactly one private label")
        return self


class M8RewardCheckpointManifest(SchemaModel):
    checkpoint_version: Literal["m8-qwen3vl-pairwise-failure-v1"] = "m8-qwen3vl-pairwise-failure-v1"
    model_id: NonEmptyStr
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: Literal["apache-2.0"]
    encoder_mode: Literal["frozen_structured_text_mean_pool"]
    hidden_size: int = Field(gt=0)
    failure_classes: tuple[FailureClass, ...]
    lambda_rank: float = Field(gt=0)
    lambda_failure: float = Field(gt=0)
    optimizer: Literal["numpy-adam"]
    optimizer_steps: int = Field(gt=0)
    seed: int = Field(ge=0)
    max_length: int = Field(gt=0)
    dataset_inputs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_labels_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class M8AgentCandidate(SchemaModel):
    candidate_id: Identifier
    decision: dict[str, JsonValue]


class M8AgentCandidateInput(SchemaModel):
    """Public validation candidate set passed to rules/RM without its label."""

    case_id: Identifier
    task_id: Identifier
    observable_context: dict[str, JsonValue]
    candidates: tuple[M8AgentCandidate, ...] = Field(min_length=2, max_length=8)
    candidate_template_version: Literal["m8-agent-candidates-v1"] = "m8-agent-candidates-v1"

    @model_validator(mode="after")
    def reject_private_fields(self) -> M8AgentCandidateInput:
        if len({item.candidate_id for item in self.candidates}) != len(self.candidates):
            raise ValueError("M8 Agent candidate IDs must be unique")
        text = canonical_json(self.model_dump(mode="json")).casefold()
        if any(marker in text for marker in _FORBIDDEN_INPUT_MARKERS):
            raise ValueError("M8 Agent candidate input contains private data")
        return self

    def render_candidate(self, candidate: M8AgentCandidate) -> str:
        return canonical_json(
            {
                "contract": "Score this Agent decision from public observable state only.",
                "observable_context": self.observable_context,
                "candidate_decision": candidate.decision,
            }
        )


class M8AgentCandidateLabel(SchemaModel):
    case_id: Identifier
    task_id: Identifier
    split: Literal["validation"]
    acceptable_candidate_ids: tuple[Identifier, ...] = Field(min_length=1)
    label_source: Literal["offline_objective_initial_action_contract"]

    @field_validator("acceptable_candidate_ids")
    @classmethod
    def unique_acceptable_candidates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("acceptable candidate IDs must be unique")
        return value


def sha256_rows(rows: tuple[SchemaModel, ...]) -> str:
    payload = "\n".join(item.model_dump_json() for item in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def summarize_failure_classes(labels: tuple[M8RMLabel, ...]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in labels:
        counts.update((item.failure_class_a, item.failure_class_b))
    return dict(counts)
