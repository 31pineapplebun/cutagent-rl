"""Evaluator-only annotations that must never enter a policy context."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, JsonValue, field_validator, model_validator

from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel


class DatasetSplit(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    VALIDATION = "validation"
    LOCKED_TEST = "locked_test"
    ADVERSARIAL_TEST = "adversarial_test"


class EvidenceAnnotation(SchemaModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    description: NonEmptyStr

    @model_validator(mode="after")
    def validate_interval(self) -> "EvidenceAnnotation":
        if self.start_ms >= self.end_ms:
            raise ValueError("evidence start_ms must be before end_ms")
        return self


class TaskAnnotation(SchemaModel):
    annotation_id: Identifier
    task_id: Identifier
    source_group_id: Identifier
    proposed_split: DatasetSplit
    annotator_ids: tuple[Identifier, ...]
    expected_evidence: tuple[EvidenceAnnotation, ...] = ()
    ground_truth: dict[str, JsonValue] = Field(default_factory=dict)
    evaluator_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class BenchmarkGold(SchemaModel):
    gold_id: Identifier
    task_id: Identifier
    source_group_id: Identifier
    split: DatasetSplit
    expected_evidence: tuple[EvidenceAnnotation, ...]
    ground_truth: dict[str, JsonValue]
    evaluator_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    adjudication_version: NonEmptyStr
    frozen_at: datetime

    @field_validator("frozen_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("frozen_at must be timezone-aware")
        return value
