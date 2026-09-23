"""Typed, leakage-auditable post-training data contracts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from cutagent.schemas.agent import POLICY_DECISION_ADAPTER, ReplanDecision
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.m4b_agent import RECOVERY_DECISION_ADAPTER

M6_DATASET_VERSION: Literal["m6-agent-sft-v1"] = "m6-agent-sft-v1"


def canonical_json(value: object) -> str:
    """Return the canonical JSON representation used for all dataset hashes."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class TrainingMessage(SchemaModel):
    role: Literal["user", "assistant"]
    content: NonEmptyStr


class M6AgentSFTRecord(SchemaModel):
    """One observable state to next-policy-decision supervised record."""

    record_id: Identifier
    dataset_version: Literal["m6-agent-sft-v1"] = M6_DATASET_VERSION
    trajectory_id: Identifier
    task_id: Identifier
    source_group_id: Identifier
    train_subsplit: Literal["train", "holdout"]
    operation: Literal["decide", "replan", "recover"]
    policy_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_source: Literal[
        "verified_success",
        "executed_oracle",
        "verified_repair",
        "observable_recovery",
        "correct_cannot_complete",
        "valid_finish",
    ]
    verification_evidence_ids: tuple[Identifier, ...] = ()
    messages: tuple[TrainingMessage, TrainingMessage]
    target: dict[str, JsonValue]

    @field_validator("verification_evidence_ids")
    @classmethod
    def evidence_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("verification evidence identifiers must be unique")
        return value

    @model_validator(mode="after")
    def validate_model_io(self) -> M6AgentSFTRecord:
        if self.messages[0].role != "user" or self.messages[1].role != "assistant":
            raise ValueError("M6 messages must be exactly user then assistant")
        expected = canonical_json(self.target)
        if self.messages[1].content != expected:
            raise ValueError("assistant message must be the canonical decision target")
        if self.operation == "recover":
            RECOVERY_DECISION_ADAPTER.validate_python(self.target)
        else:
            decision = POLICY_DECISION_ADAPTER.validate_python(self.target)
            if self.operation == "replan" and not isinstance(decision, ReplanDecision):
                raise ValueError("replan records require ReplanDecision targets")
            if self.operation == "decide" and isinstance(decision, ReplanDecision):
                raise ValueError("ordinary decide records cannot target ReplanDecision")
        forbidden = (
            "benchmarkgold",
            "benchmark_gold",
            '"split"',
            "source_group_id",
            "ground_truth",
            "evaluator_metadata",
            '"uri"',
            '"sha256"',
            "file://",
        )
        prompt = self.messages[0].content.casefold()
        if any(item in prompt for item in forbidden):
            raise ValueError("M6 model input contains evaluator-private or filesystem data")
        target_text = self.messages[1].content.casefold()
        if '"event_type":"tool_observation"' in target_text:
            raise ValueError("ToolObservation cannot be an SFT target")
        return self

    @property
    def model_io_sha256(self) -> str:
        payload = [item.model_dump(mode="json") for item in self.messages]
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


class M6DatasetManifest(SchemaModel):
    dataset_version: Literal["m6-agent-sft-v1"] = M6_DATASET_VERSION
    record_count: int = Field(gt=0)
    task_count: int = Field(gt=0)
    source_group_count: int = Field(gt=0)
    train_count: int = Field(ge=0)
    holdout_count: int = Field(ge=0)
    operation_counts: dict[str, int]
    target_source_counts: dict[str, int]
    duplicate_model_io_count: int = Field(ge=0)
    invalid_target_count: Literal[0] = 0
    policy_input_leak_count: Literal[0] = 0
    executable_audit_count: int = Field(ge=0)
    executable_audit_failures: Literal[0] = 0
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_range_note: NonEmptyStr


def build_m6_manifest(
    records: tuple[M6AgentSFTRecord, ...], *, executable_audit_count: int
) -> M6DatasetManifest:
    if not records:
        raise ValueError("M6 dataset cannot be empty")
    io_counts = Counter(item.model_io_sha256 for item in records)
    serialized = "\n".join(item.model_dump_json() for item in records).encode()
    return M6DatasetManifest(
        record_count=len(records),
        task_count=len({item.task_id for item in records}),
        source_group_count=len({item.source_group_id for item in records}),
        train_count=sum(item.train_subsplit == "train" for item in records),
        holdout_count=sum(item.train_subsplit == "holdout" for item in records),
        operation_counts=dict(Counter(item.operation for item in records)),
        target_source_counts=dict(Counter(item.target_source for item in records)),
        duplicate_model_io_count=sum(count - 1 for count in io_counts.values()),
        executable_audit_count=executable_audit_count,
        records_sha256=hashlib.sha256(serialized).hexdigest(),
        target_range_note=(
            "The initial 5k-15k research target was not padded; this manifest records only "
            "unique, executed train-source decisions available at M6."
        ),
    )
