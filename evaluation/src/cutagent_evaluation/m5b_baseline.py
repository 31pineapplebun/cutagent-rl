"""Access-controlled M5B prompt-only baseline contracts and aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cutagent.schemas.base import NonEmptyStr, SchemaModel
from cutagent.schemas.m4b_agent import M4BAgentTrajectory
from cutagent.schemas.task_input import TaskInput
from cutagent_evaluation.m5a_evaluator import (
    CutAgentBenchEvaluation,
    CutAgentBenchMetricsSummary,
    evaluate_trajectory,
    summarize_evaluations,
)
from cutagent_evaluation.m5a_schemas import BENCHMARK_VERSION, CutAgentBenchGold
from cutagent_evaluation.schemas import DatasetSplit

M5B_RUNNER_VERSION: Literal["m5b-prompt-baseline-runner-v1"] = "m5b-prompt-baseline-runner-v1"
M5B_PROMPT_VERSION: Literal["m4b-handoff-policy-v1"] = "m4b-handoff-policy-v1"
_PROTECTED = frozenset({DatasetSplit.LOCKED_TEST, DatasetSplit.ADVERSARIAL_TEST})


class M5BRunDeclaration(SchemaModel):
    runner_version: Literal["m5b-prompt-baseline-runner-v1"] = M5B_RUNNER_VERSION
    benchmark_version: Literal["cutagentbench-v0.1"] = BENCHMARK_VERSION
    split: DatasetSplit
    model_id: Literal["Qwen/Qwen3-VL-4B-Instruct"] = "Qwen/Qwen3-VL-4B-Instruct"
    model_revision: Literal["ebb281ec70b05090aa6165b016eac8ec08e71b17"] = (
        "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    )
    adapter_hash: NonEmptyStr = "none/base"
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    declared_metrics: tuple[NonEmptyStr, ...] = Field(min_length=1)
    access_reason: NonEmptyStr
    output_directory: NonEmptyStr
    resume: bool = True
    protocol_variant: Literal["handoff_only", "compact_recovery"] = "handoff_only"

    @model_validator(mode="after")
    def primary_protocol(self) -> M5BRunDeclaration:
        if self.protocol_variant == "compact_recovery" and self.split not in {
            DatasetSplit.DEV,
            DatasetSplit.VALIDATION,
            DatasetSplit.ADVERSARIAL_TEST,
        }:
            raise ValueError("compact recovery is restricted to declared recovery evaluations")
        return self


class M5BRunSummary(SchemaModel):
    declaration: M5BRunDeclaration
    status: Literal["PROVISIONAL_VALIDATION_ONLY", "OFFICIAL_PROTECTED_EVALUATION"]
    metrics: CutAgentBenchMetricsSummary
    task_count: int = Field(gt=0)
    resume_hits: int = Field(ge=0)
    deterministic_replay_count: int = Field(ge=0)
    private_trajectory_leak_count: int = Field(ge=0)
    protected_access_count: int = Field(ge=0)
    trajectory_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("deterministic_replay_count")
    @classmethod
    def replay_not_negative(cls, value: int) -> int:
        return value


def config_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def human_gate_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return isinstance(payload, dict) and payload.get("status") == "passed"


def assert_split_access(
    declaration: M5BRunDeclaration,
    *,
    human_gate_path: Path,
    access_record_path: Path | None,
) -> None:
    if declaration.split in {DatasetSplit.DEV, DatasetSplit.VALIDATION}:
        if access_record_path is not None:
            raise ValueError("development evaluation must not create a protected access record")
        return
    if declaration.split not in _PROTECTED:
        raise ValueError(
            "M5B evaluation supports dev, validation, or an authorized protected split"
        )
    if not human_gate_passed(human_gate_path):
        raise PermissionError("protected evaluation requires a passed real-human calibration gate")
    if access_record_path is None or not access_record_path.is_file():
        raise PermissionError("protected evaluation requires a predeclared access record")
    record = json.loads(access_record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("protected access record must be a JSON object")
    required = {
        "git_commit",
        "model_revision",
        "adapter_or_checkpoint_hash",
        "config_sha256",
        "benchmark_version",
        "access_reason",
        "declared_metrics",
        "split",
    }
    if not required.issubset(record):
        raise ValueError("protected access record is incomplete")
    if record["split"] != declaration.split.value:
        raise ValueError("protected access record split mismatch")
    if record["benchmark_version"] != declaration.benchmark_version:
        raise ValueError("protected access record benchmark mismatch")
    if record["model_revision"] != declaration.model_revision:
        raise ValueError("protected access record model mismatch")
    if record["config_sha256"] != declaration.config_sha256:
        raise ValueError("protected access record config mismatch")


def _load_json_list(path: Path) -> list[object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return payload


def load_cases(
    benchmark_root: Path,
    declaration: M5BRunDeclaration,
    *,
    human_gate_path: Path,
    access_record_path: Path | None = None,
) -> tuple[tuple[TaskInput, CutAgentBenchGold], ...]:
    assert_split_access(
        declaration,
        human_gate_path=human_gate_path,
        access_record_path=access_record_path,
    )
    public = tuple(
        TaskInput.model_validate(item)
        for item in _load_json_list(benchmark_root / "public" / f"{declaration.split.value}.json")
    )
    if declaration.split in _PROTECTED:
        private_path = (
            benchmark_root / "private" / "sealed" / f"{declaration.split.value}_gold.json"
        )
    else:
        private_path = benchmark_root / "private" / f"{declaration.split.value}_gold.json"
    private = tuple(
        CutAgentBenchGold.model_validate(item) for item in _load_json_list(private_path)
    )
    if tuple(item.task_id for item in public) != tuple(item.task_id for item in private):
        raise ValueError("public tasks and evaluator-private Gold differ")
    return tuple(zip(public, private, strict=True))


def evaluate_m5b_trajectories(
    trajectories: tuple[M4BAgentTrajectory, ...],
    gold: tuple[CutAgentBenchGold, ...],
) -> tuple[tuple[CutAgentBenchEvaluation, ...], CutAgentBenchMetricsSummary]:
    if tuple(item.task_input.task_id for item in trajectories) != tuple(
        item.task_id for item in gold
    ):
        raise ValueError("trajectory/Gold ordering differs")
    evaluations = tuple(
        evaluate_trajectory(trajectory, annotation)
        for trajectory, annotation in zip(trajectories, gold, strict=True)
    )
    return evaluations, summarize_evaluations(evaluations)


def trajectory_has_private_fields(trajectory: M4BAgentTrajectory) -> bool:
    serialized = trajectory.model_dump_json().casefold()
    return any(
        token in serialized
        for token in (
            "benchmarkgold",
            "source_group_id",
            '"split"',
            "gold_id",
            "acceptable_tool_sequences",
            "failure_injection",
            "expected_terminal_behavior",
        )
    )
