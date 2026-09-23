"""Discriminated Agent events used by the M0 reducer contract."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, JsonValue, field_validator, model_validator

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel

ToolStatus = Literal["success", "invalid", "timeout", "error"]
VerificationStatus = Literal["passed", "failed", "inconclusive"]
TerminalStatus = Literal["succeeded", "failed", "unfulfillable", "budget_exhausted", "system_error"]
TerminalReasonCode = Literal[
    "SUCCESS",
    "CANNOT_COMPLETE",
    "BUDGET_EXHAUSTED",
    "LOOP_DETECTED",
    "MODEL_OUTPUT_FAILURE",
    "SYSTEM_ERROR",
]
PlanNodeStatus = Literal[
    "pending",
    "ready",
    "running",
    "succeeded",
    "failed",
    "blocked",
    "skipped",
]
PlanStepStatus = PlanNodeStatus


class PlanNode(SchemaModel):
    """A dependency-aware executable plan node.

    ``step_id``/``description`` remain accepted as validation aliases so M0
    manifests can still be replayed, while all new serialization uses the M4A
    ``node_id``/``subgoal`` contract.
    """

    node_id: Identifier = Field(validation_alias=AliasChoices("node_id", "step_id"))
    subgoal: NonEmptyStr = Field(validation_alias=AliasChoices("subgoal", "description"))
    dependencies: tuple[Identifier, ...] = ()
    status: PlanNodeStatus = "pending"
    expected_evidence: tuple[NonEmptyStr, ...] = ()
    preferred_capability: Identifier | None = None
    completion_criteria: tuple[NonEmptyStr, ...] = ()

    @field_validator("status", mode="before")
    @classmethod
    def migrate_legacy_done(cls, value: object) -> object:
        return "succeeded" if value == "done" else value

    @field_validator("dependencies")
    @classmethod
    def unique_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("plan node dependencies must be unique")
        return value

    @model_validator(mode="after")
    def reject_self_dependency(self) -> "PlanNode":
        if self.node_id in self.dependencies:
            raise ValueError("plan node cannot depend on itself")
        return self

    @property
    def step_id(self) -> str:
        return self.node_id

    @property
    def description(self) -> str:
        return self.subgoal


PlanStep = PlanNode


class ToolObservation(SchemaModel):
    event_type: Literal["tool_observation"] = "tool_observation"
    call_id: Identifier
    tool_name: Identifier
    status: ToolStatus
    public_summary: NonEmptyStr
    details: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    error_code: Identifier | None = None

    @model_validator(mode="after")
    def validate_error(self) -> "ToolObservation":
        if self.status == "success" and self.error_code is not None:
            raise ValueError("successful observations cannot have an error_code")
        return self


class PlanPatch(SchemaModel):
    event_type: Literal["plan_patch"] = "plan_patch"
    revision: int = Field(gt=0)
    reason: NonEmptyStr
    steps: tuple[PlanNode, ...]

    @field_validator("steps")
    @classmethod
    def validate_unique_steps(cls, value: tuple[PlanNode, ...]) -> tuple[PlanNode, ...]:
        ids = [step.node_id for step in value]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step identifiers must be unique")
        known = set(ids)
        missing = sorted(
            dependency
            for step in value
            for dependency in step.dependencies
            if dependency not in known
        )
        if missing:
            raise ValueError(f"plan dependencies reference unknown nodes: {missing}")
        return value


class RuntimeCheckResult(SchemaModel):
    check_name: Identifier
    passed: bool
    summary: NonEmptyStr
    conclusive: bool = True


class VerificationResult(SchemaModel):
    event_type: Literal["verification_result"] = "verification_result"
    verification_id: Identifier
    status: VerificationStatus
    checks: tuple[RuntimeCheckResult, ...] = ()
    failure_types: tuple[Identifier, ...] = ()


class BudgetUpdate(SchemaModel):
    event_type: Literal["budget_update"] = "budget_update"
    used_steps_delta: int = Field(default=0, ge=0)
    used_tool_calls_delta: int = Field(default=0, ge=0)
    used_search_calls_delta: int = Field(default=0, ge=0)
    used_edit_calls_delta: int = Field(default=0, ge=0)
    used_structured_output_repairs_delta: int = Field(default=0, ge=0)
    used_model_tokens_delta: int = Field(default=0, ge=0)
    used_wall_time_ms_delta: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_nonzero(self) -> "BudgetUpdate":
        if not any(
            (
                self.used_steps_delta,
                self.used_tool_calls_delta,
                self.used_search_calls_delta,
                self.used_edit_calls_delta,
                self.used_structured_output_repairs_delta,
                self.used_model_tokens_delta,
                self.used_wall_time_ms_delta,
            )
        ):
            raise ValueError("a budget update must consume at least one resource")
        return self


class TerminalEvent(SchemaModel):
    event_type: Literal["terminal"] = "terminal"
    status: TerminalStatus
    reason: NonEmptyStr
    reason_code: TerminalReasonCode | None = None
    output_artifact_id: Identifier | None = None
    evidence_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_success_output(self) -> "TerminalEvent":
        if self.reason_code == "SUCCESS" and self.output_artifact_id is None:
            raise ValueError("SUCCESS terminal event requires an output artifact")
        return self


AgentEvent = Annotated[
    ToolObservation | PlanPatch | VerificationResult | BudgetUpdate | TerminalEvent,
    Field(discriminator="event_type"),
]

EventSource = Literal["tool", "planner", "verifier", "runtime", "budget"]


class AgentEventEnvelope(SchemaModel):
    event_id: Identifier
    task_id: Identifier
    sequence_no: int = Field(ge=1)
    event: AgentEvent
    emitted_by: EventSource
    created_at: datetime
    parent_state_version: int = Field(ge=0)

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_source(self) -> "AgentEventEnvelope":
        allowed_sources = {
            "tool_observation": {"tool"},
            "plan_patch": {"planner"},
            "verification_result": {"verifier"},
            "budget_update": {"budget", "runtime"},
            "terminal": {"runtime"},
        }
        if self.emitted_by not in allowed_sources[self.event.event_type]:
            raise ValueError(f"{self.event.event_type} cannot be emitted by {self.emitted_by}")
        return self
