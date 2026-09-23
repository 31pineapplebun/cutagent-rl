"""Versioned public contracts for the M4B constrained-recovery Agent."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter, field_validator, model_validator

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.agent import (
    FinishDecision,
    PlanGraph,
    PolicyDecision,
    PolicyModelSpec,
)
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.event import (
    BudgetUpdate,
    PlanNode,
    PlanPatch,
    TerminalEvent,
    TerminalReasonCode,
    ToolObservation,
    VerificationResult,
)
from cutagent.schemas.state import ExecutionBudget
from cutagent.schemas.task_input import TaskInput
from cutagent.schemas.tools import ToolExecutionRecord

M4BProtocolVariant = Literal["handoff_only", "compact_recovery"]
M4BPolicyOperation = Literal["plan", "decide", "replan", "recover"]
WorkingValidationStatus = Literal[
    "unknown",
    "tool_validated",
    "independently_validated",
    "failed",
    "inconclusive",
]


class ObservableMediaMetadata(SchemaModel):
    duration_ms: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    has_audio: bool | None = None
    fully_decoded: bool | None = None

    @model_validator(mode="after")
    def dimensions_are_paired(self) -> ObservableMediaMetadata:
        if (self.width is None) != (self.height is None):
            raise ValueError("observable media dimensions must be supplied together")
        return self


class WorkingMediaArtifact(SchemaModel):
    artifact_id: Identifier
    media_type: NonEmptyStr
    producing_tool: Identifier | None = None
    metadata: ObservableMediaMetadata | None = None
    validation_status: WorkingValidationStatus = "unknown"


class WorkingArtifactState(SchemaModel):
    original_input_artifact: WorkingMediaArtifact
    current_working_artifact: WorkingMediaArtifact
    latest_generated_artifact: WorkingMediaArtifact | None = None
    final_candidate_artifact: WorkingMediaArtifact | None = None

    @model_validator(mode="after")
    def preserve_original_identity(self) -> WorkingArtifactState:
        if self.original_input_artifact.producing_tool is not None:
            raise ValueError("original input artifact cannot have a producing tool")
        if (
            self.final_candidate_artifact is not None
            and self.final_candidate_artifact.artifact_id
            != self.current_working_artifact.artifact_id
        ):
            raise ValueError("final candidate must be the current working artifact")
        return self


class RemainingBudgetSummary(SchemaModel):
    steps: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    search_calls: int = Field(ge=0)
    edit_calls: int = Field(ge=0)
    structured_output_repairs: int = Field(ge=0)
    model_tokens: int | None = Field(default=None, ge=0)
    wall_time_ms: int = Field(ge=0)


class StepOutcomeSummary(SchemaModel):
    summary_id: Identifier
    active_plan_node: Identifier | None = None
    latest_tool_name: Identifier | None = None
    latest_tool_status: Literal["success", "invalid", "timeout", "error"] | None = None
    produced_artifact_id: Identifier | None = None
    online_verification_status: Literal["passed", "failed", "inconclusive"] | None = None
    satisfied_completion_criteria: tuple[NonEmptyStr, ...] = ()
    unsatisfied_completion_criteria: tuple[NonEmptyStr, ...] = ()
    unverified_items: tuple[NonEmptyStr, ...] = ()
    missing_evidence: tuple[NonEmptyStr, ...] = ()
    next_ready_plan_nodes: tuple[Identifier, ...] = ()
    remaining_budget: RemainingBudgetSummary
    completion_ready: bool

    @model_validator(mode="after")
    def completion_has_no_unsatisfied_runtime_constraints(self) -> StepOutcomeSummary:
        if self.completion_ready and (
            self.unsatisfied_completion_criteria
            or self.next_ready_plan_nodes
            or self.missing_evidence
        ):
            raise ValueError("completion_ready conflicts with unresolved runtime criteria")
        return self


class RetryCurrentNode(SchemaModel):
    recovery_type: Literal["retry_current_node"] = "retry_current_node"
    node_id: Identifier
    reason: NonEmptyStr
    preferred_capability: Identifier | None = None


class ModifyCurrentNode(SchemaModel):
    recovery_type: Literal["modify_current_node"] = "modify_current_node"
    node_id: Identifier
    reason: NonEmptyStr
    revised_subgoal: NonEmptyStr
    revised_completion_criteria: tuple[NonEmptyStr, ...] = Field(min_length=1, max_length=8)
    preferred_capability: Identifier | None = None


class InsertRecoveryNode(SchemaModel):
    recovery_type: Literal["insert_recovery_node"] = "insert_recovery_node"
    affected_node_id: Identifier
    reason: NonEmptyStr
    recovery_subgoal: NonEmptyStr
    completion_criteria: tuple[NonEmptyStr, ...] = Field(min_length=1, max_length=8)
    preferred_capability: Identifier | None = None


class SkipBlockedNode(SchemaModel):
    recovery_type: Literal["skip_blocked_node"] = "skip_blocked_node"
    node_id: Identifier
    reason: NonEmptyStr


class CannotRecover(SchemaModel):
    recovery_type: Literal["cannot_recover"] = "cannot_recover"
    reason: NonEmptyStr
    missing_evidence_or_capability: tuple[NonEmptyStr, ...] = Field(min_length=1)


RecoveryDecision = Annotated[
    RetryCurrentNode | ModifyCurrentNode | InsertRecoveryNode | SkipBlockedNode | CannotRecover,
    Field(discriminator="recovery_type"),
]
RECOVERY_DECISION_ADAPTER: TypeAdapter[RecoveryDecision] = TypeAdapter(RecoveryDecision)


class WorkingArtifactValidationEvent(SchemaModel):
    event_type: Literal["working_artifact_validation"] = "working_artifact_validation"
    artifact_id: Identifier
    source_tool_call_id: Identifier
    status: Literal["passed", "failed", "inconclusive"]
    observable_metadata: ObservableMediaMetadata | None = None


class StepOutcomeEvent(SchemaModel):
    event_type: Literal["step_outcome"] = "step_outcome"
    summary: StepOutcomeSummary


class RecoveryOperationEvent(SchemaModel):
    event_type: Literal["recovery_operation"] = "recovery_operation"
    recovery_id: Identifier
    decision: RecoveryDecision
    accepted: bool
    affected_node_ids: tuple[Identifier, ...]
    generated_plan_revision: int | None = Field(default=None, gt=0)
    rejection_reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def acceptance_fields_match(self) -> RecoveryOperationEvent:
        if self.accepted != (self.generated_plan_revision is not None):
            raise ValueError("accepted recovery requires exactly one generated plan revision")
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("accepted recovery cannot have a rejection reason")
        if not self.accepted and self.rejection_reason is None:
            raise ValueError("rejected recovery requires a reason")
        return self


_PATH_PATTERN = re.compile(r"(?:[A-Za-z]:[\\/]|file://|/(?:home|tmp|var|etc|root)/)")
_SECRET_PATTERN = re.compile(
    r"(?:password|passwd|credential|secret|api[_-]?key|access[_-]?token)", re.IGNORECASE
)


class SanitizedSystemErrorEvent(SchemaModel):
    event_type: Literal["sanitized_system_error"] = "sanitized_system_error"
    error_id: Identifier
    error_category: Identifier
    safe_message: NonEmptyStr
    component: Identifier
    operation: Identifier
    decision_id: Identifier | None = None

    @field_validator("safe_message")
    @classmethod
    def reject_sensitive_diagnostics(cls, value: str) -> str:
        if _PATH_PATTERN.search(value) or _SECRET_PATTERN.search(value):
            raise ValueError("safe diagnostic contains a path or secret-bearing term")
        return value


M4BEvent = Annotated[
    ToolObservation
    | PlanPatch
    | VerificationResult
    | BudgetUpdate
    | TerminalEvent
    | WorkingArtifactValidationEvent
    | StepOutcomeEvent
    | RecoveryOperationEvent
    | SanitizedSystemErrorEvent,
    Field(discriminator="event_type"),
]
M4BEventSource = Literal["tool", "planner", "verifier", "runtime", "budget", "recovery", "handoff"]


class M4BEventEnvelope(SchemaModel):
    event_id: Identifier
    task_id: Identifier
    sequence_no: int = Field(ge=1)
    event: M4BEvent
    emitted_by: M4BEventSource
    created_at: datetime
    parent_state_version: int = Field(ge=0)

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_source(self) -> M4BEventEnvelope:
        allowed = {
            "tool_observation": {"tool"},
            "plan_patch": {"planner", "recovery"},
            "verification_result": {"verifier"},
            "budget_update": {"budget", "runtime"},
            "terminal": {"runtime"},
            "working_artifact_validation": {"handoff"},
            "step_outcome": {"handoff"},
            "recovery_operation": {"recovery"},
            "sanitized_system_error": {"runtime"},
        }
        if self.emitted_by not in allowed[self.event.event_type]:
            raise ValueError(f"{self.event.event_type} cannot be emitted by {self.emitted_by}")
        return self


class M4BAgentState(SchemaModel):
    task_input: TaskInput
    execution_budget: ExecutionBudget
    working_artifacts: WorkingArtifactState
    plan_revision: int = Field(default=0, ge=0)
    plan_steps: tuple[PlanNode, ...] = ()
    tool_observations: tuple[ToolObservation, ...] = ()
    verification_results: tuple[VerificationResult, ...] = ()
    step_outcomes: tuple[StepOutcomeSummary, ...] = ()
    recovery_events: tuple[RecoveryOperationEvent, ...] = ()
    system_diagnostics: tuple[SanitizedSystemErrorEvent, ...] = ()
    terminal_event: TerminalEvent | None = None
    processed_event_ids: tuple[Identifier, ...] = ()
    last_sequence_no: int = Field(default=0, ge=0)
    state_version: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def event_counters_match(self) -> M4BAgentState:
        if self.last_sequence_no != len(self.processed_event_ids):
            raise ValueError("last_sequence_no must match processed M4B events")
        if self.state_version != self.last_sequence_no:
            raise ValueError("M4B state_version must increment once per event")
        return self

    @classmethod
    def initial(cls, task_input: TaskInput, budget: ExecutionBudget) -> M4BAgentState:
        original = WorkingMediaArtifact(
            artifact_id=task_input.video_ref.artifact_id,
            media_type=task_input.video_ref.media_type,
        )
        return cls(
            task_input=task_input,
            execution_budget=budget,
            working_artifacts=WorkingArtifactState(
                original_input_artifact=original,
                current_working_artifact=original,
            ),
        )


class M4BRuntimeConfig(SchemaModel):
    protocol_variant: M4BProtocolVariant
    policy_view_mode: Literal["structured_state"] = "structured_state"
    seed: int = Field(default=20_260_823, ge=0, le=2**63 - 1)
    max_steps: int = Field(default=12, gt=0, le=64)
    max_tool_calls: int = Field(default=10, gt=0, le=64)
    max_search_calls: int = Field(default=3, gt=0, le=16)
    max_edit_calls: int = Field(default=8, gt=0, le=32)
    max_repeated_identical_actions: int = Field(default=2, gt=0, le=8)
    max_structured_output_repairs: int = Field(default=2, ge=0, le=8)
    max_wall_time_ms: int = Field(default=600_000, gt=0)
    max_model_tokens: int | None = Field(default=64_000, gt=0)
    policy_maximum_new_tokens: int = Field(default=768, ge=64, le=2048)
    prompt_template_version: NonEmptyStr = "m4b-handoff-policy-v2"
    planner_template_version: NonEmptyStr = "m4b-global-planner-v5"
    replanner_template_version: NonEmptyStr = "m4b-handoff-full-replan-v1"
    recovery_template_version: NonEmptyStr = "m4b-compact-recovery-v1"
    loop_detector_version: NonEmptyStr = "m4b-loop-detector-v1"
    verifier_version: NonEmptyStr = "m4b-online-verifier-v1"


class M4BPolicyContextSnapshot(SchemaModel):
    context_version: NonEmptyStr
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: dict[str, JsonValue]


class M4BPolicyInferenceStats(SchemaModel):
    operation: M4BPolicyOperation
    latency_ms: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    repair_count: int = Field(ge=0)
    peak_allocated_bytes: int | None = Field(default=None, ge=0)
    peak_reserved_bytes: int | None = Field(default=None, ge=0)


class M4BPolicyStepRecord(SchemaModel):
    step_index: int = Field(ge=0)
    operation: M4BPolicyOperation
    context: M4BPolicyContextSnapshot
    decision: PolicyDecision | RecoveryDecision | None = None
    proposed_plan: PlanGraph | None = None
    raw_output_artifact: ArtifactRef
    stats: M4BPolicyInferenceStats

    @model_validator(mode="after")
    def validate_operation_output(self) -> M4BPolicyStepRecord:
        if self.operation == "plan" and self.proposed_plan is None:
            raise ValueError("plan inference requires a proposed plan")
        if self.operation != "plan" and self.decision is None:
            raise ValueError("non-plan inference requires a decision")
        if self.operation == "recover" and not isinstance(
            self.decision,
            (
                RetryCurrentNode,
                ModifyCurrentNode,
                InsertRecoveryNode,
                SkipBlockedNode,
                CannotRecover,
            ),
        ):
            raise ValueError("recover inference requires RecoveryDecision")
        return self


class M4BPolicyFailureRecord(SchemaModel):
    step_index: int = Field(ge=0)
    operation: M4BPolicyOperation
    context: M4BPolicyContextSnapshot
    raw_failure_artifact: ArtifactRef
    validation_error: NonEmptyStr
    attempt_count: int = Field(gt=0)
    repair_count: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class M4BRuntimeDiagnostic(SchemaModel):
    diagnostic_type: Literal[
        "identical_tool_call",
        "identical_search_result",
        "unchanged_replan",
        "repeated_verification_failure",
        "no_information_gain",
        "repeated_editor",
    ]
    summary: NonEmptyStr
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    occurrence_count: int = Field(gt=0)


class M4BTrajectoryLatency(SchemaModel):
    total_ms: int = Field(ge=0)
    policy_ms: int = Field(ge=0)
    recovery_policy_ms: int = Field(ge=0)
    retrieval_ms: int = Field(ge=0)
    editing_tool_ms: int = Field(ge=0)
    verification_ms: int = Field(ge=0)


class M4BAgentTrajectory(SchemaModel):
    trajectory_version: Literal["m4b-trajectory-v1"] = "m4b-trajectory-v1"
    trajectory_id: Identifier
    run_id: Identifier
    baseline: Literal["hierarchical"] = "hierarchical"
    protocol_variant: M4BProtocolVariant
    policy_view_mode: Literal["structured_state"] = "structured_state"
    task_input: TaskInput
    initial_state: M4BAgentState
    initial_plan: PlanGraph | None = None
    policy_steps: tuple[M4BPolicyStepRecord, ...]
    policy_failures: tuple[M4BPolicyFailureRecord, ...] = ()
    tool_records: tuple[ToolExecutionRecord, ...]
    events: tuple[M4BEventEnvelope, ...]
    diagnostics: tuple[M4BRuntimeDiagnostic, ...] = ()
    final_state: M4BAgentState
    terminal_reason: TerminalReasonCode
    final_output_artifact: ArtifactRef | None = None
    model: PolicyModelSpec
    runtime_config: M4BRuntimeConfig
    started_at: datetime
    ended_at: datetime
    latency: M4BTrajectoryLatency

    @field_validator("started_at", "ended_at")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trajectory timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_trajectory(self) -> M4BAgentTrajectory:
        if self.ended_at < self.started_at:
            raise ValueError("trajectory cannot end before it starts")
        if self.final_state.terminal_event is None:
            raise ValueError("M4B final state must be terminal")
        if self.final_state.terminal_event.reason_code != self.terminal_reason:
            raise ValueError("trajectory terminal reason differs from terminal event")
        if self.terminal_reason == "SUCCESS" and self.final_output_artifact is None:
            raise ValueError("successful M4B trajectory requires a final artifact")
        return self


def finish_decision_artifact(decision: PolicyDecision | RecoveryDecision) -> str | None:
    """Return a finish artifact without widening policy-visible state."""

    return decision.output_artifact_id if isinstance(decision, FinishDecision) else None
