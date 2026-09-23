"""Public M4A Agent planning, decision, runtime, and trajectory contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter, field_validator, model_validator

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.event import AgentEventEnvelope, PlanNode, PlanPatch, TerminalReasonCode
from cutagent.schemas.state import AgentState
from cutagent.schemas.task_input import TaskInput
from cutagent.schemas.tools import ToolCall, ToolExecutionRecord

AgentBaseline = Literal["react", "hierarchical"]
PolicyViewMode = Literal["structured_state", "multimodal_evidence"]
PolicyOperation = Literal["plan", "decide", "replan"]


class PlanGraph(SchemaModel):
    revision: int = Field(gt=0)
    nodes: tuple[PlanNode, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_graph(self) -> PlanGraph:
        identifiers = [node.node_id for node in self.nodes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("plan graph node identifiers must be unique")
        known = set(identifiers)
        for node in self.nodes:
            missing = set(node.dependencies) - known
            if missing:
                raise ValueError(f"plan node has unknown dependencies: {sorted(missing)}")
        visiting: set[str] = set()
        visited: set[str] = set()
        by_id = {node.node_id: node for node in self.nodes}

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("plan graph must be acyclic")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in by_id[node_id].dependencies:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for identifier in identifiers:
            visit(identifier)
        return self

    def as_patch(self, *, reason: str) -> PlanPatch:
        return PlanPatch(revision=self.revision, reason=reason, steps=self.nodes)


class ToolDecision(SchemaModel):
    decision_type: Literal["tool"] = "tool"
    tool_call: ToolCall
    rationale: NonEmptyStr
    expected_observation: NonEmptyStr
    success_condition: NonEmptyStr


class ReplanDecision(SchemaModel):
    decision_type: Literal["replan"] = "replan"
    reason: NonEmptyStr
    affected_plan_nodes: tuple[Identifier, ...] = Field(min_length=1)
    requested_patch: PlanPatch

    @field_validator("affected_plan_nodes")
    @classmethod
    def unique_nodes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("affected plan nodes must be unique")
        return value


class FinishDecision(SchemaModel):
    decision_type: Literal["finish"] = "finish"
    output_artifact_id: Identifier
    completion_summary: NonEmptyStr
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1)


class CannotCompleteDecision(SchemaModel):
    decision_type: Literal["cannot_complete"] = "cannot_complete"
    reason: NonEmptyStr
    missing_evidence_or_capability: tuple[NonEmptyStr, ...] = Field(min_length=1)
    attempted_action_ids: tuple[Identifier, ...] = ()


PolicyDecision = Annotated[
    ToolDecision | ReplanDecision | FinishDecision | CannotCompleteDecision,
    Field(discriminator="decision_type"),
]
POLICY_DECISION_ADAPTER: TypeAdapter[PolicyDecision] = TypeAdapter(PolicyDecision)


class PolicyModelSpec(SchemaModel):
    model_id: Literal["Qwen/Qwen3-VL-4B-Instruct"] = "Qwen/Qwen3-VL-4B-Instruct"
    revision: Literal["ebb281ec70b05090aa6165b016eac8ec08e71b17"] = (
        "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    )
    license: Literal["Apache-2.0"] = "Apache-2.0"
    dtype: Literal["bfloat16"] = "bfloat16"
    device: Literal["cuda:0"] = "cuda:0"


class AgentRuntimeConfig(SchemaModel):
    baseline: AgentBaseline
    policy_view_mode: PolicyViewMode
    seed: int = Field(default=20_260_823, ge=0, le=2**63 - 1)
    max_steps: int = Field(default=12, gt=0, le=64)
    max_tool_calls: int = Field(default=10, gt=0, le=64)
    max_search_calls: int = Field(default=3, gt=0, le=16)
    max_edit_calls: int = Field(default=8, gt=0, le=32)
    max_repeated_identical_actions: int = Field(default=2, gt=0, le=8)
    max_structured_output_repairs: int = Field(default=2, ge=0, le=8)
    max_wall_time_ms: int = Field(default=300_000, gt=0)
    max_model_tokens: int | None = Field(default=24_000, gt=0)
    maximum_visual_evidence: int = Field(default=4, ge=0, le=12)
    policy_maximum_new_tokens: int = Field(default=512, ge=64, le=2048)
    prompt_template_version: NonEmptyStr = "m4a-agent-policy-v2"
    planner_template_version: NonEmptyStr = "m4a-global-planner-v1"
    replanner_template_version: NonEmptyStr = "m4a-targeted-replanner-v1"
    loop_detector_version: NonEmptyStr = "m4a-loop-detector-v1"
    verifier_version: NonEmptyStr = "m4a-online-verifier-v1"


class PolicyContextSnapshot(SchemaModel):
    context_version: NonEmptyStr
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: dict[str, JsonValue]
    visual_evidence_ids: tuple[Identifier, ...] = ()


class PolicyInferenceStats(SchemaModel):
    operation: PolicyOperation
    latency_ms: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    repair_count: int = Field(ge=0)
    peak_allocated_bytes: int | None = Field(default=None, ge=0)
    peak_reserved_bytes: int | None = Field(default=None, ge=0)


class PolicyStepRecord(SchemaModel):
    step_index: int = Field(ge=0)
    operation: PolicyOperation
    context: PolicyContextSnapshot
    decision: PolicyDecision | None = None
    proposed_plan: PlanGraph | None = None
    raw_output_artifact: ArtifactRef
    stats: PolicyInferenceStats

    @model_validator(mode="after")
    def validate_operation_output(self) -> PolicyStepRecord:
        if self.operation == "plan" and self.proposed_plan is None:
            raise ValueError("plan inference requires a proposed plan")
        if self.operation != "plan" and self.decision is None:
            raise ValueError("decision/replan inference requires a decision")
        return self


class PolicyFailureRecord(SchemaModel):
    step_index: int = Field(ge=0)
    operation: PolicyOperation
    context: PolicyContextSnapshot
    raw_failure_artifact: ArtifactRef
    validation_error: NonEmptyStr
    attempt_count: int = Field(gt=0)
    repair_count: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class RuntimeDiagnostic(SchemaModel):
    diagnostic_type: Literal[
        "identical_tool_call",
        "identical_search_result",
        "unchanged_replan",
        "repeated_verification_failure",
        "no_information_gain",
    ]
    summary: NonEmptyStr
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    occurrence_count: int = Field(gt=0)


class TrajectoryLatency(SchemaModel):
    total_ms: int = Field(ge=0)
    policy_ms: int = Field(ge=0)
    retrieval_ms: int = Field(ge=0)
    editing_tool_ms: int = Field(ge=0)
    verification_ms: int = Field(ge=0)


class AgentTrajectory(SchemaModel):
    trajectory_id: Identifier
    run_id: Identifier
    baseline: AgentBaseline
    policy_view_mode: PolicyViewMode
    task_input: TaskInput
    initial_state: AgentState
    initial_plan: PlanGraph | None = None
    policy_steps: tuple[PolicyStepRecord, ...]
    policy_failures: tuple[PolicyFailureRecord, ...] = ()
    tool_records: tuple[ToolExecutionRecord, ...]
    events: tuple[AgentEventEnvelope, ...]
    diagnostics: tuple[RuntimeDiagnostic, ...] = ()
    final_state: AgentState
    terminal_reason: TerminalReasonCode
    final_output_artifact: ArtifactRef | None = None
    model: PolicyModelSpec
    runtime_config: AgentRuntimeConfig
    started_at: datetime
    ended_at: datetime
    latency: TrajectoryLatency

    @field_validator("started_at", "ended_at")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trajectory timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_trajectory(self) -> AgentTrajectory:
        if self.ended_at < self.started_at:
            raise ValueError("trajectory cannot end before it starts")
        if self.final_state.terminal_event is None:
            raise ValueError("trajectory final state must be terminal")
        if self.final_state.terminal_event.reason_code != self.terminal_reason:
            raise ValueError("trajectory terminal reason differs from terminal event")
        if self.terminal_reason == "SUCCESS" and self.final_output_artifact is None:
            raise ValueError("successful trajectory requires a final artifact")
        return self
