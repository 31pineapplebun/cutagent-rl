"""Observable-only M4B verification, handoff, and completion-readiness logic."""

from __future__ import annotations

from typing import Literal

from cutagent.agent.m4b_planning import active_plan_node
from cutagent.schemas.agent import FinishDecision, PlanGraph
from cutagent.schemas.event import PlanNode, VerificationResult
from cutagent.schemas.m4b_agent import (
    M4BAgentState,
    ObservableMediaMetadata,
    RemainingBudgetSummary,
    StepOutcomeSummary,
    WorkingArtifactValidationEvent,
)
from cutagent.schemas.state import AgentState
from cutagent.schemas.task_input import (
    AspectRatioConstraint,
    DurationConstraint,
    ForbiddenContentConstraint,
    RequiredContentConstraint,
)
from cutagent.schemas.tools import ToolExecutionRecord
from cutagent.verification.online import OnlineVerifier


def _surrogate(state: M4BAgentState) -> AgentState:
    return AgentState(
        task_input=state.task_input,
        execution_budget=state.execution_budget,
        plan_revision=state.plan_revision,
        plan_steps=state.plan_steps,
        tool_observations=state.tool_observations,
        verification_results=state.verification_results,
        terminal_event=state.terminal_event,
        processed_event_ids=state.processed_event_ids,
        last_sequence_no=state.last_sequence_no,
        state_version=state.state_version,
    )


def _metadata_from_record(record: ToolExecutionRecord) -> ObservableMediaMetadata | None:
    details = record.observation.details

    def integer(name: str, *, positive: bool = False) -> int | None:
        value = details.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        if value < (1 if positive else 0):
            return None
        return value

    duration = integer("duration_ms")
    width = integer("width", positive=True)
    height = integer("height", positive=True)
    if width is None or height is None:
        width = height = None
    has_audio = details.get("has_audio")
    fully_decoded = details.get("fully_decoded")
    return ObservableMediaMetadata(
        duration_ms=duration,
        width=width,
        height=height,
        has_audio=has_audio if isinstance(has_audio, bool) else None,
        fully_decoded=fully_decoded if isinstance(fully_decoded, bool) else None,
    )


class M4BOnlineVerifier:
    """Reuse the frozen observable checks while accepting M4B state."""

    version = "m4b-online-verifier-v1"

    def __init__(self, base: OnlineVerifier | None = None) -> None:
        self._base = base or OnlineVerifier()

    def verify_tool(
        self,
        state: M4BAgentState,
        record: ToolExecutionRecord,
        *,
        current_node: PlanNode | None,
    ) -> VerificationResult:
        return self._base.verify_tool(
            _surrogate(state),
            record,
            current_node=current_node,
        )

    def verify_completion(
        self,
        state: M4BAgentState,
        decision: FinishDecision,
        records: tuple[ToolExecutionRecord, ...],
    ) -> VerificationResult:
        return self._base.verify_completion(_surrogate(state), decision, records)

    @staticmethod
    def working_validation_event(
        state: M4BAgentState,
        record: ToolExecutionRecord,
    ) -> WorkingArtifactValidationEvent | None:
        if record.trace.tool_name != "validate_media":
            return None
        parents = record.trace.parent_artifacts
        if len(parents) != 1:
            return None
        artifact_id = parents[0].artifact_id
        if artifact_id != state.working_artifacts.current_working_artifact.artifact_id:
            return None
        status: Literal["passed", "failed", "inconclusive"] = (
            "passed"
            if record.observation.status == "success"
            else "inconclusive"
            if record.observation.status == "timeout"
            else "failed"
        )
        return WorkingArtifactValidationEvent(
            artifact_id=artifact_id,
            source_tool_call_id=record.observation.call_id,
            status=status,
            observable_metadata=(
                _metadata_from_record(record) if record.observation.status == "success" else None
            ),
        )


class StepOutcomeBuilder:
    """Build one compact outcome using only public state and online checks."""

    version = "m4b-step-outcome-v1"

    @staticmethod
    def _budget(state: M4BAgentState) -> RemainingBudgetSummary:
        budget = state.execution_budget
        return RemainingBudgetSummary(
            steps=budget.remaining_steps,
            tool_calls=budget.remaining_tool_calls,
            search_calls=budget.remaining_search_calls,
            edit_calls=budget.remaining_edit_calls,
            structured_output_repairs=budget.remaining_structured_output_repairs,
            model_tokens=budget.remaining_model_tokens,
            wall_time_ms=budget.remaining_wall_time_ms,
        )

    @staticmethod
    def _runtime_constraints(
        state: M4BAgentState,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        satisfied: list[str] = []
        unsatisfied: list[str] = []
        unverified: list[str] = []
        missing: list[str] = []
        metadata = state.working_artifacts.current_working_artifact.metadata
        for index, constraint in enumerate(state.task_input.user_constraints):
            if isinstance(constraint, DurationConstraint):
                observed = metadata.duration_ms if metadata is not None else None
                label = f"duration constraint {index}: {constraint.min_ms}-{constraint.max_ms} ms"
                if observed is None:
                    missing.append(f"{label}; observable duration is missing")
                elif constraint.min_ms <= observed <= constraint.max_ms:
                    satisfied.append(f"{label}; observed {observed} ms")
                else:
                    unsatisfied.append(f"{label}; observed {observed} ms")
            elif isinstance(constraint, AspectRatioConstraint):
                width = metadata.width if metadata is not None else None
                height = metadata.height if metadata is not None else None
                label = f"aspect ratio {constraint.width}:{constraint.height}"
                if width is None or height is None:
                    missing.append(f"{label}; observable dimensions are missing")
                elif width * constraint.height == height * constraint.width:
                    satisfied.append(f"{label}; observed {width}x{height}")
                else:
                    unsatisfied.append(f"{label}; observed {width}x{height}")
            elif isinstance(constraint, (RequiredContentConstraint, ForbiddenContentConstraint)):
                unverified.append(
                    f"semantic constraint {index} remains unverified by media metadata"
                )
        return satisfied, unsatisfied, unverified, missing

    def build(
        self,
        state: M4BAgentState,
        *,
        summary_index: int,
        active_node_id: str | None,
        latest_record: ToolExecutionRecord | None,
        latest_verification: VerificationResult | None,
    ) -> StepOutcomeSummary:
        satisfied, unsatisfied, unverified, missing = self._runtime_constraints(state)
        if latest_verification is not None:
            for check in latest_verification.checks:
                text = f"{check.check_name}: {check.summary}"
                if not check.conclusive:
                    unverified.append(text)
                elif check.passed:
                    satisfied.append(text)
                else:
                    unsatisfied.append(text)
        graph = (
            PlanGraph(revision=state.plan_revision, nodes=state.plan_steps)
            if state.plan_revision > 0
            else None
        )
        ready_ids: tuple[str, ...] = ()
        unresolved: list[str] = []
        if graph is not None:
            ready_graph = __import__(
                "cutagent.agent.planning", fromlist=["update_readiness"]
            ).update_readiness(graph)
            ready_ids = tuple(node.node_id for node in ready_graph.nodes if node.status == "ready")
            unresolved = [
                f"plan node {node.node_id} is {node.status}"
                for node in ready_graph.nodes
                if node.status not in {"succeeded", "skipped"}
            ]
            unsatisfied.extend(unresolved)
        working = state.working_artifacts
        candidate = working.final_candidate_artifact
        has_valid_output = (
            candidate is not None
            and candidate.artifact_id != working.original_input_artifact.artifact_id
            and candidate.validation_status == "independently_validated"
        )
        if has_valid_output:
            satisfied.append("a generated final candidate passed independent validation")
        else:
            unsatisfied.append("no generated independently validated final candidate exists")
        completion_ready = bool(
            has_valid_output and not unresolved and not unsatisfied and not missing
        )
        produced = None
        if latest_record is not None and latest_record.trace.output_artifact is not None:
            produced = latest_record.trace.output_artifact.artifact_id
        if active_node_id is None and graph is not None:
            active = active_plan_node(graph)
            active_node_id = active.node_id if active is not None else None
        return StepOutcomeSummary(
            summary_id=f"step-outcome-{summary_index:04d}",
            active_plan_node=active_node_id,
            latest_tool_name=(latest_record.trace.tool_name if latest_record else None),
            latest_tool_status=(latest_record.observation.status if latest_record else None),
            produced_artifact_id=produced,
            online_verification_status=(
                latest_verification.status if latest_verification is not None else None
            ),
            satisfied_completion_criteria=tuple(dict.fromkeys(satisfied)),
            unsatisfied_completion_criteria=tuple(dict.fromkeys(unsatisfied)),
            unverified_items=tuple(dict.fromkeys(unverified)),
            missing_evidence=tuple(dict.fromkeys(missing)),
            next_ready_plan_nodes=ready_ids,
            remaining_budget=self._budget(state),
            completion_ready=completion_ready,
        )
