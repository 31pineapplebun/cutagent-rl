"""M4B hierarchical runtime with explicit handoff and constrained recovery."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from cutagent.agent.m4b_loop_detection import M4BLoopDetector
from cutagent.agent.m4b_planning import (
    active_plan_node,
    build_local_recovery_patch,
    progress_plan_node,
)
from cutagent.agent.m4b_policy_context import (
    M4BPolicyContext,
    M4BPolicyContextBuilder,
    M4BPolicyViewSerializer,
)
from cutagent.agent.m4b_protocols import (
    M4BPolicyModel,
    M4BPolicyModelRequest,
    M4BPolicyModelResult,
)
from cutagent.agent.m4b_state_reducer import M4BStateReducer
from cutagent.agent.planning import patch_node_status, update_readiness, validate_targeted_patch
from cutagent.core.errors import StructuredOutputError
from cutagent.perception.artifacts import write_json_artifact
from cutagent.schemas.agent import (
    CannotCompleteDecision,
    FinishDecision,
    PlanGraph,
    PolicyModelSpec,
    ReplanDecision,
    ToolDecision,
)
from cutagent.schemas.event import (
    BudgetUpdate,
    RuntimeCheckResult,
    TerminalEvent,
    TerminalReasonCode,
    TerminalStatus,
    VerificationResult,
)
from cutagent.schemas.m4b_agent import (
    CannotRecover,
    M4BAgentState,
    M4BAgentTrajectory,
    M4BEvent,
    M4BEventEnvelope,
    M4BEventSource,
    M4BPolicyContextSnapshot,
    M4BPolicyFailureRecord,
    M4BPolicyOperation,
    M4BPolicyStepRecord,
    M4BRuntimeConfig,
    M4BRuntimeDiagnostic,
    M4BTrajectoryLatency,
    RecoveryDecision,
    RecoveryOperationEvent,
    SanitizedSystemErrorEvent,
    StepOutcomeEvent,
    StepOutcomeSummary,
)
from cutagent.schemas.state import ExecutionBudget
from cutagent.schemas.task_input import TaskInput
from cutagent.schemas.tools import ToolExecutionContext, ToolExecutionRecord
from cutagent.tools.errors import ToolFailure
from cutagent.tools.registry import ToolRegistry
from cutagent.verification.m4b_online import M4BOnlineVerifier, StepOutcomeBuilder

_EDIT_TOOLS = frozenset(
    {
        "trim_video",
        "concat_videos",
        "change_speed",
        "add_subtitles",
        "reframe_video",
        "normalize_audio",
    }
)


class _M4BPolicyStepFailure(Exception):
    def __init__(self, record: M4BPolicyFailureRecord) -> None:
        super().__init__(record.validation_error)
        self.record = record


class _M4BEventLog:
    def __init__(self, state: M4BAgentState, *, run_id: str) -> None:
        self.state = state
        self.run_id = run_id
        self.events: list[M4BEventEnvelope] = []

    def emit(self, event: M4BEvent, source: M4BEventSource) -> M4BEventEnvelope:
        sequence = self.state.last_sequence_no + 1
        envelope = M4BEventEnvelope(
            event_id=f"{self.run_id}-event-{sequence:04d}",
            task_id=self.state.task_input.task_id,
            sequence_no=sequence,
            event=event,
            emitted_by=source,
            created_at=datetime.now(UTC),
            parent_state_version=self.state.state_version,
        )
        self.state = M4BStateReducer.apply(self.state, envelope)
        self.events.append(envelope)
        return envelope


def _terminal_status(reason: TerminalReasonCode) -> TerminalStatus:
    if reason == "SUCCESS":
        return "succeeded"
    if reason == "CANNOT_COMPLETE":
        return "unfulfillable"
    if reason == "BUDGET_EXHAUSTED":
        return "budget_exhausted"
    if reason in {"LOOP_DETECTED", "MODEL_OUTPUT_FAILURE"}:
        return "failed"
    return "system_error"


def _safe_error(error: Exception, operation: str) -> tuple[str, str]:
    if isinstance(error, ValueError):
        return (
            "runtime_validation_error",
            f"{operation} rejected an inconsistent public runtime transition",
        )
    if isinstance(error, KeyError):
        return (
            "runtime_lookup_error",
            f"{operation} could not resolve an opaque runtime identifier",
        )
    return (
        "runtime_internal_error",
        f"{operation} failed inside the typed Agent runtime boundary",
    )


def _information_version(state: M4BAgentState) -> int:
    artifact_ids = {
        artifact.artifact_id
        for observation in state.tool_observations
        for artifact in observation.artifacts
    }
    verification_ids = {item.verification_id for item in state.verification_results}
    return len(artifact_ids) + len(verification_ids)


class M4BAgentRuntime:
    """Run M4B handoff-only or compact-recovery behavior through ToolRegistry."""

    version = "m4b-agent-runtime-v1"

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        policy_model: M4BPolicyModel,
        artifact_root: Path,
        context_builder: M4BPolicyContextBuilder | None = None,
        verifier: M4BOnlineVerifier | None = None,
        outcome_builder: StepOutcomeBuilder | None = None,
    ) -> None:
        self.registry = registry
        self.policy_model = policy_model
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.context_builder = context_builder or M4BPolicyContextBuilder()
        self.verifier = verifier or M4BOnlineVerifier()
        self.outcome_builder = outcome_builder or StepOutcomeBuilder()

    @staticmethod
    def _graph(state: M4BAgentState) -> PlanGraph | None:
        if state.plan_revision == 0:
            return None
        return PlanGraph(revision=state.plan_revision, nodes=state.plan_steps)

    @staticmethod
    def _snapshot(context: M4BPolicyContext) -> tuple[str, M4BPolicyContextSnapshot]:
        payload = M4BPolicyViewSerializer.to_dict(context)
        serialized = M4BPolicyViewSerializer.to_json(context)
        return serialized, M4BPolicyContextSnapshot(
            context_version=context.context_version,
            context_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            payload=cast(dict[str, JsonValue], payload),
        )

    def _infer(
        self,
        *,
        event_log: _M4BEventLog,
        config: M4BRuntimeConfig,
        operation: M4BPolicyOperation,
        step_index: int,
        step_outcome: StepOutcomeSummary,
    ) -> tuple[M4BPolicyModelResult, M4BPolicyStepRecord]:
        context = self.context_builder.build(event_log.state, step_outcome=step_outcome)
        serialized, snapshot = self._snapshot(context)
        prompt_version = {
            "plan": config.planner_template_version,
            "decide": config.prompt_template_version,
            "replan": config.replanner_template_version,
            "recover": config.recovery_template_version,
        }[operation]
        request = M4BPolicyModelRequest(
            operation=operation,
            protocol_variant=config.protocol_variant,
            serialized_context=serialized,
            tool_manifest=self.registry.manifest(),
            output_directory=(
                self.artifact_root
                / "policy"
                / event_log.run_id
                / f"step-{step_index:03d}-{operation}"
            ),
            maximum_new_tokens=config.policy_maximum_new_tokens,
            maximum_repairs=min(
                config.max_structured_output_repairs,
                event_log.state.execution_budget.remaining_structured_output_repairs,
            ),
            prompt_template_version=prompt_version,
            seed=config.seed,
        )
        started = time.perf_counter_ns()
        try:
            result = self.policy_model.infer(request)
        except StructuredOutputError as error:
            latency_ms = (time.perf_counter_ns() - started) // 1_000_000
            failure_artifact = write_json_artifact(
                request.output_directory / "runtime_policy_failure.json",
                {
                    "operation": operation,
                    "prompt_template_version": prompt_version,
                    "seed": config.seed,
                    "validation_error": str(error),
                    "attempts": list(error.attempts),
                },
                artifact_prefix="m4b-policy-runtime-failure",
            )
            raise _M4BPolicyStepFailure(
                M4BPolicyFailureRecord(
                    step_index=step_index,
                    operation=operation,
                    context=snapshot,
                    raw_failure_artifact=failure_artifact,
                    validation_error=str(error),
                    attempt_count=max(1, len(error.attempts)),
                    repair_count=max(0, len(error.attempts) - 1),
                    latency_ms=latency_ms,
                )
            ) from error
        record = M4BPolicyStepRecord(
            step_index=step_index,
            operation=result.operation,
            context=snapshot,
            decision=result.decision,
            proposed_plan=result.proposed_plan,
            raw_output_artifact=result.raw_output_artifact,
            stats=result.stats,
        )
        return result, record

    @staticmethod
    def _would_exceed(budget: ExecutionBudget, update: BudgetUpdate) -> bool:
        return any(
            (
                budget.used_steps + update.used_steps_delta > budget.max_steps,
                budget.used_tool_calls + update.used_tool_calls_delta > budget.max_tool_calls,
                budget.used_search_calls + update.used_search_calls_delta > budget.max_search_calls,
                budget.used_edit_calls + update.used_edit_calls_delta > budget.max_edit_calls,
                budget.used_structured_output_repairs + update.used_structured_output_repairs_delta
                > budget.max_structured_output_repairs,
                budget.used_wall_time_ms + update.used_wall_time_ms_delta > budget.max_wall_time_ms,
                budget.max_model_tokens is not None
                and budget.used_model_tokens + update.used_model_tokens_delta
                > budget.max_model_tokens,
            )
        )

    @staticmethod
    def _clamp_update(budget: ExecutionBudget, update: BudgetUpdate) -> BudgetUpdate:
        remaining_tokens = (
            update.used_model_tokens_delta
            if budget.remaining_model_tokens is None
            else min(update.used_model_tokens_delta, budget.remaining_model_tokens)
        )
        return BudgetUpdate(
            used_steps_delta=min(update.used_steps_delta, budget.remaining_steps),
            used_tool_calls_delta=min(update.used_tool_calls_delta, budget.remaining_tool_calls),
            used_search_calls_delta=min(
                update.used_search_calls_delta, budget.remaining_search_calls
            ),
            used_edit_calls_delta=min(update.used_edit_calls_delta, budget.remaining_edit_calls),
            used_structured_output_repairs_delta=min(
                update.used_structured_output_repairs_delta,
                budget.remaining_structured_output_repairs,
            ),
            used_model_tokens_delta=remaining_tokens,
            used_wall_time_ms_delta=min(
                update.used_wall_time_ms_delta, budget.remaining_wall_time_ms
            ),
        )

    def _consume(self, event_log: _M4BEventLog, update: BudgetUpdate) -> bool:
        exceeded = self._would_exceed(event_log.state.execution_budget, update)
        event_log.emit(
            self._clamp_update(event_log.state.execution_budget, update) if exceeded else update,
            "budget",
        )
        return not exceeded

    @staticmethod
    def _terminate(
        event_log: _M4BEventLog,
        reason_code: TerminalReasonCode,
        reason: str,
        *,
        output_artifact_id: str | None = None,
        evidence_ids: tuple[str, ...] = (),
    ) -> None:
        event_log.emit(
            TerminalEvent(
                status=_terminal_status(reason_code),
                reason=reason,
                reason_code=reason_code,
                output_artifact_id=output_artifact_id,
                evidence_ids=evidence_ids,
            ),
            "runtime",
        )

    def _outcome(
        self,
        event_log: _M4BEventLog,
        *,
        active_node_id: str | None,
        latest_record: ToolExecutionRecord | None,
        latest_verification: VerificationResult | None,
        emit: bool,
    ) -> StepOutcomeSummary:
        summary = self.outcome_builder.build(
            event_log.state,
            summary_index=len(event_log.state.step_outcomes) + 1,
            active_node_id=active_node_id,
            latest_record=latest_record,
            latest_verification=latest_verification,
        )
        if emit:
            event_log.emit(StepOutcomeEvent(summary=summary), "handoff")
        return summary

    def run(
        self,
        task: TaskInput,
        *,
        config: M4BRuntimeConfig,
        run_id: str,
    ) -> M4BAgentTrajectory:
        started_at = datetime.now(UTC)
        started_ns = time.perf_counter_ns()
        initial_state = M4BAgentState.initial(
            task,
            ExecutionBudget(
                max_steps=config.max_steps,
                max_tool_calls=config.max_tool_calls,
                max_search_calls=config.max_search_calls,
                max_edit_calls=config.max_edit_calls,
                max_repeated_identical_actions=config.max_repeated_identical_actions,
                max_structured_output_repairs=config.max_structured_output_repairs,
                max_model_tokens=config.max_model_tokens,
                max_wall_time_ms=config.max_wall_time_ms,
            ),
        )
        event_log = _M4BEventLog(initial_state, run_id=run_id)
        policy_steps: list[M4BPolicyStepRecord] = []
        policy_failures: list[M4BPolicyFailureRecord] = []
        tool_records: list[ToolExecutionRecord] = []
        diagnostics: list[M4BRuntimeDiagnostic] = []
        initial_plan: PlanGraph | None = None
        final_output = None
        policy_ms = recovery_ms = verification_ms = 0
        loop = M4BLoopDetector(maximum_identical_actions=config.max_repeated_identical_actions)
        tool_context = ToolExecutionContext(
            execution_id=run_id,
            allowed_output_root_id="m4b-agent",
            allowed_artifact_ids=(task.video_ref.artifact_id,),
            allowed_capabilities=tuple(
                item.capability for item in self.registry.manifest().capabilities
            ),
            timeout_ms=min(config.max_wall_time_ms, 180_000),
            maximum_output_bytes=2_000_000_000,
        )
        pending_recovery = False
        latest_outcome = self._outcome(
            event_log,
            active_node_id=None,
            latest_record=None,
            latest_verification=None,
            emit=False,
        )
        current_operation = "plan"
        current_decision_id: str | None = None

        try:
            plan_result: M4BPolicyModelResult | None
            plan_record: M4BPolicyStepRecord | None
            try:
                plan_result, plan_record = self._infer(
                    event_log=event_log,
                    config=config,
                    operation="plan",
                    step_index=0,
                    step_outcome=latest_outcome,
                )
            except _M4BPolicyStepFailure as failure:
                policy_failures.append(failure.record)
                policy_ms += failure.record.latency_ms
                self._terminate(
                    event_log,
                    "MODEL_OUTPUT_FAILURE",
                    "M4B planner output remained invalid after bounded repair",
                )
                plan_result = plan_record = None
            if plan_result is not None and plan_record is not None:
                policy_steps.append(plan_record)
                policy_ms += plan_result.stats.latency_ms
                if not self._consume(
                    event_log,
                    BudgetUpdate(
                        used_steps_delta=1,
                        used_structured_output_repairs_delta=(plan_result.stats.repair_count),
                        used_model_tokens_delta=(
                            plan_result.stats.input_tokens + plan_result.stats.output_tokens
                        ),
                        used_wall_time_ms_delta=plan_result.stats.latency_ms,
                    ),
                ):
                    self._terminate(
                        event_log,
                        "BUDGET_EXHAUSTED",
                        "M4B planner inference exhausted the configured budget",
                    )
                else:
                    if plan_result.proposed_plan is None:
                        raise ValueError("M4B planner returned no PlanGraph")
                    if plan_result.proposed_plan.revision != 1:
                        raise ValueError("initial M4B PlanGraph revision must be 1")
                    initial_plan = update_readiness(plan_result.proposed_plan)
                    patch = initial_plan.as_patch(reason="M4B initial global plan")
                    event_log.emit(patch, "planner")
                    loop.observe_patch(patch)
                    latest_outcome = self._outcome(
                        event_log,
                        active_node_id=None,
                        latest_record=None,
                        latest_verification=None,
                        emit=True,
                    )

            while event_log.state.terminal_event is None:
                budget = event_log.state.execution_budget
                if budget.remaining_steps <= 0 or budget.remaining_wall_time_ms <= 0:
                    self._terminate(
                        event_log,
                        "BUDGET_EXHAUSTED",
                        "M4B step or wall-clock budget was exhausted",
                    )
                    break
                current_operation = (
                    "replan"
                    if pending_recovery and config.protocol_variant == "handoff_only"
                    else "recover"
                    if pending_recovery
                    else "decide"
                )
                try:
                    result, record = self._infer(
                        event_log=event_log,
                        config=config,
                        operation=cast(M4BPolicyOperation, current_operation),
                        step_index=len(policy_steps),
                        step_outcome=latest_outcome,
                    )
                except _M4BPolicyStepFailure as failure:
                    policy_failures.append(failure.record)
                    policy_ms += failure.record.latency_ms
                    if failure.record.operation == "recover":
                        recovery_ms += failure.record.latency_ms
                    self._terminate(
                        event_log,
                        "MODEL_OUTPUT_FAILURE",
                        "M4B policy output remained invalid after bounded repair",
                    )
                    break
                policy_steps.append(record)
                policy_ms += result.stats.latency_ms
                if result.operation == "recover":
                    recovery_ms += result.stats.latency_ms
                if not self._consume(
                    event_log,
                    BudgetUpdate(
                        used_steps_delta=1,
                        used_structured_output_repairs_delta=result.stats.repair_count,
                        used_model_tokens_delta=(
                            result.stats.input_tokens + result.stats.output_tokens
                        ),
                        used_wall_time_ms_delta=result.stats.latency_ms,
                    ),
                ):
                    self._terminate(
                        event_log,
                        "BUDGET_EXHAUSTED",
                        "M4B policy inference exhausted the configured budget",
                    )
                    break
                decision = result.decision
                if decision is None:
                    raise ValueError("M4B policy inference returned no decision")
                current_decision_id = (
                    decision.tool_call.tool_call_id
                    if isinstance(decision, ToolDecision)
                    else f"decision-{len(policy_steps):04d}"
                )

                if isinstance(decision, ToolDecision):
                    if current_operation != "decide":
                        raise ValueError("recovery inference returned a tool decision")
                    generic_loop = loop.observe_tool_call(decision.tool_call)
                    editor_loop = loop.observe_editor_call(
                        decision.tool_call,
                        current_working_artifact_id=(
                            event_log.state.working_artifacts.current_working_artifact.artifact_id
                        ),
                        information_version=_information_version(event_log.state),
                    )
                    diagnostic = editor_loop or generic_loop
                    if diagnostic is not None:
                        diagnostics.append(diagnostic)
                        self._terminate(event_log, "LOOP_DETECTED", diagnostic.summary)
                        break
                    tool_name = decision.tool_call.tool_name
                    budget = event_log.state.execution_budget
                    if (
                        budget.remaining_tool_calls <= 0
                        or (tool_name == "search_video" and budget.remaining_search_calls <= 0)
                        or (tool_name in _EDIT_TOOLS and budget.remaining_edit_calls <= 0)
                    ):
                        self._terminate(
                            event_log,
                            "BUDGET_EXHAUSTED",
                            "M4B tool-category budget prevented the requested call",
                        )
                        break
                    graph = self._graph(event_log.state)
                    current_node = active_plan_node(graph) if graph is not None else None
                    if current_node is not None and current_node.status == "ready":
                        assert graph is not None
                        event_log.emit(
                            patch_node_status(
                                graph,
                                current_node.node_id,
                                "running",
                                reason="M4B local executor started plan node",
                            ),
                            "planner",
                        )
                        graph = self._graph(event_log.state)
                        assert graph is not None
                        current_node = active_plan_node(graph)
                    tool_started = time.perf_counter_ns()
                    tool_record = self.registry.execute(decision.tool_call, tool_context)
                    tool_elapsed = (time.perf_counter_ns() - tool_started) // 1_000_000
                    tool_records.append(tool_record)
                    event_log.emit(tool_record.observation, "tool")
                    observation_diagnostics = loop.observe_observation(tool_record.observation)
                    diagnostics.extend(observation_diagnostics)
                    verification_started = time.perf_counter_ns()
                    verification = self.verifier.verify_tool(
                        event_log.state,
                        tool_record,
                        current_node=current_node,
                    )
                    verification_ms += (time.perf_counter_ns() - verification_started) // 1_000_000
                    event_log.emit(verification, "verifier")
                    validation_event = self.verifier.working_validation_event(
                        event_log.state, tool_record
                    )
                    if validation_event is not None:
                        event_log.emit(validation_event, "handoff")
                    failure_diagnostic = loop.observe_verification(verification)
                    if failure_diagnostic is not None:
                        diagnostics.append(failure_diagnostic)
                    tool_budget_ok = self._consume(
                        event_log,
                        BudgetUpdate(
                            used_tool_calls_delta=1,
                            used_search_calls_delta=1 if tool_name == "search_video" else 0,
                            used_edit_calls_delta=1 if tool_name in _EDIT_TOOLS else 0,
                            used_wall_time_ms_delta=max(tool_elapsed, 1),
                        ),
                    )
                    graph = self._graph(event_log.state)
                    if graph is not None and current_node is not None:
                        event_log.emit(
                            progress_plan_node(
                                graph,
                                node_id=current_node.node_id,
                                verification_status=verification.status,
                            ),
                            "planner",
                        )
                    latest_outcome = self._outcome(
                        event_log,
                        active_node_id=current_node.node_id if current_node else None,
                        latest_record=tool_record,
                        latest_verification=verification,
                        emit=True,
                    )
                    if observation_diagnostics or failure_diagnostic is not None:
                        stop = (
                            observation_diagnostics[0]
                            if observation_diagnostics
                            else failure_diagnostic
                        )
                        assert stop is not None
                        self._terminate(event_log, "LOOP_DETECTED", stop.summary)
                        break
                    if not tool_budget_ok:
                        self._terminate(
                            event_log,
                            "BUDGET_EXHAUSTED",
                            "M4B tool execution exhausted the configured budget",
                        )
                        break
                    pending_recovery = verification.status == "failed"
                    continue

                if isinstance(decision, ReplanDecision):
                    if config.protocol_variant != "handoff_only":
                        raise ValueError("compact protocol emitted legacy ReplanDecision")
                    graph = self._graph(event_log.state)
                    if graph is None:
                        raise ValueError("handoff replan requires an existing graph")
                    try:
                        next_graph = validate_targeted_patch(
                            graph,
                            decision.requested_patch,
                            decision.affected_plan_nodes,
                        )
                    except ValueError:
                        failed_node = active_plan_node(graph)
                        verification = VerificationResult(
                            verification_id=f"verify-replan-{len(policy_steps):03d}",
                            status="failed",
                            failure_types=("invalid_replan",),
                        )
                        event_log.emit(verification, "verifier")
                        latest_outcome = self._outcome(
                            event_log,
                            active_node_id=(
                                failed_node.node_id if failed_node is not None else None
                            ),
                            latest_record=None,
                            latest_verification=verification,
                            emit=True,
                        )
                        pending_recovery = True
                        continue
                    patch = next_graph.as_patch(reason=decision.reason)
                    diagnostic = loop.observe_patch(patch)
                    if diagnostic is not None:
                        diagnostics.append(diagnostic)
                        self._terminate(event_log, "LOOP_DETECTED", diagnostic.summary)
                        break
                    event_log.emit(patch, "planner")
                    latest_outcome = self._outcome(
                        event_log,
                        active_node_id=None,
                        latest_record=None,
                        latest_verification=None,
                        emit=True,
                    )
                    pending_recovery = False
                    continue

                if isinstance(decision, CannotRecover):
                    event_log.emit(
                        RecoveryOperationEvent(
                            recovery_id=f"recovery-{len(event_log.state.recovery_events) + 1:03d}",
                            decision=decision,
                            accepted=False,
                            affected_node_ids=(),
                            rejection_reason="policy declared that local recovery is unavailable",
                        ),
                        "recovery",
                    )
                    self._terminate(
                        event_log,
                        "CANNOT_COMPLETE",
                        decision.reason,
                    )
                    break

                if not isinstance(
                    decision, (ToolDecision, ReplanDecision, FinishDecision, CannotCompleteDecision)
                ):
                    graph = self._graph(event_log.state)
                    if config.protocol_variant != "compact_recovery" or graph is None:
                        raise ValueError("RecoveryDecision is unavailable in this protocol state")
                    active = active_plan_node(graph)
                    recovery_id = f"recovery-{len(event_log.state.recovery_events) + 1:03d}"
                    try:
                        if active is None:
                            raise ValueError("compact recovery has no active failed node")
                        application = build_local_recovery_patch(
                            graph,
                            cast(RecoveryDecision, decision),
                            active_node_id=active.node_id,
                            recovery_index=len(event_log.state.recovery_events) + 1,
                        )
                        if application is None:
                            raise ValueError("CannotRecover must use its explicit terminal branch")
                    except ValueError:
                        event_log.emit(
                            RecoveryOperationEvent(
                                recovery_id=recovery_id,
                                decision=cast(RecoveryDecision, decision),
                                accepted=False,
                                affected_node_ids=(active.node_id,) if active else (),
                                rejection_reason="runtime rejected an unsafe local recovery patch",
                            ),
                            "recovery",
                        )
                        verification = VerificationResult(
                            verification_id=f"verify-{recovery_id}",
                            status="failed",
                            failure_types=("recovery_patch_rejected",),
                        )
                        event_log.emit(verification, "verifier")
                        latest_outcome = self._outcome(
                            event_log,
                            active_node_id=active.node_id if active else None,
                            latest_record=None,
                            latest_verification=verification,
                            emit=True,
                        )
                        pending_recovery = True
                        continue
                    event_log.emit(
                        RecoveryOperationEvent(
                            recovery_id=recovery_id,
                            decision=cast(RecoveryDecision, decision),
                            accepted=True,
                            affected_node_ids=application.affected_node_ids,
                            generated_plan_revision=application.patch.revision,
                        ),
                        "recovery",
                    )
                    event_log.emit(application.patch, "recovery")
                    latest_outcome = self._outcome(
                        event_log,
                        active_node_id=None,
                        latest_record=None,
                        latest_verification=None,
                        emit=True,
                    )
                    pending_recovery = False
                    continue

                if isinstance(decision, FinishDecision):
                    if not latest_outcome.completion_ready:
                        verification = VerificationResult(
                            verification_id=f"verify-finish-not-ready-{len(policy_steps):03d}",
                            status="failed",
                            checks=(
                                RuntimeCheckResult(
                                    check_name="completion_ready",
                                    passed=False,
                                    summary="public M4B completion-ready conditions are unresolved",
                                ),
                            ),
                            failure_types=("premature_finish",),
                        )
                        event_log.emit(verification, "verifier")
                        latest_outcome = self._outcome(
                            event_log,
                            active_node_id=None,
                            latest_record=None,
                            latest_verification=verification,
                            emit=True,
                        )
                        continue
                    candidate = event_log.state.working_artifacts.final_candidate_artifact
                    if candidate is None or decision.output_artifact_id != candidate.artifact_id:
                        raise ValueError("FinishDecision does not reference the final candidate")
                    verification_started = time.perf_counter_ns()
                    verification = self.verifier.verify_completion(
                        event_log.state,
                        decision,
                        tuple(tool_records),
                    )
                    verification_ms += (time.perf_counter_ns() - verification_started) // 1_000_000
                    event_log.emit(verification, "verifier")
                    if verification.status == "failed":
                        latest_outcome = self._outcome(
                            event_log,
                            active_node_id=None,
                            latest_record=None,
                            latest_verification=verification,
                            emit=True,
                        )
                        continue
                    try:
                        final_output, _ = self.registry.artifact_store.get(
                            decision.output_artifact_id
                        )
                    except ToolFailure:
                        raise ValueError("completion accepted an unknown opaque artifact") from None
                    self._terminate(
                        event_log,
                        "SUCCESS",
                        decision.completion_summary,
                        output_artifact_id=decision.output_artifact_id,
                        evidence_ids=decision.evidence_ids,
                    )
                    break

                if isinstance(decision, CannotCompleteDecision):
                    self._terminate(
                        event_log,
                        "CANNOT_COMPLETE",
                        decision.reason,
                        evidence_ids=decision.attempted_action_ids,
                    )
                    break
                raise ValueError("unsupported M4B policy decision")
        except Exception as error:
            if event_log.state.terminal_event is None:
                category, message = _safe_error(error, current_operation)
                event_log.emit(
                    SanitizedSystemErrorEvent(
                        error_id=f"system-error-{len(event_log.state.system_diagnostics) + 1:03d}",
                        error_category=category,
                        safe_message=message,
                        component="m4b_agent_runtime",
                        operation=current_operation.replace("_", "-"),
                        decision_id=current_decision_id,
                    ),
                    "runtime",
                )
                self._terminate(
                    event_log,
                    "SYSTEM_ERROR",
                    f"M4B Agent stopped safely after {category}",
                )

        ended_at = datetime.now(UTC)
        total_ms = (time.perf_counter_ns() - started_ns) // 1_000_000
        terminal = event_log.state.terminal_event
        if terminal is None or terminal.reason_code is None:
            raise RuntimeError("M4BAgentRuntime did not produce a typed terminal reason")
        return M4BAgentTrajectory(
            trajectory_id=f"trajectory-{run_id}",
            run_id=run_id,
            protocol_variant=config.protocol_variant,
            task_input=task,
            initial_state=initial_state,
            initial_plan=initial_plan,
            policy_steps=tuple(policy_steps),
            policy_failures=tuple(policy_failures),
            tool_records=tuple(tool_records),
            events=tuple(event_log.events),
            diagnostics=tuple(diagnostics),
            final_state=event_log.state,
            terminal_reason=terminal.reason_code,
            final_output_artifact=final_output,
            model=PolicyModelSpec(),
            runtime_config=config,
            started_at=started_at,
            ended_at=ended_at,
            latency=M4BTrajectoryLatency(
                total_ms=total_ms,
                policy_ms=policy_ms,
                recovery_policy_ms=recovery_ms,
                retrieval_ms=sum(
                    item.trace.latency_ms
                    for item in tool_records
                    if item.trace.tool_name == "search_video"
                ),
                editing_tool_ms=sum(
                    item.trace.latency_ms
                    for item in tool_records
                    if item.trace.tool_name in _EDIT_TOOLS
                ),
                verification_ms=verification_ms,
            ),
        )
