"""Real prompt-only M4A Agent loop over the frozen M3A ToolRegistry."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from cutagent.agent.loop_detection import LoopDetector
from cutagent.agent.planning import (
    current_executable_node,
    patch_node_status,
    update_readiness,
    validate_targeted_patch,
)
from cutagent.agent.policy_context import (
    PolicyContext,
    PolicyContextBuilder,
    PolicyViewSerializer,
)
from cutagent.agent.protocols import (
    AgentPolicyModel,
    PolicyModelRequest,
    PolicyModelResult,
    PolicyVisualInput,
)
from cutagent.agent.state_reducer import StateReducer
from cutagent.core.errors import StructuredOutputError
from cutagent.perception.artifacts import write_json_artifact
from cutagent.schemas.agent import (
    AgentRuntimeConfig,
    AgentTrajectory,
    CannotCompleteDecision,
    FinishDecision,
    PlanGraph,
    PolicyContextSnapshot,
    PolicyFailureRecord,
    PolicyModelSpec,
    PolicyOperation,
    PolicyStepRecord,
    ReplanDecision,
    RuntimeDiagnostic,
    ToolDecision,
    TrajectoryLatency,
)
from cutagent.schemas.event import (
    AgentEvent,
    AgentEventEnvelope,
    BudgetUpdate,
    EventSource,
    TerminalEvent,
    TerminalReasonCode,
    TerminalStatus,
    VerificationResult,
)
from cutagent.schemas.state import AgentState, ExecutionBudget
from cutagent.schemas.task_input import TaskInput
from cutagent.schemas.tools import ToolExecutionContext, ToolExecutionRecord
from cutagent.tools.errors import ToolFailure
from cutagent.tools.registry import ToolRegistry
from cutagent.verification.online import OnlineVerifier

_EDIT_TOOLS = {
    "trim_video",
    "concat_videos",
    "change_speed",
    "add_subtitles",
    "reframe_video",
    "normalize_audio",
}


class _PolicyStepFailure(Exception):
    def __init__(self, record: PolicyFailureRecord) -> None:
        super().__init__(record.validation_error)
        self.record = record


class _EventLog:
    def __init__(self, state: AgentState, *, run_id: str) -> None:
        self.state = state
        self.run_id = run_id
        self.events: list[AgentEventEnvelope] = []

    def emit(self, event: AgentEvent, source: EventSource) -> AgentEventEnvelope:
        sequence = self.state.last_sequence_no + 1
        envelope = AgentEventEnvelope(
            event_id=f"{self.run_id}-event-{sequence:04d}",
            task_id=self.state.task_input.task_id,
            sequence_no=sequence,
            event=event,
            emitted_by=source,
            created_at=datetime.now(UTC),
            parent_state_version=self.state.state_version,
        )
        self.state = StateReducer.apply(self.state, envelope)
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


class AgentRuntime:
    """Execute ReAct or hierarchical policy decisions through ToolRegistry only."""

    version = "m4a-agent-runtime-v1"

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        policy_model: AgentPolicyModel,
        artifact_root: Path,
        visual_artifacts: dict[str, Path] | None = None,
        context_builder: PolicyContextBuilder | None = None,
        verifier: OnlineVerifier | None = None,
    ) -> None:
        self.registry = registry
        self.policy_model = policy_model
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.visual_artifacts = dict(visual_artifacts or {})
        self.context_builder = context_builder or PolicyContextBuilder()
        self.verifier = verifier or OnlineVerifier()

    @staticmethod
    def _graph(state: AgentState) -> PlanGraph | None:
        if state.plan_revision == 0:
            return None
        return PlanGraph(revision=state.plan_revision, nodes=state.plan_steps)

    @staticmethod
    def _snapshot(context: PolicyContext) -> tuple[str, PolicyContextSnapshot]:
        payload = PolicyViewSerializer.to_dict(context)
        serialized = PolicyViewSerializer.to_json(context)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return serialized, PolicyContextSnapshot(
            context_version=context.context_version,
            context_sha256=digest,
            payload=cast(dict[str, JsonValue], payload),
            visual_evidence_ids=tuple(item.artifact_id for item in context.visual_evidence),
        )

    def _visual_inputs(self, context: PolicyContext) -> tuple[PolicyVisualInput, ...]:
        inputs: list[PolicyVisualInput] = []
        for evidence in context.visual_evidence:
            path = self.visual_artifacts.get(evidence.artifact_id)
            if path is None:
                raise ValueError(
                    f"policy visual evidence is not resolvable: {evidence.artifact_id}"
                )
            inputs.append(PolicyVisualInput(artifact_id=evidence.artifact_id, path=path))
        return tuple(inputs)

    def _infer(
        self,
        *,
        event_log: _EventLog,
        config: AgentRuntimeConfig,
        operation: PolicyOperation,
        step_index: int,
    ) -> tuple[PolicyModelResult, PolicyStepRecord]:
        context = self.context_builder.build(
            event_log.state,
            include_visual_evidence=config.policy_view_mode == "multimodal_evidence",
            maximum_visual_evidence=config.maximum_visual_evidence,
        )
        serialized, snapshot = self._snapshot(context)
        prompt_version = {
            "plan": config.planner_template_version,
            "decide": config.prompt_template_version,
            "replan": config.replanner_template_version,
        }[operation]
        request = PolicyModelRequest(
            operation=operation,
            baseline=config.baseline,
            policy_view_mode=config.policy_view_mode,
            serialized_context=serialized,
            tool_manifest=self.registry.manifest(),
            visual_inputs=self._visual_inputs(context),
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
        inference_started = time.perf_counter_ns()
        try:
            result = self.policy_model.infer(request)
        except StructuredOutputError as error:
            latency_ms = (time.perf_counter_ns() - inference_started) // 1_000_000
            failure_artifact = write_json_artifact(
                request.output_directory / "runtime_policy_failure.json",
                {
                    "operation": operation,
                    "prompt_template_version": prompt_version,
                    "seed": config.seed,
                    "validation_error": str(error),
                    "attempts": list(error.attempts),
                },
                artifact_prefix="policy-runtime-failure",
            )
            raise _PolicyStepFailure(
                PolicyFailureRecord(
                    step_index=step_index,
                    operation=operation,
                    context=snapshot,
                    raw_failure_artifact=failure_artifact,
                    validation_error=str(error),
                    attempt_count=max(len(error.attempts), 1),
                    repair_count=max(len(error.attempts) - 1, 0),
                    latency_ms=latency_ms,
                )
            ) from error
        record = PolicyStepRecord(
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
        checks = (
            budget.used_steps + update.used_steps_delta > budget.max_steps,
            budget.used_tool_calls + update.used_tool_calls_delta > budget.max_tool_calls,
            budget.used_search_calls + update.used_search_calls_delta > budget.max_search_calls,
            budget.used_edit_calls + update.used_edit_calls_delta > budget.max_edit_calls,
            budget.used_structured_output_repairs + update.used_structured_output_repairs_delta
            > budget.max_structured_output_repairs,
            budget.used_wall_time_ms + update.used_wall_time_ms_delta > budget.max_wall_time_ms,
            budget.max_model_tokens is not None
            and budget.used_model_tokens + update.used_model_tokens_delta > budget.max_model_tokens,
        )
        return any(checks)

    @staticmethod
    def _clamp_update(budget: ExecutionBudget, update: BudgetUpdate) -> BudgetUpdate:
        remaining_tokens = (
            update.used_model_tokens_delta
            if budget.remaining_model_tokens is None
            else min(update.used_model_tokens_delta, budget.remaining_model_tokens)
        )
        clamped = BudgetUpdate(
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
        return clamped

    def _consume(
        self,
        event_log: _EventLog,
        update: BudgetUpdate,
    ) -> bool:
        exceeded = self._would_exceed(event_log.state.execution_budget, update)
        event_log.emit(
            self._clamp_update(event_log.state.execution_budget, update) if exceeded else update,
            "budget",
        )
        return not exceeded

    @staticmethod
    def _terminate(
        event_log: _EventLog,
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

    def run(
        self,
        task: TaskInput,
        *,
        config: AgentRuntimeConfig,
        run_id: str,
    ) -> AgentTrajectory:
        started_at = datetime.now(UTC)
        started_ns = time.perf_counter_ns()
        initial_state = AgentState.initial(
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
        event_log = _EventLog(initial_state, run_id=run_id)
        policy_steps: list[PolicyStepRecord] = []
        policy_failures: list[PolicyFailureRecord] = []
        tool_records: list[ToolExecutionRecord] = []
        diagnostics: list[RuntimeDiagnostic] = []
        initial_plan: PlanGraph | None = None
        final_output = None
        policy_ms = verification_ms = 0
        loop = LoopDetector(maximum_identical_actions=config.max_repeated_identical_actions)
        tool_context = ToolExecutionContext(
            execution_id=run_id,
            allowed_output_root_id="m4a-agent",
            allowed_artifact_ids=(task.video_ref.artifact_id,),
            allowed_capabilities=tuple(
                item.capability for item in self.registry.manifest().capabilities
            ),
            timeout_ms=min(config.max_wall_time_ms, 180_000),
            maximum_output_bytes=2_000_000_000,
        )
        pending_replan = False

        try:
            if config.baseline == "hierarchical":
                result: PolicyModelResult | None = None
                record: PolicyStepRecord | None = None
                try:
                    result, record = self._infer(
                        event_log=event_log,
                        config=config,
                        operation="plan",
                        step_index=len(policy_steps),
                    )
                except _PolicyStepFailure as failure:
                    policy_failures.append(failure.record)
                    policy_ms += failure.record.latency_ms
                    self._terminate(
                        event_log,
                        "MODEL_OUTPUT_FAILURE",
                        "planner output remained invalid after bounded repair",
                    )
                if result is None or record is None:
                    raise StopIteration
                policy_steps.append(record)
                policy_ms += result.stats.latency_ms
                consumed = self._consume(
                    event_log,
                    BudgetUpdate(
                        used_steps_delta=1,
                        used_structured_output_repairs_delta=result.stats.repair_count,
                        used_model_tokens_delta=(
                            result.stats.input_tokens + result.stats.output_tokens
                        ),
                        used_wall_time_ms_delta=result.stats.latency_ms,
                    ),
                )
                if not consumed:
                    self._terminate(
                        event_log,
                        "BUDGET_EXHAUSTED",
                        "planner inference exhausted the configured Agent budget",
                    )
                else:
                    if result.proposed_plan is None:
                        raise ValueError("planner returned no PlanGraph")
                    if result.proposed_plan.revision != 1:
                        raise ValueError("initial PlanGraph revision must be 1")
                    initial_plan = update_readiness(result.proposed_plan)
                    patch = initial_plan.as_patch(reason="initial global plan")
                    event_log.emit(patch, "planner")
                    loop.observe_patch(patch)

            while event_log.state.terminal_event is None:
                budget = event_log.state.execution_budget
                if budget.remaining_steps <= 0 or budget.remaining_wall_time_ms <= 0:
                    self._terminate(
                        event_log,
                        "BUDGET_EXHAUSTED",
                        "Agent step or wall-clock budget was exhausted",
                    )
                    break
                operation: PolicyOperation = (
                    "replan" if config.baseline == "hierarchical" and pending_replan else "decide"
                )
                try:
                    result, record = self._infer(
                        event_log=event_log,
                        config=config,
                        operation=operation,
                        step_index=len(policy_steps),
                    )
                except _PolicyStepFailure as failure:
                    policy_failures.append(failure.record)
                    policy_ms += failure.record.latency_ms
                    self._terminate(
                        event_log,
                        "MODEL_OUTPUT_FAILURE",
                        "policy output remained invalid after bounded repair",
                    )
                    break
                policy_steps.append(record)
                policy_ms += result.stats.latency_ms
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
                        "policy inference exhausted the configured Agent budget",
                    )
                    break
                decision = result.decision
                if decision is None:
                    raise ValueError("policy inference returned no decision")

                if isinstance(decision, ToolDecision):
                    if operation == "replan":
                        raise ValueError("targeted replan inference returned a tool decision")
                    diagnostic = loop.observe_tool_call(decision.tool_call)
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
                            "tool-category budget prevented the requested call",
                        )
                        break
                    current_node = None
                    graph = self._graph(event_log.state)
                    if graph is not None:
                        current_node = current_executable_node(graph)
                        if current_node is not None and current_node.status == "ready":
                            patch = patch_node_status(
                                graph,
                                current_node.node_id,
                                "running",
                                reason="local executor started plan node",
                            )
                            event_log.emit(patch, "planner")
                            graph = self._graph(event_log.state)
                            assert graph is not None
                            current_node = current_executable_node(graph)
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
                        status = (
                            "succeeded"
                            if verification.status == "passed"
                            else "failed"
                            if verification.status == "failed"
                            else "running"
                        )
                        if status != current_node.status:
                            patch = patch_node_status(
                                graph,
                                current_node.node_id,
                                status,
                                reason=f"online verification was {verification.status}",
                            )
                            event_log.emit(patch, "planner")
                    if observation_diagnostics or failure_diagnostic is not None:
                        diagnostic = (
                            observation_diagnostics[0]
                            if observation_diagnostics
                            else failure_diagnostic
                        )
                        assert diagnostic is not None
                        self._terminate(event_log, "LOOP_DETECTED", diagnostic.summary)
                        break
                    if not tool_budget_ok:
                        self._terminate(
                            event_log,
                            "BUDGET_EXHAUSTED",
                            "tool execution exhausted the configured Agent budget",
                        )
                        break
                    pending_replan = (
                        config.baseline == "hierarchical" and verification.status == "failed"
                    )
                    continue

                if isinstance(decision, ReplanDecision):
                    if config.baseline != "hierarchical":
                        self._terminate(
                            event_log,
                            "MODEL_OUTPUT_FAILURE",
                            "ReAct baseline emitted a forbidden ReplanDecision",
                        )
                        break
                    graph = self._graph(event_log.state)
                    if graph is None:
                        raise ValueError("hierarchical replan requires an existing PlanGraph")
                    try:
                        next_graph = validate_targeted_patch(
                            graph,
                            decision.requested_patch,
                            decision.affected_plan_nodes,
                        )
                    except ValueError:
                        verification = VerificationResult(
                            verification_id=f"verify-replan-{len(policy_steps):03d}",
                            status="failed",
                            checks=(),
                            failure_types=("invalid_replan",),
                        )
                        event_log.emit(verification, "verifier")
                        diagnostic = loop.observe_verification(verification)
                        if diagnostic is not None:
                            diagnostics.append(diagnostic)
                            self._terminate(event_log, "LOOP_DETECTED", diagnostic.summary)
                            break
                        pending_replan = True
                        continue
                    patch = next_graph.as_patch(reason=decision.reason)
                    diagnostic = loop.observe_patch(patch)
                    if diagnostic is not None:
                        diagnostics.append(diagnostic)
                        self._terminate(event_log, "LOOP_DETECTED", diagnostic.summary)
                        break
                    event_log.emit(patch, "planner")
                    pending_replan = False
                    continue

                if isinstance(decision, FinishDecision):
                    verification_started = time.perf_counter_ns()
                    verification = self.verifier.verify_completion(
                        event_log.state,
                        decision,
                        tuple(tool_records),
                    )
                    verification_ms += (time.perf_counter_ns() - verification_started) // 1_000_000
                    event_log.emit(verification, "verifier")
                    if verification.status == "failed":
                        pending_replan = config.baseline == "hierarchical"
                        diagnostic = loop.observe_verification(verification)
                        if diagnostic is not None:
                            diagnostics.append(diagnostic)
                            self._terminate(event_log, "LOOP_DETECTED", diagnostic.summary)
                        continue
                    try:
                        final_output, _ = self.registry.artifact_store.get(
                            decision.output_artifact_id
                        )
                    except ToolFailure:
                        raise ValueError(
                            "completion verifier accepted an unknown artifact"
                        ) from None
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
                raise ValueError("unsupported policy decision")
        except StopIteration:
            pass
        except Exception as error:
            if event_log.state.terminal_event is None:
                self._terminate(
                    event_log,
                    "SYSTEM_ERROR",
                    f"Agent runtime failed safely: {type(error).__name__}",
                )

        ended_at = datetime.now(UTC)
        total_ms = (time.perf_counter_ns() - started_ns) // 1_000_000
        retrieval_ms = sum(
            item.trace.latency_ms for item in tool_records if item.trace.tool_name == "search_video"
        )
        editing_ms = sum(
            item.trace.latency_ms for item in tool_records if item.trace.tool_name in _EDIT_TOOLS
        )
        terminal = event_log.state.terminal_event
        if terminal is None or terminal.reason_code is None:
            raise RuntimeError("AgentRuntime did not produce a typed terminal reason")
        return AgentTrajectory(
            trajectory_id=f"trajectory-{run_id}",
            run_id=run_id,
            baseline=config.baseline,
            policy_view_mode=config.policy_view_mode,
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
            latency=TrajectoryLatency(
                total_ms=total_ms,
                policy_ms=policy_ms,
                retrieval_ms=retrieval_ms,
                editing_tool_ms=editing_ms,
                verification_ms=verification_ms,
            ),
        )
