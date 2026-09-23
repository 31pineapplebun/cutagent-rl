"""Evaluator-private M4B.5 failure injection, task construction, and scoring."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import Field, JsonValue, field_validator, model_validator

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.agent import ReplanDecision
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.event import PlanPatch, ToolObservation, VerificationResult
from cutagent.schemas.m4b_agent import (
    CannotRecover,
    M4BAgentTrajectory,
    RecoveryOperationEvent,
    StepOutcomeEvent,
)
from cutagent.schemas.retrieval import AppliedRetrievalFilters, RetrievalResponse
from cutagent.schemas.task_input import TaskInput
from cutagent.schemas.tools import (
    TOOL_CALL_ADAPTER,
    SearchVideoCall,
    ToolCall,
    ToolErrorCategory,
    ToolExecutionContext,
    ToolExecutionRecord,
    ToolManifest,
    ToolTrace,
    ToolValidationCheck,
)
from cutagent.tools.errors import ToolFailure
from cutagent.tools.registry import ToolRegistry
from cutagent_evaluation.m2b_dataset import HeldOutVideoGold
from cutagent_evaluation.m4a_agent import build_m4a_development_set
from cutagent_evaluation.m4b_agent import (
    M4BMetricsSummary,
    M4BTaskGold,
    M4BTrajectoryEvaluation,
    evaluate_m4b_trajectory,
    summarize_m4b_evaluations,
)

M4B5FailureType = Literal[
    "search_no_results",
    "tool_timeout",
    "invalid_tool_arguments",
    "artifact_not_allowed",
    "post_execution_validation_failure",
    "corrupt_media",
    "incompatible_concat_inputs",
    "invalid_subtitle_timing",
    "output_size_limit",
    "repeated_editor_stagnation",
]
M4B5TriggerMode = Literal[
    "empty_search",
    "pre_execute_failure",
    "post_execute_validation",
    "repeated_editor",
]
M4B5RecoveryFailure = Literal[
    "recovery_not_triggered",
    "recovery_format_error",
    "recovery_wrong_operation",
    "recovery_invalid_patch",
    "recovery_no_state_change",
    "recovery_wrong_tool_after_patch",
    "recovery_repeat_failure",
    "recovery_budget_exhaustion",
    "recovery_success_but_task_failed_later",
]
M4B5Variant = Literal["handoff_only", "compact_recovery"]
M4B5_DATASET_VERSION = "m4b5-observable-recovery-validation-v1"
M4B5_EVALUATOR_VERSION = "m4b5-recovery-evaluator-v1"

_EDIT_TOOLS = (
    "trim_video",
    "concat_videos",
    "change_speed",
    "add_subtitles",
    "reframe_video",
    "normalize_audio",
)


class FailureInjectionConfig(SchemaModel):
    """Environment-private deterministic rule; never accepted by an Agent API."""

    injection_id: Identifier
    task_id: Identifier
    failure_type: M4B5FailureType
    trigger_mode: M4B5TriggerMode
    trigger_tool_names: tuple[Identifier, ...] = Field(min_length=1)
    trigger_occurrence: int = Field(default=1, ge=1, le=8)
    single_shot: Literal[True] = True
    visibility: Literal["environment_private"] = "environment_private"
    expected_recovery_operations: tuple[Identifier, ...] = Field(min_length=1)

    @field_validator("trigger_tool_names", "expected_recovery_operations")
    @classmethod
    def unique_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("private injection values must be unique")
        return value


class FailureInjectionTrigger(SchemaModel):
    """Private audit record joined to a trajectory only by task/run identifiers."""

    injection_id: Identifier
    task_id: Identifier
    failure_type: M4B5FailureType
    triggered: bool
    eligible_call_count: int = Field(ge=0)
    trigger_tool_call_id: Identifier | None = None
    trigger_tool_name: Identifier | None = None
    public_trace_id: Identifier | None = None
    public_tool_status: Literal["success", "invalid", "timeout", "error"] | None = None
    public_error_category: Identifier | None = None
    trigger_event_id: Identifier | None = None
    trigger_source: Literal["tool_observation", "verification_result"] | None = None
    active_plan_node: Identifier | None = None

    @model_validator(mode="after")
    def trigger_fields_match(self) -> FailureInjectionTrigger:
        values = (
            self.trigger_tool_call_id,
            self.trigger_tool_name,
            self.public_trace_id,
            self.public_tool_status,
        )
        if self.triggered != all(item is not None for item in values):
            raise ValueError("triggered injection must identify the public tool outcome")
        if not self.triggered and any(
            item is not None
            for item in (
                self.trigger_event_id,
                self.trigger_source,
                self.public_error_category,
                self.active_plan_node,
            )
        ):
            raise ValueError("untriggered injection cannot reference public events")
        return self


class M4B5TaskGold(SchemaModel):
    """Private task annotation and environment control; never serialized to policy context."""

    gold_id: Identifier
    task_id: Identifier
    source_group_id: Identifier
    split: Literal["validation"] = "validation"
    task_gold: M4BTaskGold
    injection: FailureInjectionConfig

    @model_validator(mode="after")
    def identifiers_match(self) -> M4B5TaskGold:
        if len({self.task_id, self.task_gold.task_id, self.injection.task_id}) != 1:
            raise ValueError("M4B.5 private identifiers differ")
        if self.source_group_id != self.task_gold.source_group_id:
            raise ValueError("M4B.5 source-group annotations differ")
        return self


class M4B5ValidationCase(SchemaModel):
    task_input: TaskInput
    gold: M4B5TaskGold

    @model_validator(mode="after")
    def identifiers_match(self) -> M4B5ValidationCase:
        if self.task_input.task_id != self.gold.task_id:
            raise ValueError("public M4B.5 task and private annotation identifiers differ")
        return self


class M4B5RecoveryEvaluation(SchemaModel):
    task_id: Identifier
    run_id: Identifier
    variant: M4B5Variant
    failure_type: M4B5FailureType
    trigger: FailureInjectionTrigger
    recovery_triggered: bool
    recovery_decision_produced: bool
    recovery_decision_structured_valid: bool | None = None
    recovery_operation: Identifier | None = None
    recovery_operation_appropriate: bool | None = None
    affected_node_valid: bool | None = None
    patch_accepted: bool | None = None
    patch_rejection_reason: NonEmptyStr | None = None
    executable_recovery: bool
    recovery_attempted: bool
    subsequent_action_useful: bool
    failed_state_repaired: bool
    full_task_success: bool
    maximum_recovery_level: int = Field(ge=0, le=5)
    extra_steps: int = Field(ge=0)
    extra_tool_calls: int = Field(ge=0)
    additional_recovery_latency_ms: int = Field(ge=0)
    terminal_reason: Identifier
    recovery_failure: M4B5RecoveryFailure | None = None
    task_evaluation: M4BTrajectoryEvaluation


class RecoveryMetricSlice(SchemaModel):
    task_count: int = Field(ge=0)
    opportunity_count: int = Field(ge=0)
    failure_observation_rate: float = Field(ge=0, le=1)
    recovery_trigger_rate: float = Field(ge=0, le=1)
    recovery_decision_generation_rate: float | None = Field(default=None, ge=0, le=1)
    recovery_decision_structured_validity: float | None = Field(default=None, ge=0, le=1)
    patch_acceptance_rate: float | None = Field(default=None, ge=0, le=1)
    executable_recovery_rate: float | None = Field(default=None, ge=0, le=1)
    recovery_attempt_rate: float | None = Field(default=None, ge=0, le=1)
    recovery_success_rate: float | None = Field(default=None, ge=0, le=1)
    conditional_recovery_success_rate: float | None = Field(default=None, ge=0, le=1)
    task_success_rate: float = Field(ge=0, le=1)
    model_output_failure_rate: float = Field(ge=0, le=1)
    loop_stagnation_rate: float = Field(ge=0, le=1)
    budget_exhaustion_rate: float = Field(ge=0, le=1)
    average_extra_steps: float = Field(ge=0)
    average_extra_tool_calls: float = Field(ge=0)
    average_recovery_latency_ms: float = Field(ge=0)
    cumulative_level_counts: dict[int, int]
    highest_level_counts: dict[int, int]


class M4B5MetricsSummary(SchemaModel):
    variant: M4B5Variant
    overall: RecoveryMetricSlice
    per_failure_type: dict[M4B5FailureType, RecoveryMetricSlice]
    recovery_operation_counts: dict[str, int]
    recovery_failure_counts: dict[M4B5RecoveryFailure, int]
    frozen_m4b_metrics: M4BMetricsSummary


def _clone_gold(task_id: str, source: M4BTaskGold | Any) -> M4BTaskGold:
    return M4BTaskGold(
        gold_id=f"gold-{task_id}",
        task_id=task_id,
        source_group_id=source.source_group_id,
        category=source.category,
        expected_terminal_reason=source.expected_terminal_reason,
        relevant_time_ranges=source.relevant_time_ranges,
        required_tools=source.required_tools,
        expected_duration_ms=source.expected_duration_ms,
        duration_tolerance_ms=source.duration_tolerance_ms,
        expected_width=source.expected_width,
        expected_height=source.expected_height,
        expected_subtitle_text=source.expected_subtitle_text,
        expected_subtitle_range=source.expected_subtitle_range,
        designed_recovery_opportunity=True,
        evaluator_notes=("M4B.5 observable-failure validation only; not M5 benchmark data.",),
    )


def build_m4b5_recovery_set(
    videos: Sequence[HeldOutVideoGold],
    source_refs: Mapping[str, ArtifactRef],
    *,
    excluded_source_groups: set[str] | frozenset[str] = frozenset(),
) -> tuple[M4B5ValidationCase, ...]:
    """Create 34 new tasks with externally configured observable failures."""

    candidates = build_m4a_development_set(videos, source_refs)
    used_groups = {item.gold.source_group_id for item in candidates}
    overlap = used_groups & set(excluded_source_groups)
    if overlap:
        raise ValueError(f"M4B.5 source groups overlap excluded groups: {sorted(overlap)}")
    recovery_ops = {
        "search_no_results": (
            "retry_current_node",
            "modify_current_node",
        ),
        "tool_timeout": ("retry_current_node", "modify_current_node"),
        "invalid_tool_arguments": ("modify_current_node", "retry_current_node"),
        "artifact_not_allowed": (
            "modify_current_node",
            "insert_recovery_node",
            "retry_current_node",
        ),
        "post_execution_validation_failure": (
            "modify_current_node",
            "retry_current_node",
        ),
        "corrupt_media": ("cannot_recover", "modify_current_node", "retry_current_node"),
        "incompatible_concat_inputs": (
            "insert_recovery_node",
            "modify_current_node",
            "cannot_recover",
        ),
        "invalid_subtitle_timing": ("modify_current_node", "retry_current_node"),
        "output_size_limit": ("modify_current_node", "cannot_recover"),
        "repeated_editor_stagnation": ("modify_current_node", "cannot_recover"),
    }
    selections: list[tuple[M4B5FailureType, int, M4B5TriggerMode, tuple[str, ...], int]] = [
        *(
            ("search_no_results", index, "empty_search", ("search_video",), 1)
            for index in (0, 6, 12)
        ),
        *(
            ("tool_timeout", index, "pre_execute_failure", ("trim_video",), 1)
            for index in (1, 7, 13)
        ),
        *(
            ("invalid_tool_arguments", index, "pre_execute_failure", ("trim_video",), 1)
            for index in (41, 42, 43)
        ),
        *(
            ("artifact_not_allowed", index, "pre_execute_failure", ("change_speed",), 1)
            for index in (4, 10, 16)
        ),
        *(
            (
                "post_execution_validation_failure",
                index,
                "post_execute_validation",
                ("trim_video", "reframe_video"),
                1,
            )
            for index in (5, 11, 17)
        ),
        *(
            ("corrupt_media", index, "pre_execute_failure", ("trim_video",), 1)
            for index in (18, 24, 30)
        ),
        *(
            ("incompatible_concat_inputs", index, "pre_execute_failure", ("concat_videos",), 1)
            for index in (2, 8, 14)
        ),
        *(
            ("invalid_subtitle_timing", index, "pre_execute_failure", ("add_subtitles",), 1)
            for index in (3, 9, 15)
        ),
        *(
            ("output_size_limit", index, "pre_execute_failure", ("change_speed",), 1)
            for index in (22, 28, 34)
        ),
        *(
            (
                "repeated_editor_stagnation",
                index,
                "repeated_editor",
                _EDIT_TOOLS,
                2,
            )
            for index in (36, 37, 38)
        ),
        *(
            ("search_no_results", index, "empty_search", ("search_video",), 1)
            for index in (46, 47, 48, 49)
        ),
    ]
    cases: list[M4B5ValidationCase] = []
    for ordinal, (failure_type, source_index, mode, tool_names, occurrence) in enumerate(
        selections, start=1
    ):
        source = candidates[source_index]
        task_id = f"m4b5-recovery-{ordinal:03d}"
        task_input = source.task_input.model_copy(update={"task_id": task_id})
        task_gold = _clone_gold(task_id, source.gold)
        expected_operations = (
            ("cannot_recover",)
            if source.gold.category == "impossible"
            else recovery_ops[failure_type]
        )
        config = FailureInjectionConfig(
            injection_id=f"failure-{task_id}",
            task_id=task_id,
            failure_type=failure_type,
            trigger_mode=mode,
            trigger_tool_names=tool_names,
            trigger_occurrence=occurrence,
            expected_recovery_operations=expected_operations,
        )
        cases.append(
            M4B5ValidationCase(
                task_input=task_input,
                gold=M4B5TaskGold(
                    gold_id=f"private-{task_id}",
                    task_id=task_id,
                    source_group_id=source.gold.source_group_id,
                    task_gold=task_gold,
                    injection=config,
                ),
            )
        )
    if len(cases) != 34:
        raise AssertionError(f"M4B.5 recovery construction produced {len(cases)} cases")
    if len({item.task_input.task_id for item in cases}) != len(cases):
        raise AssertionError("M4B.5 public task identifiers are not unique")
    return tuple(cases)


_FAILURE_CATEGORY: dict[M4B5FailureType, ToolErrorCategory] = {
    "tool_timeout": "timeout",
    "invalid_tool_arguments": "invalid_call",
    "artifact_not_allowed": "artifact_not_allowed",
    "post_execution_validation_failure": "output_validation_failed",
    "corrupt_media": "corrupt_media",
    "incompatible_concat_inputs": "incompatible_media",
    "invalid_subtitle_timing": "invalid_subtitle",
    "output_size_limit": "output_too_large",
    "repeated_editor_stagnation": "output_validation_failed",
    "search_no_results": "output_validation_failed",
}

_PUBLIC_SUMMARY: dict[M4B5FailureType, str] = {
    "search_no_results": "search completed with no observable matching scenes",
    "tool_timeout": "tool execution exceeded its timeout",
    "invalid_tool_arguments": "tool arguments failed runtime validation",
    "artifact_not_allowed": "input artifact is not authorized in this execution",
    "post_execution_validation_failure": "generated media failed post-execution validation",
    "corrupt_media": "approved input media could not be decoded",
    "incompatible_concat_inputs": "concat inputs are not media-compatible",
    "invalid_subtitle_timing": "subtitle timing is invalid for the current media",
    "output_size_limit": "tool output exceeded the configured resource limit",
    "repeated_editor_stagnation": "equivalent editing produced no new observable progress",
}


def _normalized_call(
    call: ToolCall | Mapping[str, Any],
) -> tuple[ToolCall | None, dict[str, JsonValue]]:
    raw = call.model_dump(mode="python") if isinstance(call, SchemaModel) else call
    try:
        parsed = TOOL_CALL_ADAPTER.validate_python(raw)
    except ValueError:
        return None, {}
    return parsed, cast(dict[str, JsonValue], parsed.arguments.model_dump(mode="json"))


class DeterministicFailureInjectingRegistry(ToolRegistry):
    """Private decorator that exposes only ordinary public tool outcomes to M4B."""

    version = "m4b5-private-failure-injector-v1"

    def __init__(self, delegate: ToolRegistry, config: FailureInjectionConfig) -> None:
        super().__init__(
            artifact_store=delegate.artifact_store,
            trace_recorder=delegate.trace_recorder,
        )
        self._delegate = delegate
        self._config = config
        self._eligible_count = 0
        self._trigger: FailureInjectionTrigger | None = None
        self._editor_fingerprints: Counter[str] = Counter()

    def manifest(self) -> ToolManifest:
        return self._delegate.manifest()

    @property
    def injection_triggered(self) -> bool:
        return self._trigger is not None

    def private_trigger(self) -> FailureInjectionTrigger:
        if self._trigger is not None:
            return self._trigger
        return FailureInjectionTrigger(
            injection_id=self._config.injection_id,
            task_id=self._config.task_id,
            failure_type=self._config.failure_type,
            triggered=False,
            eligible_call_count=self._eligible_count,
        )

    def _should_inject(self, parsed: ToolCall) -> bool:
        if self._trigger is not None or parsed.tool_name not in self._config.trigger_tool_names:
            return False
        if self._config.trigger_mode == "repeated_editor":
            payload = parsed.model_dump(mode="json")
            payload.pop("tool_call_id", None)
            fingerprint = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            self._editor_fingerprints[fingerprint] += 1
            self._eligible_count = max(self._editor_fingerprints.values())
            return self._editor_fingerprints[fingerprint] == self._config.trigger_occurrence
        self._eligible_count += 1
        return self._eligible_count == self._config.trigger_occurrence

    def _remember(self, record: ToolExecutionRecord) -> None:
        trace_id = record.observation.details.get("trace_id")
        if not isinstance(trace_id, str):
            raise ValueError("injected public observation requires an opaque trace identifier")
        self._trigger = FailureInjectionTrigger(
            injection_id=self._config.injection_id,
            task_id=self._config.task_id,
            failure_type=self._config.failure_type,
            triggered=True,
            eligible_call_count=self._eligible_count,
            trigger_tool_call_id=record.observation.call_id,
            trigger_tool_name=record.observation.tool_name,
            public_trace_id=trace_id,
            public_tool_status=record.observation.status,
            public_error_category=record.observation.error_code,
        )

    def _empty_search(
        self,
        parsed: ToolCall,
        normalized: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolExecutionRecord:
        if not isinstance(parsed, SearchVideoCall):
            raise ValueError("empty-search injection requires a typed search_video call")
        started = datetime.now(UTC)
        trace_id = f"trace-{uuid4().hex}"
        trace = ToolTrace(
            trace_id=trace_id,
            execution_id=context.execution_id,
            tool_call_id=parsed.tool_call_id,
            tool_name=parsed.tool_name,
            tool_version="m3a-search-video-v1",
            normalized_arguments=normalized,
            parent_artifacts=(),
            started_at=started,
            ended_at=datetime.now(UTC),
            latency_ms=0,
            validation_results=(),
            status="success",
            cache_hit=False,
        )
        self.trace_recorder.write(trace)
        response = RetrievalResponse(
            query_id=f"query-{parsed.tool_call_id}",
            method="adaptive_hybrid",
            candidates=(),
            applied_filters=AppliedRetrievalFilters(
                video_id=parsed.arguments.video_id,
                approximate_time_range=parsed.arguments.approximate_time_range,
                required_evidence_types=parsed.arguments.required_evidence_types,
            ),
            latency_ms=0.0,
            cache_hit=False,
        )
        record = ToolExecutionRecord(
            observation=ToolObservation(
                call_id=parsed.tool_call_id,
                tool_name=parsed.tool_name,
                status="success",
                public_summary=_PUBLIC_SUMMARY["search_no_results"],
                details={
                    "trace_id": trace_id,
                    "cache_hit": False,
                    "response": response.model_dump(mode="json"),
                },
            ),
            trace=trace,
        )
        self._remember(record)
        return record

    def _post_validation_failure(self, record: ToolExecutionRecord) -> ToolExecutionRecord:
        if record.observation.status != "success":
            return record
        failed_check = ToolValidationCheck(
            check_name="injected_post_execution_integrity",
            passed=False,
            observed="failed",
            expected="passed",
        )
        trace = record.trace.model_copy(
            update={
                "trace_id": f"trace-{uuid4().hex}",
                "validation_results": (*record.trace.validation_results, failed_check),
            }
        )
        self.trace_recorder.write(trace)
        injected = ToolExecutionRecord(observation=record.observation, trace=trace)
        self._remember(injected)
        return injected

    def execute(
        self,
        call: ToolCall | Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionRecord:
        parsed, normalized = _normalized_call(call)
        if parsed is None or not self._should_inject(parsed):
            return self._delegate.execute(call, context)
        if self._config.trigger_mode == "empty_search":
            return self._empty_search(parsed, normalized, context)
        if self._config.trigger_mode == "post_execute_validation":
            return self._post_validation_failure(self._delegate.execute(call, context))
        started_at = datetime.now(UTC)
        started_ns = time.perf_counter_ns()
        category = _FAILURE_CATEGORY[self._config.failure_type]
        record = self._failure_record(
            tool_call_id=parsed.tool_call_id,
            tool_name=parsed.tool_name,
            tool_version="m4b5-environment-failure-v1",
            context=context,
            normalized_arguments=normalized,
            parents=(),
            started_at=started_at,
            started_ns=started_ns,
            failure=ToolFailure(category, _PUBLIC_SUMMARY[self._config.failure_type]),
        )
        self._remember(record)
        return record


def finalize_trigger(
    trigger: FailureInjectionTrigger,
    trajectory: M4BAgentTrajectory,
) -> FailureInjectionTrigger:
    """Join a private trigger record to its public event without changing the trajectory."""

    if not trigger.triggered or trigger.trigger_tool_call_id is None:
        return trigger
    if trigger.public_trace_id is None or trigger.trigger_tool_name is None:
        raise ValueError("triggered failure lacks public correlation fields")
    observation_event_id: str | None = None
    observation_sequence: int | None = None
    verification_event_id: str | None = None
    for envelope in trajectory.events:
        event = envelope.event
        if (
            observation_event_id is None
            and isinstance(event, ToolObservation)
            and event.call_id == trigger.trigger_tool_call_id
            and event.tool_name == trigger.trigger_tool_name
            and event.details.get("trace_id") == trigger.public_trace_id
        ):
            observation_event_id = envelope.event_id
            observation_sequence = envelope.sequence_no
        if (
            isinstance(event, VerificationResult)
            and event.verification_id == f"verify-{trigger.trigger_tool_call_id}"
            and event.status == "failed"
            and observation_sequence is not None
            and envelope.sequence_no > observation_sequence
            and verification_event_id is None
        ):
            verification_event_id = envelope.event_id
    source: Literal["tool_observation", "verification_result"] = (
        "verification_result" if trigger.public_tool_status == "success" else "tool_observation"
    )
    event_id = verification_event_id if source == "verification_result" else observation_event_id
    if event_id is None:
        raise ValueError("triggered failure has no matching public event")
    trigger_sequence = next(
        envelope.sequence_no for envelope in trajectory.events if envelope.event_id == event_id
    )
    active_plan_node = next(
        (
            envelope.event.summary.active_plan_node
            for envelope in trajectory.events
            if envelope.sequence_no > trigger_sequence
            and isinstance(envelope.event, StepOutcomeEvent)
        ),
        None,
    )
    return trigger.model_copy(
        update={
            "trigger_event_id": event_id,
            "trigger_source": source,
            "active_plan_node": active_plan_node,
        }
    )


def _first_recovery_index(trajectory: M4BAgentTrajectory) -> int | None:
    indices = [
        item.step_index
        for item in trajectory.policy_steps
        if item.operation in {"recover", "replan"}
    ]
    indices.extend(
        item.step_index
        for item in trajectory.policy_failures
        if item.operation in {"recover", "replan"}
    )
    return min(indices) if indices else None


def _event_after_trigger(
    trajectory: M4BAgentTrajectory,
    trigger: FailureInjectionTrigger,
    event_type: type[RecoveryOperationEvent],
) -> RecoveryOperationEvent | None:
    if trigger.trigger_event_id is None:
        return None
    trigger_sequence = next(
        item.sequence_no for item in trajectory.events if item.event_id == trigger.trigger_event_id
    )
    return next(
        (
            item.event
            for item in trajectory.events
            if item.sequence_no > trigger_sequence and isinstance(item.event, event_type)
        ),
        None,
    )


def _recovery_failure(
    *,
    trigger: FailureInjectionTrigger,
    recover_failure_count: int,
    decision_produced: bool,
    appropriate: bool | None,
    accepted: bool | None,
    executable: bool,
    attempted: bool,
    useful: bool,
    repaired: bool,
    task_success: bool,
    terminal_reason: str,
) -> M4B5RecoveryFailure | None:
    if not trigger.triggered:
        return "recovery_not_triggered"
    if recover_failure_count:
        return "recovery_format_error"
    if not decision_produced:
        return "recovery_not_triggered"
    if appropriate is False:
        return "recovery_wrong_operation"
    if accepted is False:
        return "recovery_invalid_patch"
    if not executable:
        return "recovery_no_state_change"
    if not attempted:
        return "recovery_no_state_change"
    if terminal_reason == "BUDGET_EXHAUSTED":
        return "recovery_budget_exhaustion"
    if not useful:
        return "recovery_wrong_tool_after_patch"
    if not repaired:
        return "recovery_repeat_failure"
    if not task_success:
        return "recovery_success_but_task_failed_later"
    return None


def evaluate_m4b5_trajectory(
    trajectory: M4BAgentTrajectory,
    gold: M4B5TaskGold,
    trigger: FailureInjectionTrigger,
    *,
    variant: M4B5Variant,
) -> M4B5RecoveryEvaluation:
    """Score public recovery mechanics while keeping expected behavior evaluator-only."""

    trigger = finalize_trigger(trigger, trajectory)
    base = evaluate_m4b_trajectory(trajectory, gold.task_gold, variant=variant)
    protocol_steps = [
        item for item in trajectory.policy_steps if item.operation in {"recover", "replan"}
    ]
    protocol_failures = [
        item for item in trajectory.policy_failures if item.operation in {"recover", "replan"}
    ]
    recovery_steps = [item for item in protocol_steps if item.operation == "recover"]
    recovery_failures = [item for item in protocol_failures if item.operation == "recover"]
    recovery_triggered = bool(protocol_steps or protocol_failures)
    recovery_step = recovery_steps[0] if recovery_steps else None
    decision = recovery_step.decision if recovery_step is not None else None
    decision_produced = bool(decision is not None)
    operation = getattr(decision, "recovery_type", None)
    appropriate = (
        operation in gold.injection.expected_recovery_operations if operation is not None else None
    )
    recovery_event = _event_after_trigger(trajectory, trigger, RecoveryOperationEvent)
    accepted = recovery_event.accepted if recovery_event is not None else None
    affected_valid: bool | None = bool(recovery_event and recovery_event.affected_node_ids)
    executable = bool(recovery_event is not None and recovery_event.accepted)
    patch_rejection_reason = recovery_event.rejection_reason if recovery_event is not None else None
    if isinstance(decision, CannotRecover):
        accepted = None
        affected_valid = None
        executable = False
        patch_rejection_reason = None
    if variant == "handoff_only":
        replan_step = next(
            (
                item
                for item in protocol_steps
                if item.operation == "replan" and isinstance(item.decision, ReplanDecision)
            ),
            None,
        )
        if replan_step is not None and isinstance(replan_step.decision, ReplanDecision):
            operation = "legacy_replan"
            revisions = {
                envelope.event.revision
                for envelope in trajectory.events
                if isinstance(envelope.event, PlanPatch)
            }
            accepted = replan_step.decision.requested_patch.revision in revisions
            affected_valid = bool(replan_step.decision.affected_plan_nodes)
            executable = accepted
            patch_rejection_reason = None if accepted else "runtime rejected legacy targeted replan"
            appropriate = None

    injected_index: int | None = None
    if trigger.trigger_tool_call_id is not None and trigger.public_trace_id is not None:
        injected_index = next(
            (
                index
                for index, record in enumerate(trajectory.tool_records)
                if record.observation.call_id == trigger.trigger_tool_call_id
                and record.observation.details.get("trace_id") == trigger.public_trace_id
            ),
            None,
        )
    subsequent_records = (
        trajectory.tool_records[injected_index + 1 :] if injected_index is not None else ()
    )
    attempted = executable and bool(subsequent_records)
    useful = attempted and subsequent_records[0].observation.status == "success"
    passed_ids = {
        item.verification_id
        for item in trajectory.final_state.verification_results
        if item.status == "passed"
    }
    repaired = useful and any(
        f"verify-{record.observation.call_id}" in passed_ids for record in subsequent_records
    )
    task_success = base.task_success
    maximum_level = 0
    if trigger.triggered:
        maximum_level = 0
    if decision_produced:
        maximum_level = 1
    if decision_produced and executable:
        maximum_level = 2
    if decision_produced and attempted and useful:
        maximum_level = 3
    if decision_produced and repaired:
        maximum_level = 4
    if decision_produced and task_success and repaired:
        maximum_level = 5

    recovery_index = _first_recovery_index(trajectory)
    extra_steps = (
        sum(item.step_index >= recovery_index for item in trajectory.policy_steps)
        + sum(item.step_index >= recovery_index for item in trajectory.policy_failures)
        if recovery_index is not None
        else 0
    )
    trigger_tool_index = (
        injected_index if injected_index is not None else len(trajectory.tool_records)
    )
    extra_tool_calls = len(trajectory.tool_records[trigger_tool_index + 1 :])
    recovery_latency = (
        sum(
            item.stats.latency_ms
            for item in trajectory.policy_steps
            if item.step_index >= recovery_index
        )
        + sum(
            item.latency_ms
            for item in trajectory.policy_failures
            if item.step_index >= recovery_index
        )
        + sum(item.trace.latency_ms for item in subsequent_records)
        if recovery_index is not None
        else 0
    )
    return M4B5RecoveryEvaluation(
        task_id=gold.task_id,
        run_id=trajectory.run_id,
        variant=variant,
        failure_type=gold.injection.failure_type,
        trigger=trigger,
        recovery_triggered=recovery_triggered,
        recovery_decision_produced=decision_produced,
        recovery_decision_structured_valid=(
            bool(recovery_steps) if recovery_steps or recovery_failures else None
        ),
        recovery_operation=operation,
        recovery_operation_appropriate=appropriate,
        affected_node_valid=affected_valid if recovery_event is not None else None,
        patch_accepted=accepted,
        patch_rejection_reason=patch_rejection_reason,
        executable_recovery=executable,
        recovery_attempted=attempted,
        subsequent_action_useful=useful,
        failed_state_repaired=repaired,
        full_task_success=task_success,
        maximum_recovery_level=maximum_level,
        extra_steps=extra_steps,
        extra_tool_calls=extra_tool_calls,
        additional_recovery_latency_ms=recovery_latency,
        terminal_reason=trajectory.terminal_reason,
        recovery_failure=(
            None
            if isinstance(decision, CannotRecover) and task_success
            else _recovery_failure(
                trigger=trigger,
                recover_failure_count=len(recovery_failures),
                decision_produced=decision_produced,
                appropriate=appropriate,
                accepted=accepted,
                executable=executable,
                attempted=attempted,
                useful=useful,
                repaired=repaired,
                task_success=task_success,
                terminal_reason=trajectory.terminal_reason,
            )
            if variant == "compact_recovery"
            else None
        ),
        task_evaluation=base,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _slice(evaluations: Sequence[M4B5RecoveryEvaluation]) -> RecoveryMetricSlice:
    total = len(evaluations)
    opportunities = [item for item in evaluations if item.trigger.triggered]
    decisions = [item for item in opportunities if item.recovery_decision_produced]
    patch_decisions = [item for item in opportunities if item.patch_accepted is not None]
    validity_values = [
        item.recovery_decision_structured_valid
        for item in opportunities
        if item.recovery_decision_structured_valid is not None
    ]
    attempted = [item for item in opportunities if item.recovery_attempted]
    cumulative = {
        level: sum(item.maximum_recovery_level >= level for item in opportunities)
        for level in range(6)
    }
    highest = Counter(item.maximum_recovery_level for item in opportunities)
    return RecoveryMetricSlice(
        task_count=total,
        opportunity_count=len(opportunities),
        failure_observation_rate=len(opportunities) / max(total, 1),
        recovery_trigger_rate=(
            sum(item.recovery_triggered for item in opportunities) / max(len(opportunities), 1)
        ),
        recovery_decision_generation_rate=_ratio(len(decisions), len(opportunities)),
        recovery_decision_structured_validity=(
            statistics.fmean(validity_values) if validity_values else None
        ),
        patch_acceptance_rate=_ratio(
            sum(item.patch_accepted is True for item in patch_decisions), len(patch_decisions)
        ),
        executable_recovery_rate=_ratio(
            sum(item.executable_recovery for item in opportunities), len(opportunities)
        ),
        recovery_attempt_rate=_ratio(len(attempted), len(opportunities)),
        recovery_success_rate=_ratio(
            sum(item.failed_state_repaired for item in opportunities), len(opportunities)
        ),
        conditional_recovery_success_rate=_ratio(
            sum(item.full_task_success for item in attempted), len(attempted)
        ),
        task_success_rate=sum(item.full_task_success for item in evaluations) / max(total, 1),
        model_output_failure_rate=sum(
            item.terminal_reason == "MODEL_OUTPUT_FAILURE" for item in evaluations
        )
        / max(total, 1),
        loop_stagnation_rate=sum(item.terminal_reason == "LOOP_DETECTED" for item in evaluations)
        / max(total, 1),
        budget_exhaustion_rate=sum(
            item.terminal_reason == "BUDGET_EXHAUSTED" for item in evaluations
        )
        / max(total, 1),
        average_extra_steps=(
            statistics.fmean(item.extra_steps for item in opportunities) if opportunities else 0.0
        ),
        average_extra_tool_calls=(
            statistics.fmean(item.extra_tool_calls for item in opportunities)
            if opportunities
            else 0.0
        ),
        average_recovery_latency_ms=(
            statistics.fmean(item.additional_recovery_latency_ms for item in opportunities)
            if opportunities
            else 0.0
        ),
        cumulative_level_counts=cumulative,
        highest_level_counts={level: highest[level] for level in range(6)},
    )


def summarize_m4b5_evaluations(
    evaluations: Sequence[M4B5RecoveryEvaluation],
) -> M4B5MetricsSummary:
    if not evaluations:
        raise ValueError("cannot summarize empty M4B.5 evaluations")
    variants = {item.variant for item in evaluations}
    if len(variants) != 1:
        raise ValueError("M4B.5 summary requires exactly one variant")
    failure_types = sorted({item.failure_type for item in evaluations})
    frozen = summarize_m4b_evaluations([item.task_evaluation for item in evaluations])
    return M4B5MetricsSummary(
        variant=next(iter(variants)),
        overall=_slice(evaluations),
        per_failure_type={
            failure_type: _slice(
                [item for item in evaluations if item.failure_type == failure_type]
            )
            for failure_type in failure_types
        },
        recovery_operation_counts=dict(
            Counter(
                item.recovery_operation
                for item in evaluations
                if item.recovery_operation is not None
            )
        ),
        recovery_failure_counts=dict(
            Counter(
                item.recovery_failure for item in evaluations if item.recovery_failure is not None
            )
        ),
        frozen_m4b_metrics=frozen,
    )


def trajectory_has_private_injection_data(trajectory: M4BAgentTrajectory) -> bool:
    """Conservative leak scan used by experiment and contract tests."""

    serialized = trajectory.model_dump_json().casefold()
    forbidden = (
        "failureinjectionconfig",
        "expected_recovery_operations",
        "environment_private",
        "source_group_id",
        '"split"',
        "gold_id",
        "benchmarkgold",
    )
    return any(token in serialized for token in forbidden)
