"""Frozen objective evaluator for CutAgentBench v0.1.

The evaluator reads persisted public trajectories and evaluator-private Gold.
It never mutates a trajectory and never feeds its results back into runtime.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal, cast

from pydantic import Field, JsonValue

from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.event import TerminalReasonCode
from cutagent.schemas.m4b_agent import M4BAgentTrajectory
from cutagent.schemas.media import TimeRange
from cutagent.schemas.tools import ToolName
from cutagent_evaluation.m5a_schemas import (
    AbsenceGoldConstraint,
    BenchmarkFailureCause,
    CutAgentBenchGold,
    DerivedArtifactGoldConstraint,
    DurationGoldConstraint,
    ExpectedTerminalBehavior,
    GroundingConstraint,
    ResolutionGoldConstraint,
    SceneOrderGoldConstraint,
    SpeedGoldConstraint,
    StreamGoldConstraint,
    SubtitleGoldConstraint,
    TaskFamily,
)


class RetrievedCandidateFact(SchemaModel):
    video_id: Identifier
    scene_id: Identifier
    time_range: TimeRange
    rank: int = Field(ge=1)


class SearchFact(SchemaModel):
    call_id: Identifier
    candidates: tuple[RetrievedCandidateFact, ...]


class ToolFact(SchemaModel):
    call_id: Identifier
    tool_name: ToolName
    status: Literal["success", "invalid", "timeout", "error"]
    normalized_arguments: dict[str, JsonValue]
    output_artifact_id: Identifier | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class TrajectoryFacts(SchemaModel):
    task_id: Identifier
    run_id: Identifier
    terminal_reason: TerminalReasonCode
    source_artifact_id: Identifier
    final_artifact_id: Identifier | None = None
    final_media_details: dict[str, JsonValue] = Field(default_factory=dict)
    searches: tuple[SearchFact, ...] = ()
    tools: tuple[ToolFact, ...] = ()
    policy_decisions: int = Field(ge=0)
    policy_failures: int = Field(ge=0)
    policy_repairs: int = Field(ge=0)
    repeated_action_count: int = Field(ge=0)
    loop_detected: bool = False
    recovery_decision_count: int = Field(ge=0)
    accepted_recovery_count: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class ConstraintResult(SchemaModel):
    constraint_type: Identifier
    mandatory: bool
    satisfied: bool
    observable: bool
    explanation: NonEmptyStr


class RetrievalMetricResult(SchemaModel):
    relevant_scene_count: int = Field(ge=0)
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    temporal_iou: float = Field(ge=0, le=1)


class CutAgentBenchEvaluation(SchemaModel):
    task_id: Identifier
    run_id: Identifier
    task_family: TaskFamily
    task_success: bool
    hard_constraint_satisfaction: float = Field(ge=0, le=1)
    all_mandatory_constraints_satisfied: bool
    correct_final_artifact: bool
    correct_impossible_refusal: bool
    retrieval: RetrievalMetricResult
    tool_selection_accuracy: float = Field(ge=0, le=1)
    tool_argument_validity: float = Field(ge=0, le=1)
    invalid_tool_call_rate: float = Field(ge=0, le=1)
    structured_output_validity: float = Field(ge=0, le=1)
    agent_steps: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    search_calls: int = Field(ge=0)
    repeated_action_rate: float = Field(ge=0, le=1)
    budget_exhausted: bool
    loop_stagnation: bool
    correct_termination: bool
    premature_finish: bool
    premature_refusal: bool
    recovery_opportunity: bool
    recovery_triggered: bool
    local_recovery: bool
    conditional_full_recovery: bool | None = None
    constraint_results: tuple[ConstraintResult, ...]
    primary_failure: BenchmarkFailureCause | None = None
    secondary_failures: tuple[BenchmarkFailureCause, ...] = ()


class CutAgentBenchMetricsSummary(SchemaModel):
    task_count: int = Field(gt=0)
    task_success_rate: float = Field(ge=0, le=1)
    hard_constraint_satisfaction: float = Field(ge=0, le=1)
    correct_final_artifact_rate: float = Field(ge=0, le=1)
    correct_impossible_refusal_rate: float | None = Field(default=None, ge=0, le=1)
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    temporal_iou: float = Field(ge=0, le=1)
    tool_selection_accuracy: float = Field(ge=0, le=1)
    tool_argument_validity: float = Field(ge=0, le=1)
    invalid_tool_call_rate: float = Field(ge=0, le=1)
    structured_output_validity: float = Field(ge=0, le=1)
    average_agent_steps: float = Field(ge=0)
    average_tool_calls: float = Field(ge=0)
    average_search_calls: float = Field(ge=0)
    repeated_action_rate: float = Field(ge=0, le=1)
    budget_exhaustion_rate: float = Field(ge=0, le=1)
    correct_termination_rate: float = Field(ge=0, le=1)
    premature_finish_rate: float = Field(ge=0, le=1)
    premature_refusal_rate: float = Field(ge=0, le=1)
    loop_stagnation_rate: float = Field(ge=0, le=1)
    recovery_opportunity_count: int = Field(ge=0)
    recovery_trigger_rate: float | None = Field(default=None, ge=0, le=1)
    local_recovery_rate: float | None = Field(default=None, ge=0, le=1)
    conditional_full_recovery_rate: float | None = Field(default=None, ge=0, le=1)
    success_by_family: dict[str, float]
    failure_counts: dict[str, int]


def _time_iou(left: TimeRange, right: TimeRange) -> float:
    intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    return intersection / union if union else 0.0


def _searches_from_trajectory(trajectory: M4BAgentTrajectory) -> tuple[SearchFact, ...]:
    searches: list[SearchFact] = []
    for record in trajectory.tool_records:
        if record.trace.tool_name != "search_video" or record.observation.status != "success":
            continue
        response = record.observation.details.get("response")
        candidates = response.get("candidates") if isinstance(response, dict) else None
        parsed: list[RetrievedCandidateFact] = []
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                video_id = candidate.get("video_id")
                scene_id = candidate.get("scene_id")
                rank = candidate.get("rank")
                if (
                    not isinstance(video_id, str)
                    or not isinstance(scene_id, str)
                    or not isinstance(rank, int)
                    or isinstance(rank, bool)
                ):
                    continue
                try:
                    parsed.append(
                        RetrievedCandidateFact(
                            video_id=video_id,
                            scene_id=scene_id,
                            time_range=TimeRange.model_validate(candidate["time_range"]),
                            rank=rank,
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        searches.append(SearchFact(call_id=record.observation.call_id, candidates=tuple(parsed)))
    return tuple(searches)


def _final_media_details(trajectory: M4BAgentTrajectory) -> dict[str, JsonValue]:
    final = trajectory.final_output_artifact
    if final is None:
        return {}
    for record in reversed(trajectory.tool_records):
        if record.trace.tool_name != "validate_media" or record.trace.status != "success":
            continue
        if any(parent.artifact_id == final.artifact_id for parent in record.trace.parent_artifacts):
            return dict(record.observation.details)
    return {}


def extract_trajectory_facts(trajectory: M4BAgentTrajectory) -> TrajectoryFacts:
    """Create immutable evaluator facts from a public M4B trajectory."""

    tools = tuple(
        ToolFact(
            call_id=record.observation.call_id,
            tool_name=cast(ToolName, record.trace.tool_name),
            status=record.observation.status,
            normalized_arguments=record.trace.normalized_arguments,
            output_artifact_id=(
                record.trace.output_artifact.artifact_id
                if record.trace.output_artifact is not None
                else None
            ),
            details=record.observation.details,
        )
        for record in trajectory.tool_records
    )
    recoveries = trajectory.final_state.recovery_events
    return TrajectoryFacts(
        task_id=trajectory.task_input.task_id,
        run_id=trajectory.run_id,
        terminal_reason=trajectory.terminal_reason,
        source_artifact_id=trajectory.task_input.video_ref.artifact_id,
        final_artifact_id=(
            trajectory.final_output_artifact.artifact_id
            if trajectory.final_output_artifact is not None
            else None
        ),
        final_media_details=_final_media_details(trajectory),
        searches=_searches_from_trajectory(trajectory),
        tools=tools,
        policy_decisions=len(trajectory.policy_steps),
        policy_failures=len(trajectory.policy_failures),
        policy_repairs=sum(step.stats.repair_count for step in trajectory.policy_steps)
        + sum(failure.repair_count for failure in trajectory.policy_failures),
        repeated_action_count=sum(item.occurrence_count for item in trajectory.diagnostics),
        loop_detected=trajectory.terminal_reason == "LOOP_DETECTED",
        recovery_decision_count=len(recoveries),
        accepted_recovery_count=sum(item.accepted for item in recoveries),
        latency_ms=trajectory.latency.total_ms,
    )


def _retrieval_metrics(facts: TrajectoryFacts, gold: CutAgentBenchGold) -> RetrievalMetricResult:
    relevant = tuple(zip(gold.relevant_scene_ids, gold.acceptable_time_ranges, strict=True))
    if not relevant:
        return RetrievalMetricResult(
            relevant_scene_count=0,
            recall_at_1=0.0,
            recall_at_5=0.0,
            mrr=0.0,
            temporal_iou=0.0,
        )
    best_ranks: list[int | None] = []
    best_top_iou: list[float] = []
    for scene_id, expected_range in relevant:
        ranks: list[int] = []
        top_iou = 0.0
        for search in facts.searches:
            for candidate in search.candidates:
                matching_video = candidate.video_id in gold.relevant_video_ids
                matching_scene = candidate.scene_id == scene_id
                temporal_match = _time_iou(candidate.time_range, expected_range) >= 0.5
                if matching_video and (matching_scene or temporal_match):
                    ranks.append(candidate.rank)
                if candidate.rank == 1 and matching_video:
                    top_iou = max(top_iou, _time_iou(candidate.time_range, expected_range))
        best_ranks.append(min(ranks) if ranks else None)
        best_top_iou.append(top_iou)
    denominator = len(relevant)
    return RetrievalMetricResult(
        relevant_scene_count=denominator,
        recall_at_1=sum(rank == 1 for rank in best_ranks) / denominator,
        recall_at_5=sum(rank is not None and rank <= 5 for rank in best_ranks) / denominator,
        mrr=sum(1 / rank if rank is not None else 0 for rank in best_ranks) / denominator,
        temporal_iou=sum(best_top_iou) / denominator,
    )


def _trim_ranges(facts: TrajectoryFacts) -> tuple[TimeRange, ...]:
    ranges: list[TimeRange] = []
    for tool in facts.tools:
        if tool.tool_name != "trim_video" or tool.status != "success":
            continue
        raw = tool.normalized_arguments.get("time_range")
        if isinstance(raw, dict):
            try:
                ranges.append(TimeRange.model_validate(raw))
            except ValueError:
                continue
    return tuple(ranges)


def _media_scalar(details: Mapping[str, JsonValue], key: str) -> int | None:
    value = details.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _resolution(details: Mapping[str, JsonValue]) -> tuple[int | None, int | None]:
    direct_width = _media_scalar(details, "width")
    direct_height = _media_scalar(details, "height")
    video = details.get("video")
    if direct_width is not None and direct_height is not None:
        return direct_width, direct_height
    if isinstance(video, dict):
        return _media_scalar(video, "width"), _media_scalar(video, "height")
    return None, None


def _subtitle_satisfied(tool: ToolFact, constraint: SubtitleGoldConstraint) -> bool:
    if tool.tool_name != "add_subtitles" or tool.status != "success":
        return False
    cues = tool.normalized_arguments.get("cues")
    if not isinstance(cues, list):
        return False
    expected_range = constraint.time_range.model_dump(mode="json")
    return any(
        isinstance(cue, dict)
        and cue.get("text") == constraint.text
        and cue.get("time_range") == expected_range
        for cue in cues
    )


def _scene_order_satisfied(facts: TrajectoryFacts, gold: CutAgentBenchGold) -> bool:
    constraint = next(
        (item for item in gold.objective_constraints if isinstance(item, SceneOrderGoldConstraint)),
        None,
    )
    if constraint is None:
        return True
    range_by_output: dict[str, TimeRange] = {}
    for tool in facts.tools:
        if tool.tool_name != "trim_video" or tool.status != "success":
            continue
        raw = tool.normalized_arguments.get("time_range")
        if isinstance(raw, dict) and tool.output_artifact_id is not None:
            range_by_output[tool.output_artifact_id] = TimeRange.model_validate(raw)
    expected_ranges = dict(zip(gold.relevant_scene_ids, gold.acceptable_time_ranges, strict=True))
    for tool in facts.tools:
        if tool.tool_name != "concat_videos" or tool.status != "success":
            continue
        inputs = tool.normalized_arguments.get("input_artifact_ids")
        if not isinstance(inputs, list) or len(inputs) != len(constraint.ordered_scene_ids):
            continue
        observed = [range_by_output.get(item) for item in inputs if isinstance(item, str)]
        if len(observed) != len(inputs) or any(item is None for item in observed):
            continue
        if all(
            _time_iou(expected_ranges[scene_id], actual) >= 0.9
            for scene_id, actual in zip(constraint.ordered_scene_ids, observed, strict=True)
            if actual is not None
        ):
            return True
    return False


def _constraint_results(
    facts: TrajectoryFacts,
    gold: CutAgentBenchGold,
) -> tuple[ConstraintResult, ...]:
    trim_ranges = _trim_ranges(facts)
    details = facts.final_media_details
    width, height = _resolution(details)
    results: list[ConstraintResult] = []
    for constraint in gold.objective_constraints:
        satisfied = False
        observable = True
        explanation = "objective evidence did not satisfy the frozen rule"
        if isinstance(constraint, GroundingConstraint):
            satisfied = all(
                any(_time_iou(expected, actual) >= 0.9 for actual in trim_ranges)
                for expected in constraint.acceptable_time_ranges
            )
            explanation = "all required source intervals were trimmed" if satisfied else explanation
        elif isinstance(constraint, DurationGoldConstraint):
            observed = _media_scalar(details, "duration_ms")
            satisfied = (
                observed is not None
                and abs(observed - constraint.target_ms) <= constraint.tolerance_ms
            )
            explanation = (
                f"observed duration={observed}, target={constraint.target_ms}"
                f"±{constraint.tolerance_ms}"
            )
        elif isinstance(constraint, ResolutionGoldConstraint):
            satisfied = (width, height) == (constraint.width, constraint.height)
            explanation = f"observed resolution={width}x{height}"
        elif isinstance(constraint, SubtitleGoldConstraint):
            satisfied = any(_subtitle_satisfied(tool, constraint) for tool in facts.tools)
            explanation = (
                "subtitle tool arguments match exact text and interval"
                if satisfied
                else explanation
            )
        elif isinstance(constraint, SceneOrderGoldConstraint):
            satisfied = _scene_order_satisfied(facts, gold)
            explanation = (
                "concat lineage matches required scene order" if satisfied else explanation
            )
        elif isinstance(constraint, SpeedGoldConstraint):
            satisfied = any(
                tool.tool_name == "change_speed"
                and tool.status == "success"
                and tool.normalized_arguments.get("speed_factor") == constraint.speed_factor
                for tool in facts.tools
            )
            explanation = "speed tool used the required factor" if satisfied else explanation
        elif isinstance(constraint, StreamGoldConstraint):
            video = details.get("video")
            audio = details.get("audio")
            has_video = isinstance(video, dict) or details.get("has_video") is True
            has_audio = bool(audio) or details.get("has_audio") is True
            satisfied = (not constraint.require_video or has_video) and (
                not constraint.require_audio or has_audio
            )
            explanation = f"observed video={has_video}, audio={has_audio}"
        elif isinstance(constraint, DerivedArtifactGoldConstraint):
            satisfied = (
                facts.final_artifact_id is not None
                and facts.final_artifact_id != facts.source_artifact_id
            )
            explanation = (
                "final artifact is derived and distinct from source" if satisfied else explanation
            )
        elif isinstance(constraint, AbsenceGoldConstraint):
            satisfied = facts.terminal_reason == "CANNOT_COMPLETE"
            explanation = "Agent correctly refused the absent request" if satisfied else explanation
        results.append(
            ConstraintResult(
                constraint_type=constraint.constraint_type,
                mandatory=constraint.mandatory,
                satisfied=satisfied,
                observable=observable,
                explanation=explanation,
            )
        )
    return tuple(results)


def _failure_attribution(
    facts: TrajectoryFacts,
    gold: CutAgentBenchGold,
    *,
    success: bool,
    correct_termination: bool,
    correct_artifact: bool,
    tools_satisfied: bool,
    retrieval: RetrievalMetricResult,
    constraints: tuple[ConstraintResult, ...],
) -> tuple[BenchmarkFailureCause | None, tuple[BenchmarkFailureCause, ...]]:
    if success:
        return None, ()
    if facts.terminal_reason == "MODEL_OUTPUT_FAILURE":
        return BenchmarkFailureCause.MODEL_FORMAT_ERROR, ()
    if facts.terminal_reason == "BUDGET_EXHAUSTED":
        return BenchmarkFailureCause.BUDGET_EXHAUSTION, ()
    if facts.terminal_reason == "LOOP_DETECTED":
        return BenchmarkFailureCause.LOOP_STAGNATION, ()
    failed = [tool for tool in facts.tools if tool.status != "success"]
    if any(tool.status == "invalid" for tool in failed):
        secondary = (
            (BenchmarkFailureCause.PREMATURE_REFUSAL,)
            if facts.terminal_reason == "CANNOT_COMPLETE"
            else ()
        )
        return BenchmarkFailureCause.INVALID_ARGUMENTS, secondary
    if failed:
        return BenchmarkFailureCause.TOOL_EXECUTION_FAILURE, ()
    if (
        facts.terminal_reason == "CANNOT_COMPLETE"
        and gold.expected_terminal_behavior.value == "SUCCESS"
    ):
        return BenchmarkFailureCause.PREMATURE_REFUSAL, ()
    if facts.terminal_reason == "SUCCESS" and gold.expected_terminal_behavior.value != "SUCCESS":
        return BenchmarkFailureCause.PREMATURE_FINISH, ()
    if not tools_satisfied:
        return BenchmarkFailureCause.WRONG_TOOL, ()
    if gold.relevant_scene_ids and retrieval.recall_at_5 < 1:
        return BenchmarkFailureCause.RETRIEVAL_ERROR, ()
    unsatisfied = {
        item.constraint_type for item in constraints if item.mandatory and not item.satisfied
    }
    if "grounding" in unsatisfied:
        return BenchmarkFailureCause.RETRIEVAL_ERROR, ()
    if not correct_artifact:
        return BenchmarkFailureCause.HANDOFF_ERROR, ()
    if unsatisfied & {"duration", "resolution", "subtitle", "scene_order", "speed"}:
        return BenchmarkFailureCause.INVALID_ARGUMENTS, ()
    if not correct_termination:
        return gold.primary_failure_if_unsolved, ()
    return BenchmarkFailureCause.VERIFICATION_ERROR, ()


def evaluate_facts(facts: TrajectoryFacts, gold: CutAgentBenchGold) -> CutAgentBenchEvaluation:
    """Apply the frozen TSR and partial-constraint rules to extracted facts."""

    if facts.task_id != gold.task_id:
        raise ValueError("trajectory facts and private Gold task IDs differ")
    constraints = _constraint_results(facts, gold)
    mandatory = tuple(item for item in constraints if item.mandatory)
    fraction = sum(item.satisfied for item in mandatory) / len(mandatory)
    all_mandatory = all(item.satisfied for item in mandatory)
    expected_terminal = gold.expected_terminal_behavior.value
    correct_termination = facts.terminal_reason == expected_terminal
    correct_refusal = (
        gold.expected_terminal_behavior == ExpectedTerminalBehavior.CANNOT_COMPLETE
        and facts.terminal_reason == "CANNOT_COMPLETE"
        and facts.final_artifact_id is None
    )
    if gold.expected_terminal_behavior == ExpectedTerminalBehavior.CANNOT_COMPLETE:
        correct_artifact = facts.final_artifact_id is None
    else:
        correct_artifact = facts.final_artifact_id is not None and bool(facts.final_media_details)
    observed_tools = {item.tool_name for item in facts.tools if item.status == "success"}
    required_tools = set(gold.required_tools)
    tools_satisfied = required_tools <= observed_tools
    selection = len(required_tools & observed_tools) / max(len(required_tools | observed_tools), 1)
    valid_args = sum(item.status != "invalid" for item in facts.tools)
    argument_validity = valid_args / max(len(facts.tools), 1)
    invalid_rate = 1 - argument_validity
    inference_count = facts.policy_decisions + facts.policy_failures
    structured = facts.policy_decisions / max(inference_count, 1)
    retrieval = _retrieval_metrics(facts, gold)
    task_success = correct_termination and tools_satisfied and all_mandatory and correct_artifact
    if gold.expected_terminal_behavior == ExpectedTerminalBehavior.CANNOT_COMPLETE:
        task_success = correct_refusal and all_mandatory and tools_satisfied
    recovery_opportunity = any(item.status != "success" for item in facts.tools)
    recovery_triggered = facts.recovery_decision_count > 0
    first_failure = next(
        (index for index, item in enumerate(facts.tools) if item.status != "success"), None
    )
    later_success = first_failure is not None and any(
        item.status == "success" for item in facts.tools[first_failure + 1 :]
    )
    local_recovery = recovery_triggered and facts.accepted_recovery_count > 0 and later_success
    conditional_full = task_success if recovery_opportunity else None
    repeated_rate = min(1.0, facts.repeated_action_count / max(len(facts.tools), 1))
    primary, secondary = _failure_attribution(
        facts,
        gold,
        success=task_success,
        correct_termination=correct_termination,
        correct_artifact=correct_artifact,
        tools_satisfied=tools_satisfied,
        retrieval=retrieval,
        constraints=constraints,
    )
    return CutAgentBenchEvaluation(
        task_id=gold.task_id,
        run_id=facts.run_id,
        task_family=gold.task_family,
        task_success=task_success,
        hard_constraint_satisfaction=fraction,
        all_mandatory_constraints_satisfied=all_mandatory,
        correct_final_artifact=correct_artifact,
        correct_impossible_refusal=correct_refusal,
        retrieval=retrieval,
        tool_selection_accuracy=selection,
        tool_argument_validity=argument_validity,
        invalid_tool_call_rate=invalid_rate,
        structured_output_validity=structured,
        agent_steps=inference_count,
        tool_calls=len(facts.tools),
        search_calls=sum(item.tool_name == "search_video" for item in facts.tools),
        repeated_action_rate=repeated_rate,
        budget_exhausted=facts.terminal_reason == "BUDGET_EXHAUSTED",
        loop_stagnation=facts.loop_detected,
        correct_termination=correct_termination,
        premature_finish=(facts.terminal_reason == "SUCCESS" and expected_terminal != "SUCCESS"),
        premature_refusal=(
            facts.terminal_reason == "CANNOT_COMPLETE" and expected_terminal == "SUCCESS"
        ),
        recovery_opportunity=recovery_opportunity,
        recovery_triggered=recovery_triggered,
        local_recovery=local_recovery,
        conditional_full_recovery=conditional_full,
        constraint_results=constraints,
        primary_failure=primary,
        secondary_failures=secondary,
    )


def evaluate_trajectory(
    trajectory: M4BAgentTrajectory,
    gold: CutAgentBenchGold,
) -> CutAgentBenchEvaluation:
    return evaluate_facts(extract_trajectory_facts(trajectory), gold)


def _mean(values: Sequence[float | bool]) -> float:
    return statistics.fmean(float(value) for value in values) if values else 0.0


def summarize_evaluations(
    evaluations: Sequence[CutAgentBenchEvaluation],
) -> CutAgentBenchMetricsSummary:
    if not evaluations:
        raise ValueError("at least one evaluation is required")
    impossible = [item for item in evaluations if item.task_family == TaskFamily.IMPOSSIBLE]
    opportunities = [item for item in evaluations if item.recovery_opportunity]
    by_family: dict[str, list[bool]] = {}
    for item in evaluations:
        by_family.setdefault(item.task_family.value, []).append(item.task_success)
    return CutAgentBenchMetricsSummary(
        task_count=len(evaluations),
        task_success_rate=_mean([item.task_success for item in evaluations]),
        hard_constraint_satisfaction=_mean(
            [item.hard_constraint_satisfaction for item in evaluations]
        ),
        correct_final_artifact_rate=_mean([item.correct_final_artifact for item in evaluations]),
        correct_impossible_refusal_rate=(
            _mean([item.correct_impossible_refusal for item in impossible]) if impossible else None
        ),
        recall_at_1=_mean([item.retrieval.recall_at_1 for item in evaluations]),
        recall_at_5=_mean([item.retrieval.recall_at_5 for item in evaluations]),
        mrr=_mean([item.retrieval.mrr for item in evaluations]),
        temporal_iou=_mean([item.retrieval.temporal_iou for item in evaluations]),
        tool_selection_accuracy=_mean([item.tool_selection_accuracy for item in evaluations]),
        tool_argument_validity=_mean([item.tool_argument_validity for item in evaluations]),
        invalid_tool_call_rate=_mean([item.invalid_tool_call_rate for item in evaluations]),
        structured_output_validity=_mean([item.structured_output_validity for item in evaluations]),
        average_agent_steps=_mean([item.agent_steps for item in evaluations]),
        average_tool_calls=_mean([item.tool_calls for item in evaluations]),
        average_search_calls=_mean([item.search_calls for item in evaluations]),
        repeated_action_rate=_mean([item.repeated_action_rate for item in evaluations]),
        budget_exhaustion_rate=_mean([item.budget_exhausted for item in evaluations]),
        correct_termination_rate=_mean([item.correct_termination for item in evaluations]),
        premature_finish_rate=_mean([item.premature_finish for item in evaluations]),
        premature_refusal_rate=_mean([item.premature_refusal for item in evaluations]),
        loop_stagnation_rate=_mean([item.loop_stagnation for item in evaluations]),
        recovery_opportunity_count=len(opportunities),
        recovery_trigger_rate=(
            _mean([item.recovery_triggered for item in opportunities]) if opportunities else None
        ),
        local_recovery_rate=(
            _mean([item.local_recovery for item in opportunities]) if opportunities else None
        ),
        conditional_full_recovery_rate=(
            _mean([item.task_success for item in opportunities]) if opportunities else None
        ),
        success_by_family={key: _mean(values) for key, values in sorted(by_family.items())},
        failure_counts=dict(
            Counter(
                item.primary_failure.value
                for item in evaluations
                if item.primary_failure is not None
            )
        ),
    )
