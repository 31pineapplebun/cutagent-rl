"""Evaluator-private M4B held-out tasks, recovery metrics, and attribution."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal, cast

from pydantic import Field, JsonValue, model_validator

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.agent import AgentTrajectory, FinishDecision, ReplanDecision, ToolDecision
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.event import PlanPatch
from cutagent.schemas.m4b_agent import M4BAgentTrajectory
from cutagent.schemas.media import TimeRange
from cutagent.schemas.task_input import (
    AspectRatioConstraint,
    DurationConstraint,
    ObservableConstraint,
    RequiredContentConstraint,
    TaskInput,
)
from cutagent_evaluation.m2b_dataset import HeldOutVideoGold
from cutagent_evaluation.m4a_agent import (
    M4ATaskGold,
    M4ATrajectoryEvaluation,
    evaluate_trajectory,
)

M4BTaskCategory = Literal[
    "search_trim",
    "search_trim_validate",
    "retrieve_concat",
    "subtitle",
    "speed",
    "reframe",
    "multi_constraint",
    "hard_negative",
    "impossible",
]
M4BFailureCause = Literal[
    "planning_error",
    "retrieval_error",
    "upstream_perception_error",
    "wrong_tool",
    "invalid_arguments",
    "tool_execution_failure",
    "handoff_error",
    "recovery_decision_error",
    "recovery_patch_error",
    "verification_error",
    "premature_success",
    "premature_refusal",
    "loop/stagnation",
    "budget_exhaustion",
    "model_format_error",
    "system_error",
]
M4BTrajectory = AgentTrajectory | M4BAgentTrajectory


class M4BTaskGold(SchemaModel):
    """Private validation annotation; runtime interfaces never accept this type."""

    gold_id: Identifier
    task_id: Identifier
    source_group_id: Identifier
    split: Literal["validation"] = "validation"
    category: M4BTaskCategory
    expected_terminal_reason: Literal["SUCCESS", "CANNOT_COMPLETE"]
    relevant_time_ranges: tuple[TimeRange, ...] = ()
    required_tools: tuple[Identifier, ...] = ()
    expected_duration_ms: int | None = Field(default=None, gt=0)
    duration_tolerance_ms: int = Field(default=250, ge=0)
    expected_width: int | None = Field(default=None, gt=0)
    expected_height: int | None = Field(default=None, gt=0)
    expected_subtitle_text: NonEmptyStr | None = None
    expected_subtitle_range: TimeRange | None = None
    designed_recovery_opportunity: bool = False
    evaluator_notes: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def dimensions_are_paired(self) -> M4BTaskGold:
        if (self.expected_width is None) != (self.expected_height is None):
            raise ValueError("expected dimensions must be supplied together")
        return self


class M4BValidationCase(SchemaModel):
    task_input: TaskInput
    gold: M4BTaskGold

    @model_validator(mode="after")
    def identifiers_match(self) -> M4BValidationCase:
        if self.task_input.task_id != self.gold.task_id:
            raise ValueError("public M4B task and private Gold identifiers differ")
        return self


class M4BTrajectoryEvaluation(SchemaModel):
    task_id: Identifier
    run_id: Identifier
    variant: NonEmptyStr
    category: M4BTaskCategory
    task_success: bool
    editing_task_success: bool
    correct_impossible_refusal: bool
    hard_constraints_satisfied: bool
    correct_final_artifact: bool
    structured_output_validity: float = Field(ge=0, le=1)
    model_output_failure: bool
    invalid_tool_call_rate: float = Field(ge=0, le=1)
    loop_or_stagnation: bool
    repeated_editor_rate: float = Field(ge=0, le=1)
    budget_exhausted: bool
    premature_finish: bool
    premature_refusal: bool
    agent_steps: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    search_calls: int = Field(ge=0)
    recovery_opportunity: bool
    recovery_attempted: bool
    recovery_decision_validity: float | None = Field(default=None, ge=0, le=1)
    recovery_succeeded: bool
    conditional_recovery_success: bool | None = None
    recovery_additional_steps: int | None = Field(default=None, ge=0)
    recovery_additional_latency_ms: int | None = Field(default=None, ge=0)
    replan_recovery_structured_validity: float | None = Field(default=None, ge=0, le=1)
    patch_acceptance_rate: float | None = Field(default=None, ge=0, le=1)
    patch_rejection_reasons: tuple[NonEmptyStr, ...] = ()
    first_error_type: M4BFailureCause | None = None
    primary_failure: M4BFailureCause | None = None
    secondary_failures: tuple[M4BFailureCause, ...] = ()


class M4BMetricsSummary(SchemaModel):
    variant: NonEmptyStr
    task_count: int = Field(gt=0)
    task_success_rate: float = Field(ge=0, le=1)
    editing_task_success_rate: float = Field(ge=0, le=1)
    correct_impossible_refusal_rate: float = Field(ge=0, le=1)
    correct_final_artifact_rate: float = Field(ge=0, le=1)
    hard_constraint_satisfaction_rate: float = Field(ge=0, le=1)
    structured_output_validity: float = Field(ge=0, le=1)
    model_output_failure_rate: float = Field(ge=0, le=1)
    invalid_tool_call_rate: float = Field(ge=0, le=1)
    loop_stagnation_rate: float = Field(ge=0, le=1)
    repeated_editor_rate: float = Field(ge=0, le=1)
    budget_exhaustion_rate: float = Field(ge=0, le=1)
    premature_finish_rate: float = Field(ge=0, le=1)
    premature_refusal_rate: float = Field(ge=0, le=1)
    average_agent_steps: float = Field(ge=0)
    average_tool_calls: float = Field(ge=0)
    average_search_calls: float = Field(ge=0)
    recovery_opportunity_count: int = Field(ge=0)
    recovery_attempt_rate: float | None = Field(default=None, ge=0, le=1)
    recovery_decision_validity: float | None = Field(default=None, ge=0, le=1)
    recovery_success_rate: float | None = Field(default=None, ge=0, le=1)
    conditional_recovery_success: float | None = Field(default=None, ge=0, le=1)
    median_additional_recovery_steps: float | None = Field(default=None, ge=0)
    median_additional_recovery_latency_ms: float | None = Field(default=None, ge=0)
    replan_recovery_structured_validity: float | None = Field(default=None, ge=0, le=1)
    patch_acceptance_rate: float | None = Field(default=None, ge=0, le=1)
    patch_rejection_counts: dict[str, int]
    failure_counts: dict[M4BFailureCause, int]
    failure_origin_fraction: dict[Literal["protocol_runtime", "policy_model", "upstream"], float]
    success_by_category: dict[M4BTaskCategory, float]


class M4BRecoveryCaseRecord(SchemaModel):
    task_id: Identifier
    run_id: Identifier
    variant: NonEmptyStr
    first_error_type: M4BFailureCause
    recovery_possible: bool
    recovery_attempted: bool
    recovery_decision: dict[str, JsonValue] | None = None
    patch_accepted: bool | None = None
    changed_state: tuple[NonEmptyStr, ...] = ()
    subsequent_tool_improved_state: bool
    eventual_task_success: bool


def _scene_description(video: HeldOutVideoGold, scene_index: int) -> str:
    scene = video.scenes[scene_index]
    return (
        f"the {scene.primary_entity} is {scene.action.replace('_', ' ')} "
        f"near the {scene.companion_entity}"
    )


def build_m4b_validation_set(
    videos: Sequence[HeldOutVideoGold],
    source_refs: Mapping[str, ArtifactRef],
    *,
    excluded_source_groups: set[str],
) -> tuple[M4BValidationCase, ...]:
    """Create 40 frozen validation tasks from source groups excluded from M4A."""

    if len(videos) < 2:
        raise ValueError("M4B validation requires at least two multi-scene videos")
    observed_groups = {item.source_group_id for item in videos}
    overlap = observed_groups & excluded_source_groups
    if overlap:
        raise ValueError(f"M4B validation source groups overlap M4A: {sorted(overlap)}")
    cases: list[M4BValidationCase] = []

    def add(
        *,
        category: M4BTaskCategory,
        video_index: int,
        instruction: str,
        ranges: tuple[TimeRange, ...],
        tools: tuple[str, ...],
        duration_ms: int | None,
        constraints: tuple[ObservableConstraint, ...] = (),
        dimensions: tuple[int, int] | None = None,
        subtitle_text: str | None = None,
        subtitle_range: TimeRange | None = None,
        expected_terminal: Literal["SUCCESS", "CANNOT_COMPLETE"] = "SUCCESS",
        recovery_opportunity: bool = False,
    ) -> None:
        video = videos[video_index % len(videos)]
        reference = source_refs.get(video.source_group_id)
        if reference is None:
            raise ValueError(f"missing M4B source reference for {video.source_group_id}")
        task_id = f"m4b-val-{len(cases) + 1:03d}"
        cases.append(
            M4BValidationCase(
                task_input=TaskInput(
                    task_id=task_id,
                    video_ref=reference,
                    instruction=instruction,
                    user_constraints=constraints,
                ),
                gold=M4BTaskGold(
                    gold_id=f"gold-{task_id}",
                    task_id=task_id,
                    source_group_id=video.source_group_id,
                    category=category,
                    expected_terminal_reason=expected_terminal,
                    relevant_time_ranges=ranges,
                    required_tools=tools,
                    expected_duration_ms=duration_ms,
                    expected_width=dimensions[0] if dimensions else None,
                    expected_height=dimensions[1] if dimensions else None,
                    expected_subtitle_text=subtitle_text,
                    expected_subtitle_range=subtitle_range,
                    designed_recovery_opportunity=recovery_opportunity,
                    evaluator_notes=(
                        "M4B held-out validation only; not M5 locked benchmark data.",
                    ),
                ),
            )
        )

    for index in range(5):
        video = videos[index % len(videos)]
        scene_index = (index + 1) % 4
        scene = video.scenes[scene_index]
        description = _scene_description(video, scene_index)
        add(
            category="search_trim",
            video_index=index,
            instruction=f"Find the full scene where {description}; export only that scene.",
            ranges=(scene.nominal_time_range,),
            tools=("search_video", "trim_video", "validate_media"),
            duration_ms=3000,
            constraints=(RequiredContentConstraint(description=description),),
            recovery_opportunity=index >= 2,
        )
        add(
            category="search_trim_validate",
            video_index=index,
            instruction=(
                f"Find visible marker {scene.ocr_text}, export its complete scene, validate it, "
                "and return only the validated artifact."
            ),
            ranges=(scene.nominal_time_range,),
            tools=("search_video", "trim_video", "validate_media"),
            duration_ms=3000,
            recovery_opportunity=index >= 2,
        )
        first = video.scenes[index % 2]
        second = video.scenes[2 + (index % 2)]
        add(
            category="retrieve_concat",
            video_index=index,
            instruction=(
                f"Create one clip with marker {first.ocr_text} first and marker "
                f"{second.ocr_text} second; use each full scene and validate the result."
            ),
            ranges=(first.nominal_time_range, second.nominal_time_range),
            tools=("search_video", "trim_video", "concat_videos", "validate_media"),
            duration_ms=6000,
            constraints=(DurationConstraint(min_ms=5700, max_ms=6300),),
            recovery_opportunity=True,
        )
        subtitle = f"M4B HOLDOUT {index + 1}"
        add(
            category="subtitle",
            video_index=index,
            instruction=(
                f"Find quote '{scene.transcript}', export that complete scene, burn subtitle "
                f"'{subtitle}' from 0 to 1500 ms, validate it, and return the final artifact."
            ),
            ranges=(scene.nominal_time_range,),
            tools=("search_video", "trim_video", "add_subtitles", "validate_media"),
            duration_ms=3000,
            subtitle_text=subtitle,
            subtitle_range=TimeRange(start_ms=0, end_ms=1500),
            recovery_opportunity=index >= 3,
        )
        speed = 1.5 if index % 2 == 0 else 0.75
        expected_duration = round(3000 / speed)
        add(
            category="speed",
            video_index=index,
            instruction=(
                f"Find marker {scene.ocr_text}, export its full scene, change speed to "
                f"{speed:g}x, validate it, and return the speed-adjusted artifact."
            ),
            ranges=(scene.nominal_time_range,),
            tools=("search_video", "trim_video", "change_speed", "validate_media"),
            duration_ms=expected_duration,
            constraints=(
                DurationConstraint(
                    min_ms=expected_duration - 300,
                    max_ms=expected_duration + 300,
                ),
            ),
            recovery_opportunity=index >= 3,
        )
        reframe_scene = video.scenes[(scene_index + 1) % 4]
        add(
            category="reframe",
            video_index=index,
            instruction=(
                f"Find marker {reframe_scene.ocr_text}, export that full scene, crop it to "
                "portrait 216x384, validate it, and return only the portrait artifact."
            ),
            ranges=(reframe_scene.nominal_time_range,),
            tools=("search_video", "trim_video", "reframe_video", "validate_media"),
            duration_ms=3000,
            constraints=(AspectRatioConstraint(width=9, height=16),),
            dimensions=(216, 384),
            recovery_opportunity=index >= 3,
        )
    for index in range(4):
        video = videos[index % len(videos)]
        scene = video.scenes[2]
        add(
            category="multi_constraint",
            video_index=index,
            instruction=(
                f"Find the full scene where {_scene_description(video, 2)}, change speed to "
                "2x, crop to 216x384, validate, and return only the final artifact."
            ),
            ranges=(scene.nominal_time_range,),
            tools=(
                "search_video",
                "trim_video",
                "change_speed",
                "reframe_video",
                "validate_media",
            ),
            duration_ms=1500,
            constraints=(
                DurationConstraint(min_ms=1200, max_ms=1800),
                AspectRatioConstraint(width=9, height=16),
            ),
            dimensions=(216, 384),
            recovery_opportunity=True,
        )
        add(
            category="hard_negative",
            video_index=index,
            instruction=(
                "The same object appears in several scenes with different actions. Find only "
                f"the scene where {_scene_description(video, 2)}, export the complete scene, "
                "validate it, and do not substitute a stationary or opposite-direction scene."
            ),
            ranges=(scene.nominal_time_range,),
            tools=("search_video", "trim_video", "validate_media"),
            duration_ms=3000,
            constraints=(RequiredContentConstraint(description=_scene_description(video, 2)),),
            recovery_opportunity=True,
        )
    for index in range(2):
        add(
            category="impossible",
            video_index=index,
            instruction=(
                "Find a scene where a silver airplane lands in a lake, export exactly five "
                "seconds, and refuse rather than inventing evidence if it is absent."
            ),
            ranges=(),
            tools=("search_video",),
            duration_ms=None,
            expected_terminal="CANNOT_COMPLETE",
            constraints=(
                DurationConstraint(min_ms=5000, max_ms=5000),
                RequiredContentConstraint(description="silver airplane lands in a lake"),
            ),
            recovery_opportunity=False,
        )
    if len(cases) != 40:
        raise AssertionError(f"M4B validation construction produced {len(cases)} cases")
    return tuple(cases)


def _m4a_gold(gold: M4BTaskGold) -> M4ATaskGold:
    return M4ATaskGold(
        gold_id=gold.gold_id,
        task_id=gold.task_id,
        source_group_id=gold.source_group_id,
        category=gold.category,
        expected_terminal_reason=gold.expected_terminal_reason,
        relevant_time_ranges=gold.relevant_time_ranges,
        required_tools=gold.required_tools,
        expected_duration_ms=gold.expected_duration_ms,
        duration_tolerance_ms=gold.duration_tolerance_ms,
        expected_width=gold.expected_width,
        expected_height=gold.expected_height,
        expected_subtitle_text=gold.expected_subtitle_text,
        expected_subtitle_range=gold.expected_subtitle_range,
        evaluator_notes=gold.evaluator_notes,
    )


def _time_iou(left: TimeRange, right: TimeRange) -> float:
    intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    return intersection / union if union else 0.0


def _first_observable_error(
    trajectory: M4BTrajectory, gold: M4BTaskGold
) -> tuple[int | None, M4BFailureCause | None]:
    editing = {
        "trim_video",
        "concat_videos",
        "change_speed",
        "add_subtitles",
        "reframe_video",
        "normalize_audio",
    }
    for index, record in enumerate(trajectory.tool_records):
        if record.observation.status != "success":
            cause: M4BFailureCause = (
                "invalid_arguments"
                if record.trace.error_category == "invalid_call"
                else "tool_execution_failure"
            )
            return index, cause
        if record.trace.tool_name == "search_video" and gold.relevant_time_ranges:
            response = record.observation.details.get("response")
            candidates = response.get("candidates") if isinstance(response, dict) else None
            raw_range = (
                candidates[0].get("time_range")
                if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict)
                else None
            )
            if not isinstance(raw_range, dict):
                return index, "upstream_perception_error"
            observed = TimeRange.model_validate(raw_range)
            if not any(_time_iou(item, observed) >= 0.9 for item in gold.relevant_time_ranges):
                return index, "retrieval_error"
        if record.trace.tool_name == "trim_video" and gold.relevant_time_ranges:
            raw_range = record.trace.normalized_arguments.get("time_range")
            if not isinstance(raw_range, dict):
                return index, "invalid_arguments"
            observed = TimeRange.model_validate(raw_range)
            if not any(_time_iou(item, observed) >= 0.9 for item in gold.relevant_time_ranges):
                return index, "invalid_arguments"
        if record.trace.tool_name in editing and record.trace.tool_name not in gold.required_tools:
            return index, "wrong_tool"
    return None, None


def _legacy_patch_outcomes(
    decisions: list[ReplanDecision], emitted_revisions: set[int]
) -> tuple[int, tuple[str, ...]]:
    """Count accepted legacy replans without relying on M4B recovery events."""

    accepted = sum(decision.requested_patch.revision in emitted_revisions for decision in decisions)
    rejected = tuple(
        "runtime rejected invalid targeted legacy replan"
        for decision in decisions
        if decision.requested_patch.revision not in emitted_revisions
    )
    return accepted, rejected


def _recovery_protocol_counts(
    trajectory: M4BTrajectory,
) -> tuple[int, int, int, int, tuple[str, ...]]:
    successful = sum(step.operation in {"replan", "recover"} for step in trajectory.policy_steps)
    failed = sum(
        failure.operation in {"replan", "recover"} for failure in trajectory.policy_failures
    )
    attempts = successful + failed
    legacy_decisions = [
        step.decision
        for step in trajectory.policy_steps
        if step.operation == "replan" and isinstance(step.decision, ReplanDecision)
    ]
    revisions = {
        envelope.event.revision
        for envelope in trajectory.events
        if isinstance(envelope.event, PlanPatch)
    }
    accepted = 0
    rejected: tuple[str, ...] = ()
    if legacy_decisions:
        accepted, rejected = _legacy_patch_outcomes(legacy_decisions, revisions)
    elif isinstance(trajectory, M4BAgentTrajectory):
        accepted = sum(event.accepted for event in trajectory.final_state.recovery_events)
        rejected = tuple(
            event.rejection_reason
            for event in trajectory.final_state.recovery_events
            if not event.accepted and event.rejection_reason is not None
        )
    return attempts, successful, accepted, failed, rejected


def _repeated_editor_rate(trajectory: M4BTrajectory) -> float:
    edits = [
        record
        for record in trajectory.tool_records
        if record.trace.tool_name
        in {
            "trim_video",
            "concat_videos",
            "change_speed",
            "add_subtitles",
            "reframe_video",
            "normalize_audio",
        }
    ]
    counts: Counter[str] = Counter()
    repeats = 0
    for record in edits:
        fingerprint = json.dumps(
            {
                "tool": record.trace.tool_name,
                "arguments": record.trace.normalized_arguments,
                "parents": [item.artifact_id for item in record.trace.parent_artifacts],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        counts[fingerprint] += 1
        repeats += counts[fingerprint] > 1
    return repeats / max(len(edits), 1)


def _failure_attribution(
    trajectory: M4BTrajectory,
    gold: M4BTaskGold,
    base: M4ATrajectoryEvaluation,
    first_error: M4BFailureCause | None,
) -> tuple[M4BFailureCause | None, tuple[M4BFailureCause, ...]]:
    if base.task_success:
        return None, ()
    secondary: list[M4BFailureCause] = []
    if trajectory.terminal_reason == "MODEL_OUTPUT_FAILURE":
        if any(item.operation == "recover" for item in trajectory.policy_failures):
            secondary.append("recovery_decision_error")
        return "model_format_error", tuple(secondary)
    if trajectory.terminal_reason == "SYSTEM_ERROR":
        return "system_error", ()
    if trajectory.terminal_reason == "BUDGET_EXHAUSTED":
        return "budget_exhaustion", ()
    if trajectory.terminal_reason == "LOOP_DETECTED":
        if any(item.diagnostic_type == "repeated_editor" for item in trajectory.diagnostics):
            return "handoff_error", ("loop/stagnation",)
        return "loop/stagnation", ()
    if (
        trajectory.terminal_reason == "CANNOT_COMPLETE"
        and gold.expected_terminal_reason == "SUCCESS"
    ):
        return "premature_refusal", ()
    if trajectory.terminal_reason == "SUCCESS" and gold.expected_terminal_reason != "SUCCESS":
        return "premature_success", ()
    if isinstance(trajectory, M4BAgentTrajectory) and any(
        not event.accepted for event in trajectory.final_state.recovery_events
    ):
        return "recovery_patch_error", ()
    if first_error is not None:
        return first_error, ()
    if base.primary_failure == "planning_error":
        return "planning_error", ()
    if base.primary_failure == "verification_error":
        return "verification_error", ()
    if base.primary_failure == "wrong_tool":
        return "wrong_tool", ()
    return "planning_error", ()


def evaluate_m4b_trajectory(
    trajectory: M4BTrajectory,
    gold: M4BTaskGold,
    *,
    variant: str,
) -> M4BTrajectoryEvaluation:
    base = evaluate_trajectory(cast(AgentTrajectory, trajectory), _m4a_gold(gold))
    first_index, first_error = _first_observable_error(trajectory, gold)
    opportunity = bool(
        first_index is not None
        and gold.designed_recovery_opportunity
        and gold.expected_terminal_reason == "SUCCESS"
    )
    attempts, valid_decisions, accepted, failed_decisions, rejection_reasons = (
        _recovery_protocol_counts(trajectory)
    )
    later_tool = first_index is not None and len(trajectory.tool_records) > first_index + 1
    recovery_attempted = opportunity and (attempts > 0 or later_tool)
    recovery_succeeded = recovery_attempted and base.task_success
    first_step = None
    if first_index is not None:
        call_id = trajectory.tool_records[first_index].observation.call_id
        first_step = next(
            (
                step.step_index
                for step in trajectory.policy_steps
                if isinstance(step.decision, ToolDecision)
                and step.decision.tool_call.tool_call_id == call_id
            ),
            None,
        )
    later_steps = (
        [step for step in trajectory.policy_steps if step.step_index > first_step]
        if first_step is not None
        else []
    )
    later_failures = (
        [failure for failure in trajectory.policy_failures if failure.step_index > first_step]
        if first_step is not None
        else []
    )
    additional_latency = sum(step.stats.latency_ms for step in later_steps) + sum(
        failure.latency_ms for failure in later_failures
    )
    if first_index is not None:
        additional_latency += sum(
            record.trace.latency_ms for record in trajectory.tool_records[first_index + 1 :]
        )
    inference_count = len(trajectory.policy_steps) + len(trajectory.policy_failures)
    finish_count = sum(
        isinstance(step.decision, FinishDecision) for step in trajectory.policy_steps
    )
    primary, secondary = _failure_attribution(trajectory, gold, base, first_error)
    return M4BTrajectoryEvaluation(
        task_id=gold.task_id,
        run_id=trajectory.run_id,
        variant=variant,
        category=gold.category,
        task_success=base.task_success,
        editing_task_success=base.task_success and gold.category != "impossible",
        correct_impossible_refusal=base.task_success and gold.category == "impossible",
        hard_constraints_satisfied=base.hard_constraints_satisfied,
        correct_final_artifact=base.correct_final_artifact,
        structured_output_validity=base.structured_output_validity,
        model_output_failure=trajectory.terminal_reason == "MODEL_OUTPUT_FAILURE",
        invalid_tool_call_rate=base.invalid_tool_call_rate,
        loop_or_stagnation=trajectory.terminal_reason == "LOOP_DETECTED",
        repeated_editor_rate=_repeated_editor_rate(trajectory),
        budget_exhausted=trajectory.terminal_reason == "BUDGET_EXHAUSTED",
        premature_finish=finish_count > 0 and not base.task_success,
        premature_refusal=(
            trajectory.terminal_reason == "CANNOT_COMPLETE"
            and gold.expected_terminal_reason == "SUCCESS"
        ),
        agent_steps=inference_count,
        tool_calls=len(trajectory.tool_records),
        search_calls=sum(
            record.trace.tool_name == "search_video" for record in trajectory.tool_records
        ),
        recovery_opportunity=opportunity,
        recovery_attempted=recovery_attempted,
        recovery_decision_validity=(valid_decisions / attempts if attempts else None),
        recovery_succeeded=recovery_succeeded,
        conditional_recovery_success=(recovery_succeeded if recovery_attempted else None),
        recovery_additional_steps=(
            len(later_steps) + len(later_failures) if recovery_attempted else None
        ),
        recovery_additional_latency_ms=(additional_latency if recovery_attempted else None),
        replan_recovery_structured_validity=(
            valid_decisions / (valid_decisions + failed_decisions)
            if valid_decisions + failed_decisions
            else None
        ),
        patch_acceptance_rate=(accepted / valid_decisions if valid_decisions else None),
        patch_rejection_reasons=rejection_reasons,
        first_error_type=first_error,
        primary_failure=primary,
        secondary_failures=secondary,
    )


def analyze_m4b_recovery_case(
    trajectory: M4BTrajectory,
    gold: M4BTaskGold,
    evaluation: M4BTrajectoryEvaluation,
) -> M4BRecoveryCaseRecord | None:
    """Persist evaluator-side recovery facts for every first observable error."""

    first_index, first_error = _first_observable_error(trajectory, gold)
    if first_index is None or first_error is None:
        return None
    decision_payload: dict[str, JsonValue] | None = None
    accepted: bool | None = None
    changed: list[str] = []
    if isinstance(trajectory, M4BAgentTrajectory) and trajectory.final_state.recovery_events:
        event = trajectory.final_state.recovery_events[0]
        decision_payload = cast(dict[str, JsonValue], event.decision.model_dump(mode="json"))
        accepted = event.accepted
        changed.extend(f"affected node {item}" for item in event.affected_node_ids)
        if event.generated_plan_revision is not None:
            changed.append(f"generated plan revision {event.generated_plan_revision}")
    else:
        step = next(
            (
                item
                for item in trajectory.policy_steps
                if item.operation == "replan" and isinstance(item.decision, ReplanDecision)
            ),
            None,
        )
        if step is not None and isinstance(step.decision, ReplanDecision):
            decision_payload = cast(dict[str, JsonValue], step.decision.model_dump(mode="json"))
            revisions = {
                envelope.event.revision
                for envelope in trajectory.events
                if isinstance(envelope.event, PlanPatch)
            }
            accepted = step.decision.requested_patch.revision in revisions
            changed.extend(f"affected node {item}" for item in step.decision.affected_plan_nodes)
    subsequent = trajectory.tool_records[first_index + 1 :]
    return M4BRecoveryCaseRecord(
        task_id=gold.task_id,
        run_id=trajectory.run_id,
        variant=evaluation.variant,
        first_error_type=first_error,
        recovery_possible=evaluation.recovery_opportunity,
        recovery_attempted=evaluation.recovery_attempted,
        recovery_decision=decision_payload,
        patch_accepted=accepted,
        changed_state=tuple(changed),
        subsequent_tool_improved_state=any(
            record.observation.status == "success" for record in subsequent
        ),
        eventual_task_success=evaluation.task_success,
    )


def summarize_m4b_evaluations(
    evaluations: Sequence[M4BTrajectoryEvaluation],
) -> M4BMetricsSummary:
    if not evaluations:
        raise ValueError("cannot summarize empty M4B evaluations")
    variants = {item.variant for item in evaluations}
    if len(variants) != 1:
        raise ValueError("M4B summary requires exactly one variant")
    total = len(evaluations)
    editing = [item for item in evaluations if item.category != "impossible"]
    impossible = [item for item in evaluations if item.category == "impossible"]
    opportunities = [item for item in evaluations if item.recovery_opportunity]
    attempted = [item for item in opportunities if item.recovery_attempted]
    recovery_validity = [
        item.recovery_decision_validity
        for item in evaluations
        if item.recovery_decision_validity is not None
    ]
    structured_recovery = [
        item.replan_recovery_structured_validity
        for item in evaluations
        if item.replan_recovery_structured_validity is not None
    ]
    patch_rates = [
        item.patch_acceptance_rate for item in evaluations if item.patch_acceptance_rate is not None
    ]
    failures: Counter[M4BFailureCause] = Counter(
        item.primary_failure for item in evaluations if item.primary_failure is not None
    )
    protocol: set[M4BFailureCause] = {
        "handoff_error",
        "recovery_patch_error",
        "verification_error",
        "system_error",
    }
    policy: set[M4BFailureCause] = {
        "planning_error",
        "wrong_tool",
        "invalid_arguments",
        "recovery_decision_error",
        "premature_success",
        "premature_refusal",
        "loop/stagnation",
        "budget_exhaustion",
        "model_format_error",
        "tool_execution_failure",
    }
    upstream: set[M4BFailureCause] = {"retrieval_error", "upstream_perception_error"}
    failed_total = max(sum(failures.values()), 1)
    origins = {
        "protocol_runtime": sum(failures[item] for item in protocol) / failed_total,
        "policy_model": sum(failures[item] for item in policy) / failed_total,
        "upstream": sum(failures[item] for item in upstream) / failed_total,
    }
    categories = sorted({item.category for item in evaluations})
    rejection_counts = Counter(
        reason for item in evaluations for reason in item.patch_rejection_reasons
    )
    return M4BMetricsSummary(
        variant=next(iter(variants)),
        task_count=total,
        task_success_rate=sum(item.task_success for item in evaluations) / total,
        editing_task_success_rate=(
            sum(item.editing_task_success for item in editing) / max(len(editing), 1)
        ),
        correct_impossible_refusal_rate=(
            sum(item.correct_impossible_refusal for item in impossible) / max(len(impossible), 1)
        ),
        correct_final_artifact_rate=(
            sum(item.correct_final_artifact for item in editing) / max(len(editing), 1)
        ),
        hard_constraint_satisfaction_rate=(
            sum(item.hard_constraints_satisfied for item in evaluations) / total
        ),
        structured_output_validity=statistics.fmean(
            item.structured_output_validity for item in evaluations
        ),
        model_output_failure_rate=sum(item.model_output_failure for item in evaluations) / total,
        invalid_tool_call_rate=statistics.fmean(
            item.invalid_tool_call_rate for item in evaluations
        ),
        loop_stagnation_rate=sum(item.loop_or_stagnation for item in evaluations) / total,
        repeated_editor_rate=statistics.fmean(item.repeated_editor_rate for item in evaluations),
        budget_exhaustion_rate=sum(item.budget_exhausted for item in evaluations) / total,
        premature_finish_rate=sum(item.premature_finish for item in evaluations) / total,
        premature_refusal_rate=sum(item.premature_refusal for item in evaluations) / total,
        average_agent_steps=statistics.fmean(item.agent_steps for item in evaluations),
        average_tool_calls=statistics.fmean(item.tool_calls for item in evaluations),
        average_search_calls=statistics.fmean(item.search_calls for item in evaluations),
        recovery_opportunity_count=len(opportunities),
        recovery_attempt_rate=(len(attempted) / len(opportunities) if opportunities else None),
        recovery_decision_validity=(
            statistics.fmean(recovery_validity) if recovery_validity else None
        ),
        recovery_success_rate=(
            sum(item.recovery_succeeded for item in opportunities) / len(opportunities)
            if opportunities
            else None
        ),
        conditional_recovery_success=(
            sum(item.recovery_succeeded for item in attempted) / len(attempted)
            if attempted
            else None
        ),
        median_additional_recovery_steps=(
            statistics.median(
                item.recovery_additional_steps
                for item in attempted
                if item.recovery_additional_steps is not None
            )
            if any(item.recovery_additional_steps is not None for item in attempted)
            else None
        ),
        median_additional_recovery_latency_ms=(
            statistics.median(
                item.recovery_additional_latency_ms
                for item in attempted
                if item.recovery_additional_latency_ms is not None
            )
            if any(item.recovery_additional_latency_ms is not None for item in attempted)
            else None
        ),
        replan_recovery_structured_validity=(
            statistics.fmean(structured_recovery) if structured_recovery else None
        ),
        patch_acceptance_rate=(statistics.fmean(patch_rates) if patch_rates else None),
        patch_rejection_counts=dict(rejection_counts),
        failure_counts=dict(failures),
        failure_origin_fraction=cast(
            dict[Literal["protocol_runtime", "policy_model", "upstream"], float],
            origins,
        ),
        success_by_category={
            category: sum(item.task_success for item in evaluations if item.category == category)
            / sum(item.category == category for item in evaluations)
            for category in categories
        },
    )
