"""Evaluator-private M4A task construction, scoring, and failure attribution."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.agent import AgentTrajectory, ToolDecision
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.media import TimeRange
from cutagent.schemas.task_input import (
    AspectRatioConstraint,
    DurationConstraint,
    ObservableConstraint,
    RequiredContentConstraint,
    TaskInput,
)
from cutagent_evaluation.m2b_dataset import HeldOutVideoGold

M4ATaskCategory = Literal[
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
M4AFailureCause = Literal[
    "planning_error",
    "retrieval_error",
    "upstream_perception_error",
    "wrong_tool",
    "invalid_arguments",
    "tool_execution_failure",
    "verification_error",
    "premature_success",
    "premature_cannot_complete",
    "loop/stagnation",
    "budget_exhaustion",
    "model_format_error",
    "system_error",
]


class M4ATaskGold(SchemaModel):
    """Private annotation; never accepted by AgentRuntime or policy interfaces."""

    gold_id: Identifier
    task_id: Identifier
    source_group_id: Identifier
    split: Literal["dev"] = "dev"
    category: M4ATaskCategory
    expected_terminal_reason: Literal["SUCCESS", "CANNOT_COMPLETE"]
    relevant_time_ranges: tuple[TimeRange, ...] = ()
    required_tools: tuple[Identifier, ...] = ()
    expected_duration_ms: int | None = Field(default=None, gt=0)
    duration_tolerance_ms: int = Field(default=250, ge=0)
    expected_width: int | None = Field(default=None, gt=0)
    expected_height: int | None = Field(default=None, gt=0)
    expected_subtitle_text: NonEmptyStr | None = None
    expected_subtitle_range: TimeRange | None = None
    evaluator_notes: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def dimensions_are_paired(self) -> M4ATaskGold:
        if (self.expected_width is None) != (self.expected_height is None):
            raise ValueError("expected dimensions must be supplied together")
        return self


class M4ADevelopmentCase(SchemaModel):
    task_input: TaskInput
    gold: M4ATaskGold

    @model_validator(mode="after")
    def identifiers_match(self) -> M4ADevelopmentCase:
        if self.task_input.task_id != self.gold.task_id:
            raise ValueError("public task and private Gold task IDs differ")
        return self


class M4ATrajectoryEvaluation(SchemaModel):
    task_id: Identifier
    run_id: Identifier
    baseline: NonEmptyStr
    policy_view_mode: NonEmptyStr
    category: M4ATaskCategory
    task_success: bool
    hard_constraints_satisfied: bool
    correct_final_artifact: bool
    tool_selection_accuracy: float = Field(ge=0, le=1)
    tool_argument_validity: float = Field(ge=0, le=1)
    invalid_tool_call_rate: float = Field(ge=0, le=1)
    agent_steps: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    search_calls: int = Field(ge=0)
    repeated_action_rate: float = Field(ge=0, le=1)
    budget_exhausted: bool
    loop_detected: bool
    correct_termination: bool
    structured_output_validity: float = Field(ge=0, le=1)
    repair_rate: float = Field(ge=0, le=1)
    recovery_attempted: bool
    recovery_succeeded: bool
    recovery_additional_steps: int | None = Field(default=None, ge=1)
    recovery_additional_tool_calls: int | None = Field(default=None, ge=1)
    recovery_additional_latency_ms: int | None = Field(default=None, ge=0)
    primary_failure: M4AFailureCause | None = None
    secondary_failures: tuple[M4AFailureCause, ...] = ()
    duration_error_ms: int | None = Field(default=None, ge=0)


class M4AMetricsSummary(SchemaModel):
    baseline: NonEmptyStr
    policy_view_mode: NonEmptyStr
    task_count: int = Field(gt=0)
    task_success_rate: float = Field(ge=0, le=1)
    hard_constraint_satisfaction_rate: float = Field(ge=0, le=1)
    correct_final_artifact_rate: float = Field(ge=0, le=1)
    mean_tool_selection_accuracy: float = Field(ge=0, le=1)
    mean_tool_argument_validity: float = Field(ge=0, le=1)
    invalid_tool_call_rate: float = Field(ge=0, le=1)
    average_agent_steps: float = Field(ge=0)
    average_tool_calls: float = Field(ge=0)
    average_search_calls: float = Field(ge=0)
    repeated_action_rate: float = Field(ge=0, le=1)
    budget_exhaustion_rate: float = Field(ge=0, le=1)
    loop_detection_rate: float = Field(ge=0, le=1)
    correct_termination_rate: float = Field(ge=0, le=1)
    structured_output_validity: float = Field(ge=0, le=1)
    repair_rate: float = Field(ge=0, le=1)
    recovery_attempted_rate: float = Field(ge=0, le=1)
    recovery_success_rate: float | None = Field(default=None, ge=0, le=1)
    median_additional_recovery_steps: float | None = Field(default=None, ge=0)
    median_additional_recovery_tool_calls: float | None = Field(default=None, ge=0)
    median_additional_recovery_latency_ms: float | None = Field(default=None, ge=0)
    failure_counts: dict[M4AFailureCause, int]
    success_by_category: dict[M4ATaskCategory, float]


def _scene_description(video: HeldOutVideoGold, scene_index: int) -> str:
    scene = video.scenes[scene_index]
    action = scene.action.replace("_", " ")
    return f"the {scene.primary_entity} is {action} near the {scene.companion_entity}"


def build_m4a_development_set(
    videos: Sequence[HeldOutVideoGold],
    source_refs: Mapping[str, ArtifactRef],
) -> tuple[M4ADevelopmentCase, ...]:
    """Create exactly 50 deterministic public/private development cases."""

    if len(videos) < 6:
        raise ValueError("M4A development set requires at least six multi-scene videos")
    cases: list[M4ADevelopmentCase] = []

    def add(
        *,
        category: M4ATaskCategory,
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
    ) -> None:
        video = videos[video_index % len(videos)]
        source = source_refs.get(video.source_group_id)
        if source is None:
            raise ValueError(f"missing public source ref for {video.source_group_id}")
        task_id = f"m4a-dev-{len(cases) + 1:03d}"
        cases.append(
            M4ADevelopmentCase(
                task_input=TaskInput(
                    task_id=task_id,
                    video_ref=source,
                    instruction=instruction,
                    user_constraints=constraints,
                ),
                gold=M4ATaskGold(
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
                    evaluator_notes=("M4A development diagnostic; not locked benchmark data.",),
                ),
            )
        )

    for index in range(6):
        video = videos[index]
        scene_index = 2
        scene = video.scenes[scene_index]
        description = _scene_description(video, scene_index)
        add(
            category="search_trim",
            video_index=index,
            instruction=f"Find the scene where {description}; export only that full scene.",
            ranges=(scene.nominal_time_range,),
            tools=("search_video", "trim_video", "validate_media"),
            duration_ms=3000,
            constraints=(RequiredContentConstraint(description=description),),
        )
        add(
            category="search_trim_validate",
            video_index=index,
            instruction=(
                f"Find the scene with visible marker {scene.ocr_text}, export the full scene, "
                "and validate the resulting media."
            ),
            ranges=(scene.nominal_time_range,),
            tools=("search_video", "trim_video", "validate_media"),
            duration_ms=3000,
        )
        left, right = video.scenes[0], video.scenes[2]
        add(
            category="retrieve_concat",
            video_index=index,
            instruction=(
                f"Create one clip with marker {left.ocr_text} first and marker "
                f"{right.ocr_text} second; include each complete scene and validate it."
            ),
            ranges=(left.nominal_time_range, right.nominal_time_range),
            tools=("search_video", "trim_video", "concat_videos", "validate_media"),
            duration_ms=6000,
            constraints=(DurationConstraint(min_ms=5700, max_ms=6300),),
        )
        subtitle = f"CUTAGENT {index + 1}"
        add(
            category="subtitle",
            video_index=index,
            instruction=(
                f"Find the quote '{scene.transcript}', export that scene, then burn subtitle "
                f"'{subtitle}' from 0 to 1500 ms and validate the output."
            ),
            ranges=(scene.nominal_time_range,),
            tools=("search_video", "trim_video", "add_subtitles", "validate_media"),
            duration_ms=3000,
            subtitle_text=subtitle,
            subtitle_range=TimeRange(start_ms=0, end_ms=1500),
        )
        speed = 2.0 if index % 2 == 0 else 0.5
        expected = round(3000 / speed)
        add(
            category="speed",
            video_index=index,
            instruction=(
                f"Find the scene with marker {scene.ocr_text}, export it, change speed to "
                f"{speed:g}x, and validate the output."
            ),
            ranges=(scene.nominal_time_range,),
            tools=("search_video", "trim_video", "change_speed", "validate_media"),
            duration_ms=expected,
            constraints=(DurationConstraint(min_ms=expected - 300, max_ms=expected + 300),),
        )
        second = video.scenes[1]
        add(
            category="reframe",
            video_index=index,
            instruction=(
                f"Find the full scene with marker {second.ocr_text}, export it, convert it to "
                "portrait 216x384 using crop, and validate the output."
            ),
            ranges=(second.nominal_time_range,),
            tools=("search_video", "trim_video", "reframe_video", "validate_media"),
            duration_ms=3000,
            constraints=(AspectRatioConstraint(width=9, height=16),),
            dimensions=(216, 384),
        )

    for index in range(5):
        video = videos[index]
        scene = video.scenes[2]
        add(
            category="multi_constraint",
            video_index=index,
            instruction=(
                f"Find the full scene where {_scene_description(video, 2)}, speed it to 2x, "
                "reframe to 216x384 crop, validate it, and return only the final artifact."
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
        )
    for index in range(5):
        video = videos[index]
        scene = video.scenes[2]
        add(
            category="hard_negative",
            video_index=index,
            instruction=(
                f"This video repeats the same object in several scenes. Find only the scene "
                f"where {_scene_description(video, 2)} and export that complete scene."
            ),
            ranges=(scene.nominal_time_range,),
            tools=("search_video", "trim_video", "validate_media"),
            duration_ms=3000,
            constraints=(RequiredContentConstraint(description=_scene_description(video, 2)),),
        )
    for index in range(4):
        add(
            category="impossible",
            video_index=index,
            instruction=(
                "Find a scene where a silver airplane lands in a lake, export exactly five "
                "seconds, and do not fabricate an answer if no evidence exists."
            ),
            ranges=(),
            tools=("search_video",),
            duration_ms=None,
            expected_terminal="CANNOT_COMPLETE",
            constraints=(
                DurationConstraint(min_ms=5000, max_ms=5000),
                RequiredContentConstraint(description="silver airplane lands in a lake"),
            ),
        )
    if len(cases) != 50:
        raise AssertionError(f"M4A task construction produced {len(cases)} cases")
    return tuple(cases)


def _time_iou(left: TimeRange, right: TimeRange) -> float:
    intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    return intersection / union if union else 0.0


def _observed_trim_ranges(trajectory: AgentTrajectory) -> tuple[TimeRange, ...]:
    ranges: list[TimeRange] = []
    for record in trajectory.tool_records:
        if record.trace.tool_name != "trim_video":
            continue
        raw = record.trace.normalized_arguments.get("time_range")
        if isinstance(raw, dict):
            ranges.append(TimeRange.model_validate(raw))
    return tuple(ranges)


def _concat_order_satisfied(trajectory: AgentTrajectory, gold: M4ATaskGold) -> bool:
    if gold.category != "retrieve_concat":
        return True
    range_by_output: dict[str, TimeRange] = {}
    for record in trajectory.tool_records:
        if record.trace.tool_name != "trim_video" or record.trace.output_artifact is None:
            continue
        raw = record.trace.normalized_arguments.get("time_range")
        if isinstance(raw, dict):
            range_by_output[record.trace.output_artifact.artifact_id] = TimeRange.model_validate(
                raw
            )
    concat = next(
        (
            record
            for record in trajectory.tool_records
            if record.trace.tool_name == "concat_videos" and record.trace.status == "success"
        ),
        None,
    )
    if concat is None:
        return False
    raw_inputs = concat.trace.normalized_arguments.get("input_artifact_ids")
    if not isinstance(raw_inputs, list) or len(raw_inputs) != len(gold.relevant_time_ranges):
        return False
    observed = [range_by_output.get(item) for item in raw_inputs if isinstance(item, str)]
    return len(observed) == len(gold.relevant_time_ranges) and all(
        actual is not None and _time_iou(expected, actual) >= 0.9
        for expected, actual in zip(gold.relevant_time_ranges, observed, strict=True)
    )


def _final_media_details(trajectory: AgentTrajectory) -> dict[str, object]:
    if trajectory.final_output_artifact is None:
        return {}
    for record in reversed(trajectory.tool_records):
        if record.trace.tool_name != "validate_media" or record.trace.status != "success":
            continue
        if any(
            item.artifact_id == trajectory.final_output_artifact.artifact_id
            for item in record.trace.parent_artifacts
        ):
            return dict(record.observation.details)
    return {}


def _subtitle_matches(record_arguments: Mapping[str, object], gold: M4ATaskGold) -> bool:
    raw_cues = record_arguments.get("cues")
    if not isinstance(raw_cues, list) or gold.expected_subtitle_text is None:
        return False
    expected_range = (
        gold.expected_subtitle_range.model_dump(mode="json")
        if gold.expected_subtitle_range is not None
        else None
    )
    return any(
        isinstance(cue, dict)
        and cue.get("text") == gold.expected_subtitle_text
        and (expected_range is None or cue.get("time_range") == expected_range)
        for cue in raw_cues
    )


def _failure_attribution(
    trajectory: AgentTrajectory,
    gold: M4ATaskGold,
    *,
    task_success: bool,
    tools_satisfied: bool,
    ranges_satisfied: bool,
) -> tuple[M4AFailureCause | None, tuple[M4AFailureCause, ...]]:
    if task_success:
        return None, ()
    terminal = trajectory.terminal_reason
    if terminal == "MODEL_OUTPUT_FAILURE":
        return "model_format_error", ()
    if terminal == "SYSTEM_ERROR":
        return "system_error", ()
    if terminal == "BUDGET_EXHAUSTED":
        return "budget_exhaustion", ()
    if terminal == "LOOP_DETECTED":
        return "loop/stagnation", ()
    if terminal == "CANNOT_COMPLETE" and gold.expected_terminal_reason == "SUCCESS":
        return "premature_cannot_complete", ()
    if terminal == "SUCCESS" and gold.expected_terminal_reason != "SUCCESS":
        return "premature_success", ()
    failed = [record for record in trajectory.tool_records if record.trace.status != "success"]
    if any(record.trace.error_category == "invalid_call" for record in failed):
        return "invalid_arguments", ()
    if failed:
        return "tool_execution_failure", ()
    if not tools_satisfied:
        return "wrong_tool", ()
    if not ranges_satisfied:
        return "retrieval_error", ()
    if trajectory.initial_plan is not None:
        return "planning_error", ()
    return "verification_error", ()


def _first_observable_mistake(
    trajectory: AgentTrajectory,
    gold: M4ATaskGold,
) -> int | None:
    """Locate the first evaluator-visible wrong attempt without feeding Gold to runtime."""

    for index, record in enumerate(trajectory.tool_records):
        if record.observation.status != "success":
            return index
        if record.trace.tool_name == "search_video" and gold.relevant_time_ranges:
            response = record.observation.details.get("response")
            candidates = response.get("candidates") if isinstance(response, dict) else None
            top_range = (
                candidates[0].get("time_range")
                if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict)
                else None
            )
            if not isinstance(top_range, dict):
                return index
            observed = TimeRange.model_validate(top_range)
            if not any(
                _time_iou(expected, observed) >= 0.9 for expected in gold.relevant_time_ranges
            ):
                return index
        if record.trace.tool_name == "trim_video" and gold.relevant_time_ranges:
            raw = record.trace.normalized_arguments.get("time_range")
            if not isinstance(raw, dict):
                return index
            observed = TimeRange.model_validate(raw)
            if not any(
                _time_iou(expected, observed) >= 0.9 for expected in gold.relevant_time_ranges
            ):
                return index
        editing_tools = {
            "trim_video",
            "concat_videos",
            "change_speed",
            "add_subtitles",
            "reframe_video",
            "normalize_audio",
        }
        if record.trace.tool_name in editing_tools and record.trace.tool_name not in set(
            gold.required_tools
        ):
            return index
    return None


def _recovery_tail_metrics(
    trajectory: AgentTrajectory,
    first_mistake: int,
) -> tuple[int, int]:
    """Count and time recorded policy work after the first wrong observation."""

    mistaken_call_id = trajectory.tool_records[first_mistake].observation.call_id
    decision_step = next(
        (
            step.step_index
            for step in trajectory.policy_steps
            if isinstance(step.decision, ToolDecision)
            and step.decision.tool_call.tool_call_id == mistaken_call_id
        ),
        -1,
    )
    later_steps = [step for step in trajectory.policy_steps if step.step_index > decision_step]
    later_failures = [
        failure for failure in trajectory.policy_failures if failure.step_index > decision_step
    ]
    policy_latency = sum(step.stats.latency_ms for step in later_steps) + sum(
        failure.latency_ms for failure in later_failures
    )
    tool_latency = sum(
        record.trace.latency_ms for record in trajectory.tool_records[first_mistake + 1 :]
    )
    return len(later_steps) + len(later_failures), policy_latency + tool_latency


def evaluate_trajectory(
    trajectory: AgentTrajectory,
    gold: M4ATaskGold,
) -> M4ATrajectoryEvaluation:
    if trajectory.task_input.task_id != gold.task_id:
        raise ValueError("trajectory and Gold task IDs differ")
    tool_names = [record.trace.tool_name for record in trajectory.tool_records]
    required = set(gold.required_tools)
    observed = set(tool_names)
    tools_satisfied = required <= observed
    tool_selection_accuracy = len(required & observed) / max(len(required | observed), 1)
    successful_args = sum(
        record.observation.status != "invalid" for record in trajectory.tool_records
    )
    argument_validity = successful_args / max(len(trajectory.tool_records), 1)
    invalid_rate = 1.0 - argument_validity
    observed_ranges = _observed_trim_ranges(trajectory)
    ranges_satisfied = all(
        any(_time_iou(expected, actual) >= 0.9 for actual in observed_ranges)
        for expected in gold.relevant_time_ranges
    )
    concat_order_ok = _concat_order_satisfied(trajectory, gold)
    details = _final_media_details(trajectory)
    duration = details.get("duration_ms")
    duration_error = (
        abs(duration - gold.expected_duration_ms)
        if isinstance(duration, int) and gold.expected_duration_ms is not None
        else None
    )
    duration_ok = gold.expected_duration_ms is None or (
        duration_error is not None and duration_error <= gold.duration_tolerance_ms
    )
    video = details.get("video")
    dimensions_ok = gold.expected_width is None or (
        isinstance(video, dict)
        and video.get("width") == gold.expected_width
        and video.get("height") == gold.expected_height
    )
    subtitle_ok = True
    if gold.expected_subtitle_text is not None:
        subtitle_ok = any(
            record.trace.tool_name == "add_subtitles"
            and _subtitle_matches(record.trace.normalized_arguments, gold)
            for record in trajectory.tool_records
        )
    correct_termination = trajectory.terminal_reason == gold.expected_terminal_reason
    if gold.expected_terminal_reason == "CANNOT_COMPLETE":
        hard_constraints = correct_termination
        correct_artifact = trajectory.final_output_artifact is None
    else:
        hard_constraints = (
            ranges_satisfied and concat_order_ok and duration_ok and dimensions_ok and subtitle_ok
        )
        correct_artifact = trajectory.final_output_artifact is not None and bool(details)
    task_success = correct_termination and tools_satisfied and hard_constraints and correct_artifact
    repeated = sum(
        item.diagnostic_type in {"identical_tool_call", "no_information_gain"}
        for item in trajectory.diagnostics
    )
    repaired_inferences = sum(
        item.stats.repair_count > 0 for item in trajectory.policy_steps
    ) + sum(item.repair_count > 0 for item in trajectory.policy_failures)
    inference_count = len(trajectory.policy_steps) + len(trajectory.policy_failures)
    first_mistake = _first_observable_mistake(trajectory, gold)
    recovery_attempted = (
        first_mistake is not None and len(trajectory.tool_records) > first_mistake + 1
    )
    recovery_succeeded = recovery_attempted and task_success
    additional_calls = None
    if recovery_attempted and first_mistake is not None:
        additional_calls = len(trajectory.tool_records) - first_mistake - 1
    recovery_tail = (
        _recovery_tail_metrics(trajectory, first_mistake)
        if recovery_attempted and first_mistake is not None
        else None
    )
    primary, secondary = _failure_attribution(
        trajectory,
        gold,
        task_success=task_success,
        tools_satisfied=tools_satisfied,
        ranges_satisfied=ranges_satisfied,
    )
    return M4ATrajectoryEvaluation(
        task_id=gold.task_id,
        run_id=trajectory.run_id,
        baseline=trajectory.baseline,
        policy_view_mode=trajectory.policy_view_mode,
        category=gold.category,
        task_success=task_success,
        hard_constraints_satisfied=hard_constraints,
        correct_final_artifact=correct_artifact,
        tool_selection_accuracy=tool_selection_accuracy,
        tool_argument_validity=argument_validity,
        invalid_tool_call_rate=invalid_rate,
        agent_steps=inference_count,
        tool_calls=len(trajectory.tool_records),
        search_calls=sum(name == "search_video" for name in tool_names),
        repeated_action_rate=repeated / max(len(trajectory.policy_steps), 1),
        budget_exhausted=trajectory.terminal_reason == "BUDGET_EXHAUSTED",
        loop_detected=trajectory.terminal_reason == "LOOP_DETECTED",
        correct_termination=correct_termination,
        structured_output_validity=len(trajectory.policy_steps) / max(inference_count, 1),
        repair_rate=repaired_inferences / max(inference_count, 1),
        recovery_attempted=recovery_attempted,
        recovery_succeeded=recovery_succeeded,
        recovery_additional_steps=recovery_tail[0] if recovery_tail is not None else None,
        recovery_additional_tool_calls=additional_calls,
        recovery_additional_latency_ms=recovery_tail[1] if recovery_tail is not None else None,
        primary_failure=primary,
        secondary_failures=secondary,
        duration_error_ms=duration_error,
    )


def summarize_evaluations(
    evaluations: Sequence[M4ATrajectoryEvaluation],
) -> M4AMetricsSummary:
    if not evaluations:
        raise ValueError("cannot summarize an empty M4A evaluation")
    identity = {(item.baseline, item.policy_view_mode) for item in evaluations}
    if len(identity) != 1:
        raise ValueError("M4A summary requires one baseline/view configuration")
    total = len(evaluations)
    attempted = [item for item in evaluations if item.recovery_attempted]
    artifact_tasks = [item for item in evaluations if item.category != "impossible"]
    failure_counts: Counter[M4AFailureCause] = Counter(
        item.primary_failure for item in evaluations if item.primary_failure is not None
    )
    categories = sorted({item.category for item in evaluations})
    success_by_category = {
        category: sum(item.task_success for item in evaluations if item.category == category)
        / sum(item.category == category for item in evaluations)
        for category in categories
    }
    baseline, view = next(iter(identity))
    return M4AMetricsSummary(
        baseline=baseline,
        policy_view_mode=view,
        task_count=total,
        task_success_rate=sum(item.task_success for item in evaluations) / total,
        hard_constraint_satisfaction_rate=(
            sum(item.hard_constraints_satisfied for item in evaluations) / total
        ),
        correct_final_artifact_rate=(
            sum(item.correct_final_artifact for item in artifact_tasks)
            / max(len(artifact_tasks), 1)
        ),
        mean_tool_selection_accuracy=statistics.fmean(
            item.tool_selection_accuracy for item in evaluations
        ),
        mean_tool_argument_validity=statistics.fmean(
            item.tool_argument_validity for item in evaluations
        ),
        invalid_tool_call_rate=statistics.fmean(
            item.invalid_tool_call_rate for item in evaluations
        ),
        average_agent_steps=statistics.fmean(item.agent_steps for item in evaluations),
        average_tool_calls=statistics.fmean(item.tool_calls for item in evaluations),
        average_search_calls=statistics.fmean(item.search_calls for item in evaluations),
        repeated_action_rate=statistics.fmean(item.repeated_action_rate for item in evaluations),
        budget_exhaustion_rate=sum(item.budget_exhausted for item in evaluations) / total,
        loop_detection_rate=sum(item.loop_detected for item in evaluations) / total,
        correct_termination_rate=sum(item.correct_termination for item in evaluations) / total,
        structured_output_validity=statistics.fmean(
            item.structured_output_validity for item in evaluations
        ),
        repair_rate=statistics.fmean(item.repair_rate for item in evaluations),
        recovery_attempted_rate=len(attempted) / total,
        recovery_success_rate=(
            sum(item.recovery_succeeded for item in attempted) / len(attempted)
            if attempted
            else None
        ),
        median_additional_recovery_steps=(
            float(
                statistics.median(
                    item.recovery_additional_steps
                    for item in attempted
                    if item.recovery_additional_steps is not None
                )
            )
            if attempted
            else None
        ),
        median_additional_recovery_tool_calls=(
            float(
                statistics.median(
                    item.recovery_additional_tool_calls
                    for item in attempted
                    if item.recovery_additional_tool_calls is not None
                )
            )
            if attempted
            else None
        ),
        median_additional_recovery_latency_ms=(
            float(
                statistics.median(
                    item.recovery_additional_latency_ms
                    for item in attempted
                    if item.recovery_additional_latency_ms is not None
                )
            )
            if attempted
            else None
        ),
        failure_counts=dict(failure_counts),
        success_by_category=success_by_category,
    )
