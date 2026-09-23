"""Handcrafted objective-evaluator calibration and human-review packet support."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import Field, JsonValue

from cutagent.schemas.base import Identifier, SchemaModel
from cutagent.schemas.event import TerminalReasonCode
from cutagent.schemas.media import TimeRange
from cutagent.schemas.tools import ToolName
from cutagent_evaluation.m5a_evaluator import (
    RetrievedCandidateFact,
    SearchFact,
    ToolFact,
    TrajectoryFacts,
    evaluate_facts,
)
from cutagent_evaluation.m5a_schemas import (
    CutAgentBenchCase,
    DurationGoldConstraint,
    ExpectedTerminalBehavior,
    ResolutionGoldConstraint,
    SceneOrderGoldConstraint,
    SpeedGoldConstraint,
    SubtitleGoldConstraint,
)
from cutagent_evaluation.schemas import DatasetSplit

CalibrationMutation = Literal[
    "exact_success",
    "wrong_scene",
    "wrong_duration",
    "wrong_aspect_ratio",
    "wrong_subtitle",
    "wrong_order",
    "partial_constraints",
    "correct_refusal",
    "false_refusal",
    "premature_finish",
    "loop",
    "invalid_arguments",
    "recovery_success",
    "recovery_failure",
]


class CalibrationCaseResult(SchemaModel):
    case_id: Identifier
    task_id: Identifier
    mutation: CalibrationMutation
    expected_task_success: bool
    observed_task_success: bool
    agreement: bool
    observed_constraint_satisfaction: float = Field(ge=0, le=1)
    observed_primary_failure: str | None = None


class EvaluatorCalibrationSummary(SchemaModel):
    calibration_version: Literal["m5a-handcrafted-calibration-v1"] = (
        "m5a-handcrafted-calibration-v1"
    )
    case_count: int = Field(ge=40, le=60)
    exact_tsr_agreement: float = Field(ge=0, le=1)
    mutation_counts: dict[str, int]
    results: tuple[CalibrationCaseResult, ...]
    human_rating_status: Literal["pending_human_review"] = "pending_human_review"
    human_rater_count: Literal[0] = 0
    inter_rater_agreement: None = None


def _constraint(case: CutAgentBenchCase, constraint_type: type[object]) -> object | None:
    return next(
        (
            item
            for item in case.private_gold.objective_constraints
            if isinstance(item, constraint_type)
        ),
        None,
    )


def _mutation_for(case: CutAgentBenchCase, index: int) -> CalibrationMutation:
    gold = case.private_gold
    if gold.expected_terminal_behavior == ExpectedTerminalBehavior.CANNOT_COMPLETE:
        return "correct_refusal" if index % 2 == 0 else "premature_finish"
    if gold.failure_injection is not None:
        return "recovery_success" if index % 2 == 0 else "recovery_failure"
    choices: list[CalibrationMutation] = [
        "exact_success",
        "wrong_scene",
        "partial_constraints",
        "false_refusal",
        "loop",
        "invalid_arguments",
    ]
    if _constraint(case, DurationGoldConstraint) is not None:
        choices.append("wrong_duration")
    if _constraint(case, ResolutionGoldConstraint) is not None:
        choices.append("wrong_aspect_ratio")
    if _constraint(case, SubtitleGoldConstraint) is not None:
        choices.append("wrong_subtitle")
    if _constraint(case, SceneOrderGoldConstraint) is not None:
        choices.append("wrong_order")
    return choices[index % len(choices)]


def _final_details(case: CutAgentBenchCase) -> dict[str, JsonValue]:
    duration_constraint = _constraint(case, DurationGoldConstraint)
    resolution_constraint = _constraint(case, ResolutionGoldConstraint)
    duration = (
        duration_constraint.target_ms
        if isinstance(duration_constraint, DurationGoldConstraint)
        else 3000
    )
    width = (
        resolution_constraint.width
        if isinstance(resolution_constraint, ResolutionGoldConstraint)
        else 384
    )
    height = (
        resolution_constraint.height
        if isinstance(resolution_constraint, ResolutionGoldConstraint)
        else 256
    )
    return {
        "duration_ms": duration,
        "video": {"width": width, "height": height},
        "audio": [{"codec": "aac"}],
        "fully_decoded": True,
    }


def _successful_tools(case: CutAgentBenchCase) -> tuple[ToolFact, ...]:
    gold = case.private_gold
    sequence = (
        gold.acceptable_tool_sequences[0] if gold.acceptable_tool_sequences else gold.required_tools
    )
    tools: list[ToolFact] = []
    trim_outputs: list[str] = []
    trim_index = 0
    for index, tool_name in enumerate(sequence):
        arguments: dict[str, JsonValue] = {}
        output_id: str | None = None
        details: dict[str, JsonValue] = {}
        if tool_name == "search_video":
            arguments = {"query": case.public_task.instruction, "top_k": 5}
        elif tool_name == "trim_video":
            time_range = gold.acceptable_time_ranges[
                min(trim_index, len(gold.acceptable_time_ranges) - 1)
            ]
            arguments = {
                "input_artifact_id": case.public_task.video_ref.artifact_id,
                "time_range": time_range.model_dump(mode="json"),
            }
            output_id = f"calibration-trim-{index}"
            trim_outputs.append(output_id)
            trim_index += 1
        elif tool_name == "concat_videos":
            arguments = {"input_artifact_ids": list(trim_outputs)}
            output_id = f"calibration-concat-{index}"
        elif tool_name == "change_speed":
            speed = _constraint(case, SpeedGoldConstraint)
            arguments = {
                "input_artifact_id": trim_outputs[-1] if trim_outputs else "working-artifact",
                "speed_factor": speed.speed_factor
                if isinstance(speed, SpeedGoldConstraint)
                else 2.0,
            }
            output_id = f"calibration-speed-{index}"
        elif tool_name == "reframe_video":
            resolution = _constraint(case, ResolutionGoldConstraint)
            arguments = {
                "input_artifact_id": "working-artifact",
                "width": resolution.width
                if isinstance(resolution, ResolutionGoldConstraint)
                else 216,
                "height": resolution.height
                if isinstance(resolution, ResolutionGoldConstraint)
                else 384,
                "fit": "crop",
            }
            output_id = f"calibration-reframe-{index}"
        elif tool_name == "add_subtitles":
            subtitle = _constraint(case, SubtitleGoldConstraint)
            cue_range = (
                subtitle.time_range
                if isinstance(subtitle, SubtitleGoldConstraint)
                else TimeRange(start_ms=0, end_ms=1500)
            )
            cue_text = subtitle.text if isinstance(subtitle, SubtitleGoldConstraint) else "subtitle"
            arguments = {
                "input_artifact_id": "working-artifact",
                "cues": [
                    {
                        "cue_id": "calibration-cue",
                        "time_range": cue_range.model_dump(mode="json"),
                        "text": cue_text,
                    }
                ],
            }
            output_id = f"calibration-subtitle-{index}"
        elif tool_name == "validate_media":
            arguments = {"input_artifact_id": "calibration-final"}
            details = _final_details(case)
        tools.append(
            ToolFact(
                call_id=f"calibration-call-{index}",
                tool_name=tool_name,
                status="success",
                normalized_arguments=arguments,
                output_artifact_id=output_id,
                details=details,
            )
        )
    return tuple(tools)


def _search_facts(case: CutAgentBenchCase) -> tuple[SearchFact, ...]:
    gold = case.private_gold
    searches = (
        sum(tool == "search_video" for tool in gold.acceptable_tool_sequences[0])
        if gold.acceptable_tool_sequences
        else 0
    )
    if not gold.relevant_scene_ids:
        return tuple(
            SearchFact(call_id=f"calibration-search-{index}", candidates=())
            for index in range(max(1, searches))
        )
    return tuple(
        SearchFact(
            call_id=f"calibration-search-{index}",
            candidates=(
                RetrievedCandidateFact(
                    video_id=gold.relevant_video_ids[0],
                    scene_id=gold.relevant_scene_ids[min(index, len(gold.relevant_scene_ids) - 1)],
                    time_range=gold.acceptable_time_ranges[
                        min(index, len(gold.acceptable_time_ranges) - 1)
                    ],
                    rank=1,
                ),
            ),
        )
        for index in range(max(1, searches))
    )


def handcrafted_facts(
    case: CutAgentBenchCase,
    mutation: CalibrationMutation,
) -> TrajectoryFacts:
    """Build a known-outcome fact record without pretending it is an Agent run."""

    gold = case.private_gold
    tools: tuple[ToolFact, ...]
    terminal: TerminalReasonCode
    if gold.expected_terminal_behavior == ExpectedTerminalBehavior.CANNOT_COMPLETE:
        terminal = "CANNOT_COMPLETE"
        tools = (
            ToolFact(
                call_id="calibration-search-0",
                tool_name="search_video",
                status="success",
                normalized_arguments={"query": case.public_task.instruction, "top_k": 5},
            ),
        )
        final_id = None
        details: dict[str, JsonValue] = {}
    else:
        terminal = "SUCCESS"
        tools = _successful_tools(case)
        final_id = "calibration-final"
        details = _final_details(case)
    searches = _search_facts(case)
    recovery_decisions = 0
    accepted_recoveries = 0
    loop = False
    repeated = 0
    if mutation in {"wrong_scene", "partial_constraints"} and gold.acceptable_time_ranges:
        expected_starts = {item.start_ms for item in gold.acceptable_time_ranges}
        wrong_start = next(
            start for start in range(0, 15_000, 3000) if start not in expected_starts
        )
        wrong = TimeRange(start_ms=wrong_start, end_ms=wrong_start + 3000)
        tools = tuple(
            item.model_copy(
                update={
                    "normalized_arguments": {
                        **item.normalized_arguments,
                        "time_range": wrong.model_dump(mode="json"),
                    }
                }
            )
            if item.tool_name == "trim_video"
            else item
            for item in tools
        )
        searches = ()
    elif mutation == "wrong_duration":
        details = {**details, "duration_ms": 99_000}
    elif mutation == "wrong_aspect_ratio":
        details = {**details, "video": {"width": 640, "height": 360}}
    elif mutation == "wrong_subtitle":
        tools = tuple(
            item.model_copy(
                update={
                    "normalized_arguments": {
                        **item.normalized_arguments,
                        "cues": [
                            {
                                "cue_id": "wrong-cue",
                                "time_range": {"start_ms": 0, "end_ms": 1000},
                                "text": "WRONG",
                            }
                        ],
                    }
                }
            )
            if item.tool_name == "add_subtitles"
            else item
            for item in tools
        )
    elif mutation == "wrong_order":

        def reverse_inputs(item: ToolFact) -> ToolFact:
            raw_inputs = item.normalized_arguments.get("input_artifact_ids")
            inputs = list(reversed(raw_inputs)) if isinstance(raw_inputs, list) else []
            return item.model_copy(
                update={
                    "normalized_arguments": {
                        **item.normalized_arguments,
                        "input_artifact_ids": inputs,
                    }
                }
            )

        tools = tuple(
            reverse_inputs(item) if item.tool_name == "concat_videos" else item for item in tools
        )
    elif mutation == "false_refusal":
        terminal, final_id, details = "CANNOT_COMPLETE", None, {}
    elif mutation == "premature_finish":
        terminal, final_id, details = (
            "SUCCESS",
            "calibration-final",
            {
                "duration_ms": 3000,
                "video": {"width": 384, "height": 256},
            },
        )
    elif mutation == "loop":
        terminal, final_id, details = "LOOP_DETECTED", None, {}
        loop, repeated = True, 3
    elif mutation == "invalid_arguments":
        if tools:
            tools = (
                tools[0].model_copy(update={"status": "invalid"}),
                *tools[1:],
            )
        terminal, final_id, details = "CANNOT_COMPLETE", None, {}
    elif mutation in {"recovery_success", "recovery_failure"}:
        failed_tool_name: ToolName = (
            gold.failure_injection.inject_on_tool
            if gold.failure_injection is not None
            else "trim_video"
        )
        failed = ToolFact(
            call_id="calibration-injected-failure",
            tool_name=failed_tool_name,
            status=(
                gold.failure_injection.expected_online_status
                if gold.failure_injection is not None
                else "error"
            ),
            normalized_arguments={"input_artifact_id": case.public_task.video_ref.artifact_id},
        )
        tools = (failed, *tools)
        recovery_decisions = 1
        accepted_recoveries = 1
        if mutation == "recovery_failure":
            terminal, final_id, details = "BUDGET_EXHAUSTED", None, {}
    return TrajectoryFacts(
        task_id=case.public_task.task_id,
        run_id=f"calibration-{case.public_task.task_id}",
        terminal_reason=terminal,
        source_artifact_id=case.public_task.video_ref.artifact_id,
        final_artifact_id=final_id,
        final_media_details=details,
        searches=searches,
        tools=tools,
        policy_decisions=max(1, len(tools) + 1),
        policy_failures=0,
        policy_repairs=0,
        repeated_action_count=repeated,
        loop_detected=loop,
        recovery_decision_count=recovery_decisions,
        accepted_recovery_count=accepted_recoveries,
        latency_ms=100,
    )


def run_handcrafted_calibration(
    cases: tuple[CutAgentBenchCase, ...],
) -> EvaluatorCalibrationSummary:
    """Calibrate on 50 dev/validation cases; never access protected split Gold."""

    eligible = tuple(
        case
        for case in cases
        if case.private_gold.split in {DatasetSplit.DEV, DatasetSplit.VALIDATION}
    )[:50]
    if len(eligible) != 50:
        raise ValueError("M5A calibration requires exactly 50 dev/validation cases")
    results: list[CalibrationCaseResult] = []
    for index, case in enumerate(eligible):
        mutation = _mutation_for(case, index)
        facts = handcrafted_facts(case, mutation)
        evaluation = evaluate_facts(facts, case.private_gold)
        expected = mutation in {"exact_success", "correct_refusal", "recovery_success"}
        results.append(
            CalibrationCaseResult(
                case_id=f"calibration-case-{index + 1:02d}",
                task_id=case.public_task.task_id,
                mutation=mutation,
                expected_task_success=expected,
                observed_task_success=evaluation.task_success,
                agreement=expected == evaluation.task_success,
                observed_constraint_satisfaction=evaluation.hard_constraint_satisfaction,
                observed_primary_failure=(
                    evaluation.primary_failure.value
                    if evaluation.primary_failure is not None
                    else None
                ),
            )
        )
    agreement = sum(item.agreement for item in results) / len(results)
    return EvaluatorCalibrationSummary(
        case_count=len(results),
        exact_tsr_agreement=agreement,
        mutation_counts=dict(Counter(item.mutation for item in results)),
        results=tuple(results),
    )
