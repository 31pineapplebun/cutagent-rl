"""Offline-only small development regression; never imported by the online entry."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from cutagent.agent.m4b_state_reducer import M4BStateReducer
from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.agent import ToolDecision
from cutagent.schemas.base import SchemaModel
from cutagent.schemas.m4b_agent import M4BAgentTrajectory
from cutagent.schemas.media import TimeRange
from cutagent.schemas.tools import ToolExecutionContext
from cutagent.tools.artifacts import ArtifactStore
from cutagent.tools.errors import ToolFailure
from cutagent.tools.executor import FFmpegExecutor
from cutagent.tools.validation import MediaValidator

DATASET_VERSION = "harness-development-regression-v1"
SCORER_VERSION = "harness-independent-media-and-source-interval-v1"
SOURCE_GROUP = "m2b-heldout-source-01"
ILLEGAL_ERRORS = frozenset(
    {
        "invalid_call",
        "unknown_tool",
        "capability_denied",
        "artifact_not_allowed",
        "artifact_not_found",
        "filesystem_violation",
        "invalid_interval",
        "invalid_subtitle",
    }
)


class HarnessCase(SchemaModel):
    task_id: str
    category: Literal["normal", "injected", "impossible"]
    instruction: str
    duration_ms: int = Field(gt=0)
    expected_interval: TimeRange | None = None
    requires_retrieval: bool = False


def development_cases() -> tuple[HarnessCase, ...]:
    """Predeclared development reuse: one CC0 generated source, not generalization."""
    cases: list[HarnessCase] = []
    intervals = (
        (0, 1500),
        (500, 2000),
        (1500, 3000),
        (3000, 4500),
        (4500, 6000),
        (6000, 7500),
        (7500, 9000),
        (9000, 10500),
    )
    for index, (start, end) in enumerate(intervals, 1):
        cases.append(
            HarnessCase(
                task_id=f"harness-task-{index:02d}",
                category="normal",
                duration_ms=end - start,
                instruction=f"Trim the original input from {start} ms to {end} ms, validate "
                f"and return the {end - start} ms clip. Only trim and validate are needed.",
                expected_interval=TimeRange(start_ms=start, end_ms=end),
            )
        )
    for scene in range(1, 5):
        cases.append(
            HarnessCase(
                task_id=f"harness-task-{8 + scene:02d}",
                category="normal",
                duration_ms=3000,
                instruction="Use search_video to locate the scene with the visible marker "
                f"M2B01 S{scene}. "
                "Trim that matching scene from the original input, validate and return "
                "a 3000 ms clip. Ground the interval in retrieved evidence.",
                expected_interval=TimeRange(start_ms=(scene - 1) * 3000, end_ms=scene * 3000),
                requires_retrieval=True,
            )
        )
    for index, (start, end) in enumerate(intervals[:6], 1):
        cases.append(
            HarnessCase(
                task_id=f"harness-task-{12 + index:02d}",
                category="injected",
                duration_ms=end - start,
                instruction=f"Trim the original input from {start} ms to {end} ms, validate "
                f"and return the {end - start} ms clip. Only trim and validate are needed.",
                expected_interval=TimeRange(start_ms=start, end_ms=end),
            )
        )
    for index, start in enumerate((20000, 30000), 1):
        cases.append(
            HarnessCase(
                task_id=f"harness-task-{18 + index:02d}",
                category="impossible",
                duration_ms=1500,
                instruction=f"Inspect the original source and return exactly its interval "
                f"{start}-{start + 1500} ms. Do not pad, loop, invent frames or substitute "
                "a different interval. If that interval does not exist, explain that "
                "the task cannot be completed.",
            )
        )
    return tuple(cases)


def score_run(
    case: HarnessCase, trajectory: M4BAgentTrajectory, store: ArtifactStore, root: Path
) -> dict[str, Any]:
    """Same offline acceptance for both variants, independent of claimed validation.

    A successful edit needs SUCCESS, an authentic generated artifact, full decode,
    duration within 250 ms, and a direct original-source trim within 125 ms of
    the annotated/requested boundaries. Semantic cases also need actual retrieval.
    Refusals never enter edit-success numerators.
    """
    final = trajectory.final_output_artifact
    media_ok = lineage_ok = retrieval_used = False
    duration: int | None = None
    scoring_error: str | None = None
    if final is not None:
        try:
            reference, path = store.get(final.artifact_id)
            actual = ArtifactRef.from_path(
                path, artifact_id=reference.artifact_id, media_type=reference.media_type
            )
            if actual.sha256 != final.sha256 or actual.size_bytes != final.size_bytes:
                raise ValueError("output content differs from the recorded artifact")
            validator = MediaValidator(
                artifact_store=ArtifactStore(root / "offline_validation"),
                executor=FFmpegExecutor("ffmpeg"),
            )
            validated = validator.probe(
                reference,
                path,
                decode_entire_video=True,
                context=ToolExecutionContext(
                    execution_id="offline-score",
                    allowed_output_root_id="offline-score",
                    allowed_artifact_ids=(reference.artifact_id,),
                    allowed_capabilities=("media.inspect", "media.decode"),
                ),
            )
            duration = validated.asset.duration_ms
            media_ok = (
                all(c.passed for c in validated.checks) and abs(duration - case.duration_ms) <= 250
            )
        except (ValueError, OSError, RuntimeError, ToolFailure) as error:
            scoring_error = f"{type(error).__name__}: {error}"
        retrieval_seen = False
        for record in trajectory.tool_records:
            trace = record.trace
            if trace.status == "success" and trace.tool_name == "search_video":
                response = record.observation.details.get("response")
                retrieval_seen |= isinstance(response, dict) and bool(response.get("candidates"))
            if (
                trace.tool_name == "trim_video"
                and trace.status == "success"
                and trace.output_artifact is not None
                and trace.output_artifact.sha256 == final.sha256
                and trace.output_artifact.artifact_id == final.artifact_id
                and len(trace.parent_artifacts) == 1
                and trace.parent_artifacts[0].sha256 == trajectory.task_input.video_ref.sha256
                and case.expected_interval is not None
            ):
                interval = TimeRange.model_validate(trace.normalized_arguments["time_range"])
                lineage_ok = (
                    abs(interval.start_ms - case.expected_interval.start_ms) <= 125
                    and abs(interval.end_ms - case.expected_interval.end_ms) <= 125
                )
                retrieval_used = retrieval_seen
    replay = (
        M4BStateReducer.replay(trajectory.initial_state, trajectory.events)
        == trajectory.final_state
    )
    success = (
        case.category != "impossible"
        and trajectory.terminal_reason == "SUCCESS"
        and media_ok
        and lineage_ok
        and (retrieval_used or not case.requires_retrieval)
        and replay
    )
    illegal = sum(
        record.trace.error_category in ILLEGAL_ERRORS for record in trajectory.tool_records
    )
    return {
        "task_id": case.task_id,
        "category": case.category,
        "requires_retrieval": case.requires_retrieval,
        "terminal_reason": trajectory.terminal_reason,
        "editing_success": success,
        "correct_refusal": case.category == "impossible"
        and trajectory.terminal_reason == "CANNOT_COMPLETE"
        and final is None,
        "media_valid_and_duration_matched": media_ok,
        "duration_ms": duration,
        "source_interval_matched": lineage_ok,
        "retrieval_used": retrieval_used,
        "state_replay_identical": replay,
        "scoring_error": scoring_error,
        "illegal_tool_calls": illegal,
        "tool_call_attempts": len(trajectory.tool_records),
        "policy_tool_proposals": sum(
            isinstance(s.decision, ToolDecision) for s in trajectory.policy_steps
        ),
        "unparsed_policy_attempts": sum(f.attempt_count for f in trajectory.policy_failures),
        "elapsed_ms": trajectory.latency.total_ms,
        "measured_valid_step_input_tokens": sum(
            s.stats.input_tokens for s in trajectory.policy_steps
        ),
        "measured_valid_step_output_tokens": sum(
            s.stats.output_tokens for s in trajectory.policy_steps
        ),
        "token_accounting_complete": not trajectory.policy_failures,
    }
