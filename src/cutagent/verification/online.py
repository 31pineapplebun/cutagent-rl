"""Observable, evidence-aware online checks for M4A Agent execution."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar, cast

from cutagent.schemas.agent import FinishDecision
from cutagent.schemas.event import (
    PlanNode,
    RuntimeCheckResult,
    VerificationResult,
    VerificationStatus,
)
from cutagent.schemas.state import AgentState
from cutagent.schemas.task_input import (
    AspectRatioConstraint,
    DurationConstraint,
    ForbiddenContentConstraint,
    RequiredContentConstraint,
)
from cutagent.schemas.tools import ToolExecutionRecord


def _check(name: str, passed: bool, summary: str, *, conclusive: bool = True) -> RuntimeCheckResult:
    return RuntimeCheckResult(
        check_name=name,
        passed=passed,
        summary=summary,
        conclusive=conclusive,
    )


def _trace_outputs(records: Iterable[ToolExecutionRecord]) -> dict[str, ToolExecutionRecord]:
    return {
        record.trace.output_artifact.artifact_id: record
        for record in records
        if record.trace.status == "success" and record.trace.output_artifact is not None
    }


def _ancestor_tools(output_artifact_id: str, records: tuple[ToolExecutionRecord, ...]) -> set[str]:
    outputs = _trace_outputs(records)
    visited: set[str] = set()
    tools: set[str] = set()
    pending = [output_artifact_id]
    while pending:
        artifact_id = pending.pop()
        if artifact_id in visited:
            continue
        visited.add(artifact_id)
        record = outputs.get(artifact_id)
        if record is None:
            continue
        tools.add(record.trace.tool_name)
        pending.extend(item.artifact_id for item in record.trace.parent_artifacts)
    return tools


def _known_evidence_ids(state: AgentState) -> set[str]:
    known = {state.task_input.video_ref.artifact_id}
    for observation in state.tool_observations:
        known.update(item.artifact_id for item in observation.artifacts)
        response = observation.details.get("response")
        if not isinstance(response, dict):
            continue
        candidates = response.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            for key in ("scene_id", "video_id"):
                value = candidate.get(key)
                if isinstance(value, str):
                    known.add(value)
            evidence = candidate.get("evidence_refs")
            if isinstance(evidence, list):
                for reference in evidence:
                    if isinstance(reference, dict) and isinstance(
                        reference.get("artifact_id"), str
                    ):
                        known.add(cast(str, reference["artifact_id"]))
    return known


class OnlineVerifier:
    """Use only task input, public state, and authentic ToolRegistry records."""

    version = "m4a-online-verifier-v1"

    _tool_capabilities: ClassVar[dict[str, set[str]]] = {
        "search_video": {"retrieval.read"},
        "inspect_media": {"media.inspect"},
        "validate_media": {"media.inspect", "media.decode"},
        "trim_video": {"media.inspect", "media.decode", "media.write"},
        "concat_videos": {"media.inspect", "media.decode", "media.write"},
        "change_speed": {"media.inspect", "media.decode", "media.write"},
        "add_subtitles": {"media.inspect", "media.decode", "media.write"},
        "reframe_video": {"media.inspect", "media.decode", "media.write"},
        "normalize_audio": {
            "media.inspect",
            "media.decode",
            "media.write",
            "audio.write",
        },
    }

    def verify_tool(
        self,
        state: AgentState,
        record: ToolExecutionRecord,
        *,
        current_node: PlanNode | None,
    ) -> VerificationResult:
        del state
        checks: list[RuntimeCheckResult] = [
            _check(
                "tool_status",
                record.observation.status == "success",
                (
                    "tool returned a successful structured observation"
                    if record.observation.status == "success"
                    else f"tool failed with {record.observation.error_code or 'unknown_error'}"
                ),
            )
        ]
        checks.extend(
            _check(
                f"post_{item.check_name}",
                item.passed,
                f"post-execution check {item.check_name}={'passed' if item.passed else 'failed'}",
            )
            for item in record.trace.validation_results
        )
        if record.trace.status == "success" and record.trace.tool_name == "search_video":
            response = record.observation.details.get("response")
            candidates: Any = response.get("candidates") if isinstance(response, dict) else None
            count = len(candidates) if isinstance(candidates, list) else 0
            checks.append(
                _check(
                    "retrieval_nonempty",
                    count > 0,
                    f"observable retrieval candidate count is {count}",
                )
            )
        if record.trace.status == "success" and record.trace.tool_name in {
            "trim_video",
            "concat_videos",
            "change_speed",
            "add_subtitles",
            "reframe_video",
            "normalize_audio",
        }:
            checks.append(
                _check(
                    "output_artifact_created",
                    record.trace.output_artifact is not None,
                    "editing tool produced a content-addressed output artifact",
                )
            )
        if current_node is not None and current_node.preferred_capability is not None:
            capabilities = self._tool_capabilities.get(record.trace.tool_name, set())
            checks.append(
                _check(
                    "preferred_capability_used",
                    current_node.preferred_capability in capabilities,
                    (
                        "tool belongs to the current plan node preferred capability"
                        if current_node.preferred_capability in capabilities
                        else "tool does not match the current plan node preferred capability"
                    ),
                )
            )
        failed = [item for item in checks if item.conclusive and not item.passed]
        inconclusive = [item for item in checks if not item.conclusive]
        status: VerificationStatus = (
            "failed" if failed else "inconclusive" if inconclusive else "passed"
        )
        return VerificationResult(
            verification_id=f"verify-{record.observation.call_id}",
            status=status,
            checks=tuple(checks),
            failure_types=(record.observation.error_code,)
            if failed and record.observation.error_code
            else (),
        )

    def verify_completion(
        self,
        state: AgentState,
        decision: FinishDecision,
        records: tuple[ToolExecutionRecord, ...],
    ) -> VerificationResult:
        outputs = _trace_outputs(records)
        output = outputs.get(decision.output_artifact_id)
        checks: list[RuntimeCheckResult] = [
            _check(
                "output_exists",
                output is not None,
                "finish output references a real ToolRegistry artifact",
            )
        ]
        validated = next(
            (
                record
                for record in reversed(records)
                if record.trace.tool_name == "validate_media"
                and record.trace.status == "success"
                and any(
                    item.artifact_id == decision.output_artifact_id
                    for item in record.trace.parent_artifacts
                )
            ),
            None,
        )
        checks.append(
            _check(
                "output_validated",
                validated is not None,
                "final artifact has an independent successful validate_media record",
            )
        )
        details = validated.observation.details if validated is not None else {}
        duration = details.get("duration_ms")
        video = details.get("video")
        width = video.get("width") if isinstance(video, dict) else details.get("width")
        height = video.get("height") if isinstance(video, dict) else details.get("height")
        for index, constraint in enumerate(state.task_input.user_constraints):
            if isinstance(constraint, DurationConstraint):
                passed = (
                    isinstance(duration, int) and constraint.min_ms <= duration <= constraint.max_ms
                )
                checks.append(
                    _check(
                        f"duration_constraint_{index}",
                        passed,
                        (
                            f"observed duration={duration}; required "
                            f"[{constraint.min_ms},{constraint.max_ms}] ms"
                        ),
                    )
                )
            elif isinstance(constraint, AspectRatioConstraint):
                passed = (
                    isinstance(width, int)
                    and isinstance(height, int)
                    and width * constraint.height == height * constraint.width
                )
                checks.append(
                    _check(
                        f"aspect_ratio_constraint_{index}",
                        passed,
                        (
                            f"observed dimensions={width}x{height}; requested ratio="
                            f"{constraint.width}:{constraint.height}"
                        ),
                    )
                )
            elif isinstance(constraint, (RequiredContentConstraint, ForbiddenContentConstraint)):
                checks.append(
                    _check(
                        f"semantic_constraint_{index}",
                        bool(decision.evidence_ids),
                        (
                            "semantic constraint has cited runtime evidence but cannot be "
                            "proven from media metadata"
                        ),
                        conclusive=False,
                    )
                )
        ancestors = _ancestor_tools(decision.output_artifact_id, records)
        if "subtitle" in state.task_input.instruction.casefold():
            checks.append(
                _check(
                    "subtitle_operation_used",
                    "add_subtitles" in ancestors,
                    "final artifact lineage includes add_subtitles",
                )
            )
        known = _known_evidence_ids(state)
        unknown = set(decision.evidence_ids) - known
        checks.append(
            _check(
                "completion_evidence_exists",
                not unknown,
                (
                    "all completion evidence IDs exist in public runtime state"
                    if not unknown
                    else "finish decision cited unknown evidence IDs"
                ),
            )
        )
        failed = [item for item in checks if item.conclusive and not item.passed]
        inconclusive = [item for item in checks if not item.conclusive]
        status: VerificationStatus = (
            "failed" if failed else "inconclusive" if inconclusive else "passed"
        )
        return VerificationResult(
            verification_id=f"verify-finish-{decision.output_artifact_id}",
            status=status,
            checks=tuple(checks),
            failure_types=tuple(item.check_name for item in failed),
        )
