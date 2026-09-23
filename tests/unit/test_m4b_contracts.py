"""Contract tests for M4B handoff, recovery safety, and leakage boundaries."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from cutagent.agent.m4b_loop_detection import M4BLoopDetector
from cutagent.agent.m4b_planning import build_local_recovery_patch
from cutagent.agent.m4b_policy_context import (
    M4BPolicyContextBuilder,
    M4BPolicyViewSerializer,
)
from cutagent.agent.m4b_state_reducer import M4BStateReducer
from cutagent.agent.planning import update_readiness
from cutagent.core.artifacts import ArtifactRef
from cutagent.core.errors import InvalidEventError, StructuredOutputError
from cutagent.models.qwen_policy_m4b import Qwen3VLPolicyBackendM4B
from cutagent.schemas.agent import PlanGraph
from cutagent.schemas.event import PlanNode, ToolObservation
from cutagent.schemas.m4b_agent import (
    M4BAgentState,
    M4BEventEnvelope,
    M4BRuntimeConfig,
    RemainingBudgetSummary,
    RetryCurrentNode,
    SanitizedSystemErrorEvent,
    StepOutcomeSummary,
    WorkingArtifactValidationEvent,
)
from cutagent.schemas.media import TimeRange
from cutagent.schemas.state import ExecutionBudget
from cutagent.schemas.task_input import DurationConstraint, TaskInput
from cutagent.schemas.tools import TrimVideoArgs, TrimVideoCall


def _task(source: ArtifactRef) -> TaskInput:
    return TaskInput(
        task_id="m4b-contract-task",
        video_ref=source,
        instruction="Trim 0-1000 ms and validate the output.",
        user_constraints=(DurationConstraint(min_ms=900, max_ms=1100),),
    )


def _state(source: ArtifactRef) -> M4BAgentState:
    return M4BAgentState.initial(
        _task(source),
        ExecutionBudget(
            max_steps=12,
            max_tool_calls=10,
            max_wall_time_ms=120_000,
        ),
    )


def _envelope(state: M4BAgentState, event: object, source: str) -> M4BEventEnvelope:
    return M4BEventEnvelope.model_validate(
        {
            "event_id": f"m4b-event-{state.last_sequence_no + 1}",
            "task_id": state.task_input.task_id,
            "sequence_no": state.last_sequence_no + 1,
            "event": event,
            "emitted_by": source,
            "created_at": datetime.now(UTC),
            "parent_state_version": state.state_version,
        }
    )


def test_editor_observation_updates_working_artifact_and_replays(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    output_path = tmp_path / "output.mp4"
    source_path.write_bytes(b"source")
    output_path.write_bytes(b"output")
    source = ArtifactRef.from_path(source_path, artifact_id="source-video", media_type="video/mp4")
    output = ArtifactRef.from_path(output_path, artifact_id="edited-video", media_type="video/mp4")
    initial = _state(source)
    observation = ToolObservation(
        call_id="trim-call",
        tool_name="trim_video",
        status="success",
        public_summary="trim succeeded",
        details={
            "duration_ms": 1000,
            "width": 320,
            "height": 180,
            "has_audio": True,
        },
        artifacts=(output,),
    )
    first = _envelope(initial, observation, "tool")
    after_edit = M4BStateReducer.apply(initial, first)
    assert after_edit.working_artifacts.original_input_artifact.artifact_id == "source-video"
    assert after_edit.working_artifacts.current_working_artifact.artifact_id == "edited-video"
    assert after_edit.working_artifacts.latest_generated_artifact is not None
    assert after_edit.working_artifacts.final_candidate_artifact is None

    validation = WorkingArtifactValidationEvent(
        artifact_id="edited-video",
        source_tool_call_id="validate-call",
        status="passed",
    )
    second = _envelope(after_edit, validation, "handoff")
    final = M4BStateReducer.apply(after_edit, second)
    assert final.working_artifacts.final_candidate_artifact is not None
    assert (
        final.working_artifacts.final_candidate_artifact.validation_status
        == "independently_validated"
    )
    assert M4BStateReducer.replay(initial, (first, second)) == final


def test_validation_cannot_commit_an_unrelated_artifact(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    source = ArtifactRef.from_path(source_path, artifact_id="source-video", media_type="video/mp4")
    state = _state(source)
    event = WorkingArtifactValidationEvent(
        artifact_id="unknown-artifact",
        source_tool_call_id="validate-call",
        status="passed",
    )
    with pytest.raises(InvalidEventError):
        M4BStateReducer.apply(state, _envelope(state, event, "handoff"))


def test_compact_recovery_cannot_rewrite_succeeded_history() -> None:
    graph = update_readiness(
        PlanGraph(
            revision=3,
            nodes=(
                PlanNode(
                    node_id="search",
                    subgoal="find scene",
                    status="succeeded",
                    completion_criteria=("scene evidence",),
                ),
                PlanNode(
                    node_id="trim",
                    subgoal="trim scene",
                    dependencies=("search",),
                    status="failed",
                    completion_criteria=("clip exists",),
                ),
            ),
        )
    )
    application = build_local_recovery_patch(
        graph,
        RetryCurrentNode(node_id="trim", reason="use corrected observable arguments"),
        active_node_id="trim",
        recovery_index=1,
    )
    assert application is not None
    by_id = {node.node_id: node for node in application.patch.steps}
    assert by_id["search"] == graph.nodes[0]
    assert by_id["trim"].status == "ready"
    with pytest.raises(ValueError, match="active"):
        build_local_recovery_patch(
            graph,
            RetryCurrentNode(node_id="search", reason="forbidden rewrite"),
            active_node_id="trim",
            recovery_index=2,
        )


def test_policy_context_whitelists_handoff_without_private_fields(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    source = ArtifactRef.from_path(source_path, artifact_id="source-video", media_type="video/mp4")
    state = _state(source)
    outcome = StepOutcomeSummary(
        summary_id="summary-1",
        remaining_budget=RemainingBudgetSummary(
            steps=12,
            tool_calls=10,
            search_calls=3,
            edit_calls=8,
            structured_output_repairs=2,
            model_tokens=64_000,
            wall_time_ms=120_000,
        ),
        unsatisfied_completion_criteria=("no validated output",),
        completion_ready=False,
    )
    context = M4BPolicyContextBuilder().build(state, step_outcome=outcome)
    serialized = M4BPolicyViewSerializer.to_json(context).casefold()
    for forbidden in (
        "benchmarkgold",
        "source_group_id",
        '"split"',
        "sha256",
        "file://",
        str(tmp_path).casefold(),
        "evaluator_metadata",
    ):
        assert forbidden not in serialized
    assert '"current_working_media"' in serialized
    assert '"completion_ready":false' in serialized


def test_sanitized_system_error_rejects_paths_and_secrets() -> None:
    with pytest.raises(ValidationError):
        SanitizedSystemErrorEvent(
            error_id="error-1",
            error_category="runtime_error",
            safe_message="failed at C:/private/source.mp4",
            component="m4b_runtime",
            operation="decide",
        )
    with pytest.raises(ValidationError):
        SanitizedSystemErrorEvent(
            error_id="error-2",
            error_category="runtime_error",
            safe_message="password was rejected",
            component="m4b_runtime",
            operation="decide",
        )


def test_repeated_editor_distinguishes_changed_working_artifact() -> None:
    detector = M4BLoopDetector(maximum_identical_actions=1)
    first = TrimVideoCall(
        tool_call_id="call-1",
        arguments=TrimVideoArgs(
            input_artifact_id="source-video",
            time_range=TimeRange(start_ms=0, end_ms=1000),
        ),
    )
    changed = TrimVideoCall(
        tool_call_id="call-2",
        arguments=TrimVideoArgs(
            input_artifact_id="working-video",
            time_range=TimeRange(start_ms=0, end_ms=500),
        ),
    )
    assert (
        detector.observe_editor_call(
            first,
            current_working_artifact_id="source-video",
            information_version=0,
        )
        is None
    )
    assert (
        detector.observe_editor_call(
            changed,
            current_working_artifact_id="working-video",
            information_version=1,
        )
        is None
    )


def test_m4b_primary_view_is_structured_only() -> None:
    config = M4BRuntimeConfig(protocol_variant="compact_recovery")
    assert config.policy_view_mode == "structured_state"
    with pytest.raises(ValidationError):
        M4BRuntimeConfig.model_validate(
            {"protocol_variant": "compact_recovery", "policy_view_mode": "multimodal_evidence"}
        )


def test_repeated_structured_object_repair_rejects_divergent_output() -> None:
    assert Qwen3VLPolicyBackendM4B._parse_payload('{"revision":1}{"revision":1}') == {"revision": 1}
    with pytest.raises(StructuredOutputError):
        Qwen3VLPolicyBackendM4B._parse_payload('{"revision":1}{"revision":2}')
