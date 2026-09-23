"""Policy-visible serialization must not leak private or infrastructure data."""

import json
from datetime import UTC, datetime

import pytest
from cutagent_evaluation.schemas import BenchmarkGold, DatasetSplit

from cutagent.agent.policy_context import PolicyContextBuilder, PolicyViewSerializer
from cutagent.agent.state_reducer import StateReducer
from cutagent.core.errors import PolicyVisibilityError
from cutagent.schemas.event import AgentEventEnvelope, ToolObservation
from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import (
    EvidenceDescriptor,
    EvidenceRef,
    ScenePerception,
    VideoWorldState,
    VisualEntity,
)
from cutagent.schemas.state import AgentState


def _state_with_internal_observation(initial_state: AgentState) -> AgentState:
    observation = ToolObservation(
        call_id="call-private",
        tool_name="synthetic_tool",
        status="success",
        public_summary="observable operation completed",
        details={
            "split": "locked_test",
            "source_group_id": "group-secret",
            "filesystem_path": "/srv/private/result.json",
            "artifact_hash": "f" * 64,
            "run_metadata": {"run_id": "private-run"},
        },
        artifacts=(initial_state.task_input.video_ref,),
    )
    envelope = AgentEventEnvelope(
        event_id="event-private",
        task_id=initial_state.task_input.task_id,
        sequence_no=1,
        event=observation,
        emitted_by="tool",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        parent_state_version=0,
    )
    return StateReducer.apply(initial_state, envelope)


def test_policy_context_excludes_internal_and_evaluator_fields(initial_state: AgentState) -> None:
    state = _state_with_internal_observation(initial_state)
    payload = PolicyViewSerializer.to_dict(PolicyContextBuilder().build(state))
    serialized = json.dumps(payload, sort_keys=True)

    forbidden_values = (
        "BenchmarkGold",
        "locked_test",
        "group-secret",
        "/srv/private/result.json",
        state.task_input.video_ref.uri,
        state.task_input.video_ref.sha256,
        "private-run",
    )
    assert all(value not in serialized for value in forbidden_values)

    forbidden_keys = {
        "split",
        "source_group_id",
        "uri",
        "sha256",
        "run_metadata",
        "evaluator_metadata",
        "expected_evidence",
        "ground_truth",
    }

    def collect_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            collected_keys = set(value)
            for child in value.values():
                collected_keys.update(collect_keys(child))
            return collected_keys
        if isinstance(value, list):
            collected_keys = set()
            for child in value:
                collected_keys.update(collect_keys(child))
            return collected_keys
        return set()

    assert collect_keys(payload).isdisjoint(forbidden_keys)


def test_policy_context_builder_has_no_gold_input(initial_state: AgentState) -> None:
    gold = BenchmarkGold(
        gold_id="gold-1",
        task_id=initial_state.task_input.task_id,
        source_group_id="source-1",
        split=DatasetSplit.LOCKED_TEST,
        expected_evidence=(),
        ground_truth={"answer": "private"},
        adjudication_version="v1",
        frozen_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    builder = PolicyContextBuilder()
    context = builder.build(initial_state)
    private_answer = gold.ground_truth["answer"]

    assert not hasattr(context, "gold")
    assert isinstance(private_answer, str)
    assert private_answer not in PolicyViewSerializer.to_json(context)


def test_path_like_public_summary_is_rejected(initial_state: AgentState) -> None:
    observation = ToolObservation(
        call_id="call-path",
        tool_name="synthetic_tool",
        status="success",
        public_summary="/srv/private/output.mp4",
    )
    event = AgentEventEnvelope(
        event_id="event-path",
        task_id=initial_state.task_input.task_id,
        sequence_no=1,
        event=observation,
        emitted_by="tool",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        parent_state_version=0,
    )
    state = StateReducer.apply(initial_state, event)
    with pytest.raises(PolicyVisibilityError, match="filesystem path"):
        PolicyViewSerializer.to_dict(PolicyContextBuilder().build(state))


def test_policy_world_state_view_whitelists_no_internal_artifact_data(
    initial_state: AgentState,
) -> None:
    evidence = EvidenceRef(
        artifact_id="frame-visible-001",
        evidence_kind="keyframe",
        segment_id="scene-001",
        observed_ms=500,
    )
    world = VideoWorldState(
        video_id="world-video-001",
        duration_ms=1000,
        evidence_catalog=(
            EvidenceDescriptor(
                artifact_id=initial_state.task_input.video_ref.artifact_id,
                evidence_kind="source_media",
                media_type="video/mp4",
                time_range=TimeRange(start_ms=0, end_ms=1000),
            ),
            EvidenceDescriptor(**evidence.model_dump(), media_type="image/jpeg"),
        ),
        scenes=(
            ScenePerception(
                segment_id="scene-001",
                time_range=TimeRange(start_ms=0, end_ms=1000),
                keyframe_evidence=(evidence,),
                transcript_spans=(),
                ocr_spans=(),
                entities=(VisualEntity(label="red square", evidence_refs=(evidence,)),),
                actions=(),
                temporal_events=(),
            ),
        ),
    )
    payload = PolicyViewSerializer.to_json(
        PolicyContextBuilder().build(initial_state, world_state=world)
    ).casefold()
    for forbidden in (
        "benchmarkgold",
        "source_group_id",
        '"split"',
        '"uri"',
        "sha256",
        "filesystem_path",
        "run_metadata",
        "evaluator_metadata",
    ):
        assert forbidden not in payload
