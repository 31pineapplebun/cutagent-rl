"""M1B.5 private motion schema and metric regression tests."""

from pathlib import Path

import pytest
from cutagent_evaluation.m1b5_motion import (
    MotionCaseGold,
    MotionEntityGold,
    MotionEventGold,
    evaluate_motion_result,
)
from cutagent_evaluation.motion_dataset import MOTION_TASK_TYPES, generate_motion_case

from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import (
    EvidenceDescriptor,
    EvidenceRef,
    ModelPerformance,
    PerceptionConfig,
    PerceptionResult,
    ScenePerception,
    TemporalEvent,
    VideoWorldState,
    VisualAction,
    VisualEntity,
    VisualObservation,
)


def _result() -> PerceptionResult:
    scene_range = TimeRange(start_ms=0, end_ms=4000)
    evidence = EvidenceRef(
        artifact_id="frame-1",
        evidence_kind="keyframe",
        segment_id="scene-1",
        observed_ms=1000,
    )
    entities = (
        VisualEntity(label="red square", evidence_refs=(evidence,)),
        VisualEntity(label="blue circle", evidence_refs=(evidence,)),
    )
    actions = (
        VisualAction(subject="red square", action="moves right", evidence_refs=(evidence,)),
        VisualAction(subject="blue circle", action="moves right", evidence_refs=(evidence,)),
    )
    events = (
        TemporalEvent(
            event_id="event-red",
            description="red square moves right",
            time_range=TimeRange(start_ms=500, end_ms=1500),
            evidence_refs=(evidence,),
        ),
        TemporalEvent(
            event_id="event-blue",
            description="blue circle moves right",
            time_range=TimeRange(start_ms=2200, end_ms=3200),
            evidence_refs=(evidence,),
        ),
    )
    observation = VisualObservation(
        observation_id="observation-1",
        segment_id="scene-1",
        mode="keyframes",
        time_range=scene_range,
        entities=entities,
        actions=actions,
        temporal_events=events,
    )
    scene = ScenePerception(
        segment_id="scene-1",
        time_range=scene_range,
        keyframe_evidence=(evidence,),
        transcript_spans=(),
        ocr_spans=(),
        entities=entities,
        actions=actions,
        temporal_events=events,
        visual_observation_ids=(observation.observation_id,),
    )
    world = VideoWorldState(
        video_id="video-1",
        duration_ms=4000,
        evidence_catalog=(
            EvidenceDescriptor(
                **evidence.model_dump(),
                media_type="image/jpeg",
            ),
        ),
        scenes=(scene,),
    )
    return PerceptionResult(
        source_video_id="video-1",
        config=PerceptionConfig(asr_enabled=False),
        config_sha256="a" * 64,
        transcript_spans=(),
        visual_observations=(observation,),
        ocr_spans=(),
        temporal_events=events,
        world_state=world,
        provenance_artifacts=(),
        cache_records=(),
        performance=(
            ModelPerformance(
                operation="qwen_keyframes",
                model_id="fixture",
                model_revision="fixture",
                dtype="float32",
                device="cpu",
                latency_ms=12,
                peak_allocated_bytes=100,
                peak_reserved_bytes=120,
                frames=3,
            ),
        ),
        runtime_versions={"fixture": "1"},
        processing_time_ms=12,
    )


def test_motion_metrics_score_direction_order_and_localization() -> None:
    gold = MotionCaseGold(
        case_id="case-001",
        source_group_id="generated-motion-v1",
        task_type="a_before_b",
        seed=1,
        duration_ms=4000,
        source_sha256="b" * 64,
        expected_entities=(
            MotionEntityGold(label="red square", aliases=("red rectangle",)),
            MotionEntityGold(label="blue circle"),
        ),
        supported_entity_terms=("background",),
        expected_events=(
            MotionEventGold(
                actor="red square",
                actor_aliases=("red rectangle",),
                action="move_right",
                time_range=TimeRange(start_ms=500, end_ms=1500),
            ),
            MotionEventGold(
                actor="blue circle",
                action="move_right",
                time_range=TimeRange(start_ms=2000, end_ms=3000),
            ),
        ),
        expected_direction="right",
        expected_order=("red square", "blue circle"),
        generation_config={"fps": 8},
    )

    metrics = evaluate_motion_result(_result(), gold)

    assert metrics["entity_correctness"] == 1
    assert metrics["action_correct"] is True
    assert metrics["temporal_event_correct"] is True
    assert metrics["direction_correct"] is True
    assert metrics["event_order_correct"] is True
    assert metrics["temporal_localization_error_ms"] == 100
    assert metrics["unsupported_claim_rate"] == 0


def test_motion_gold_rejects_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        MotionCaseGold(
            case_id="case-invalid",
            source_group_id="generated-motion-v1",
            task_type="stationary",
            seed=0,
            duration_ms=0,
            source_sha256="a" * 64,
            expected_entities=(MotionEntityGold(label="red square"),),
            generation_config={},
        )


def test_motion_taxonomy_has_51_frozen_cases_and_fixture_is_reproducible(
    tmp_path: Path,
) -> None:
    from tests.media_fixtures import require_ffmpeg

    pytest.importorskip("PIL.Image")
    ffmpeg, _ = require_ffmpeg()
    first_path, first_gold = generate_motion_case(
        tmp_path / "first",
        task_type="short_motion",
        variant_index=0,
        seed=20260822,
        ffmpeg=ffmpeg,
    )
    second_path, second_gold = generate_motion_case(
        tmp_path / "second",
        task_type="short_motion",
        variant_index=0,
        seed=20260822,
        ffmpeg=ffmpeg,
    )

    assert len(MOTION_TASK_TYPES) * 3 == 51
    assert first_path.is_file() and second_path.is_file()
    assert first_gold.source_sha256 == second_gold.source_sha256
    assert first_gold.expected_events[0].time_range == TimeRange(start_ms=1750, end_ms=2250)
