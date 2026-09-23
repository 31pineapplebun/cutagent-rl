"""M1B public schema and evidence-integrity contracts."""

import pytest
from pydantic import ValidationError

from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import (
    DirectlyVisibleText,
    EvidenceDescriptor,
    EvidenceRef,
    OCRSpan,
    ScenePerception,
    VideoWorldState,
    VisualEntity,
    VisualObservation,
)


def _keyframe_evidence() -> EvidenceRef:
    return EvidenceRef(
        artifact_id="frame-001",
        evidence_kind="keyframe",
        segment_id="scene-001",
        observed_ms=500,
    )


def test_direct_visual_claims_require_evidence() -> None:
    with pytest.raises(ValidationError):
        VisualEntity(label="red square", evidence_refs=())
    with pytest.raises(ValidationError):
        VisualObservation(
            observation_id="visual-001",
            segment_id="scene-001",
            mode="keyframes",
            time_range=TimeRange(start_ms=0, end_ms=1000),
            scene_summary="unsupported summary",
        )


def test_ocr_cannot_masquerade_as_inferred_text() -> None:
    with pytest.raises(ValidationError):
        OCRSpan.model_validate(
            {
                "span_id": "ocr-001",
                "exact_text": "CUTAGENT",
                "text_source": "inferred_semantic_text",
                "segment_id": "scene-001",
                "observed_ms": 500,
                "evidence_refs": [_keyframe_evidence().model_dump()],
            }
        )


def test_world_state_rejects_unknown_evidence_artifact() -> None:
    scene = ScenePerception(
        segment_id="scene-001",
        time_range=TimeRange(start_ms=0, end_ms=1000),
        keyframe_evidence=(_keyframe_evidence(),),
        transcript_spans=(),
        ocr_spans=(),
        entities=(VisualEntity(label="square", evidence_refs=(_keyframe_evidence(),)),),
        actions=(),
        temporal_events=(),
    )
    with pytest.raises(ValidationError, match="unknown evidence occurrence"):
        VideoWorldState(
            video_id="video-001",
            duration_ms=1000,
            evidence_catalog=(
                EvidenceDescriptor(
                    artifact_id="different-frame",
                    evidence_kind="keyframe",
                    media_type="image/jpeg",
                    segment_id="scene-001",
                    observed_ms=500,
                ),
            ),
            scenes=(scene,),
        )


def test_world_state_accepts_closed_evidence_catalog() -> None:
    evidence = _keyframe_evidence()
    scene = ScenePerception(
        segment_id="scene-001",
        time_range=TimeRange(start_ms=0, end_ms=1000),
        keyframe_evidence=(evidence,),
        transcript_spans=(),
        ocr_spans=(),
        entities=(VisualEntity(label="red square", evidence_refs=(evidence,)),),
        actions=(),
        temporal_events=(),
        scene_summary="A red square.",
        summary_evidence_refs=(evidence,),
    )
    world = VideoWorldState(
        video_id="video-001",
        duration_ms=1000,
        evidence_catalog=(
            EvidenceDescriptor(
                **evidence.model_dump(),
                media_type="image/jpeg",
            ),
        ),
        scenes=(scene,),
        global_summary="A red square.",
        global_summary_evidence_refs=(evidence,),
    )
    assert world.scenes[0].entities[0].label == "red square"


def test_same_content_artifact_may_be_evidence_at_distinct_timestamps() -> None:
    first = EvidenceRef(
        artifact_id="frame-same-content",
        evidence_kind="keyframe",
        segment_id="scene-001",
        observed_ms=250,
    )
    second = first.model_copy(update={"observed_ms": 750})
    scene = ScenePerception(
        segment_id="scene-001",
        time_range=TimeRange(start_ms=0, end_ms=1000),
        keyframe_evidence=(first, second),
        transcript_spans=(),
        ocr_spans=(),
        entities=(VisualEntity(label="static square", evidence_refs=(first, second)),),
        actions=(),
        temporal_events=(),
    )

    world = VideoWorldState(
        video_id="video-001",
        duration_ms=1000,
        evidence_catalog=(
            EvidenceDescriptor(**first.model_dump(), media_type="image/jpeg"),
            EvidenceDescriptor(**second.model_dump(), media_type="image/jpeg"),
        ),
        scenes=(scene,),
    )

    assert len(world.evidence_catalog) == 2


def test_visible_and_inferred_text_remain_separate() -> None:
    evidence = _keyframe_evidence()
    observation = VisualObservation(
        observation_id="visual-001",
        segment_id="scene-001",
        mode="keyframes",
        time_range=TimeRange(start_ms=0, end_ms=1000),
        directly_visible_text=(DirectlyVisibleText(exact_text="SALE", evidence_refs=(evidence,)),),
        inferred_semantic_text=("this may be an advertisement",),
    )
    assert observation.directly_visible_text[0].exact_text == "SALE"
    assert observation.inferred_semantic_text == ("this may be an advertisement",)
