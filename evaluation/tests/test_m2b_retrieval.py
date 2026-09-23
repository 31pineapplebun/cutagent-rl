"""M2B held-out/private boundaries and offline analysis contracts."""

import pytest
from cutagent_evaluation.m2b_dataset import (
    HeldOutAction,
    HeldOutSceneGold,
    HeldOutVideoGold,
    assert_source_group_disjoint,
    build_heldout_validation_cases,
)
from cutagent_evaluation.schemas import DatasetSplit

from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import (
    EvidenceDescriptor,
    EvidenceRef,
    OCRSpan,
    PerceptionConfig,
    PerceptionResult,
    ScenePerception,
    TranscriptSpan,
    VideoWorldState,
    VisualAction,
    VisualEntity,
)


def _result(video_id: str) -> PerceptionResult:
    audio = EvidenceRef(
        artifact_id=f"audio-{video_id}",
        evidence_kind="source_audio_pcm",
        time_range=TimeRange(start_ms=0, end_ms=12000),
    )
    catalog = [EvidenceDescriptor(**audio.model_dump(), media_type="audio/wav")]
    scenes = []
    for index in range(4):
        start = index * 3000
        frame = EvidenceRef(
            artifact_id=f"frame-{video_id}-{index}",
            evidence_kind="keyframe",
            segment_id=f"scene-{index}",
            observed_ms=start + 1500,
        )
        vlm = EvidenceRef(
            artifact_id=f"vlm-{video_id}-{index}",
            evidence_kind="vlm_raw_output",
            segment_id=f"scene-{index}",
            time_range=TimeRange(start_ms=start, end_ms=start + 3000),
        )
        catalog.extend(
            (
                EvidenceDescriptor(**frame.model_dump(), media_type="image/jpeg"),
                EvidenceDescriptor(**vlm.model_dump(), media_type="application/json"),
            )
        )
        scenes.append(
            ScenePerception(
                segment_id=f"scene-{index}",
                time_range=TimeRange(start_ms=start, end_ms=start + 3000),
                keyframe_evidence=(frame,),
                transcript_spans=(
                    TranscriptSpan(
                        span_id=f"span-{index}",
                        text=f"Video marker scene {index}",
                        time_range=TimeRange(start_ms=start, end_ms=start + 2000),
                        evidence_refs=(audio,),
                    ),
                ),
                ocr_spans=(
                    OCRSpan(
                        span_id=f"ocr-{index}",
                        exact_text=f"M2B S{index + 1}",
                        segment_id=f"scene-{index}",
                        observed_ms=start + 1500,
                        evidence_refs=(frame,),
                    ),
                ),
                entities=(VisualEntity(label="red square", evidence_refs=(vlm,)),),
                actions=(
                    VisualAction(
                        subject="red square",
                        action="moves right",
                        evidence_refs=(vlm,),
                    ),
                ),
                temporal_events=(),
                scene_summary="A red square and companion object.",
                summary_evidence_refs=(vlm,),
            )
        )
    world = VideoWorldState(
        video_id=video_id,
        duration_ms=12000,
        evidence_catalog=tuple(catalog),
        scenes=tuple(scenes),
        global_summary="Four synthetic scenes.",
        global_summary_evidence_refs=tuple(scene.summary_evidence_refs[0] for scene in scenes),
    )
    return PerceptionResult(
        source_video_id=video_id,
        config=PerceptionConfig(),
        config_sha256="0" * 64,
        transcript_spans=tuple(span for scene in scenes for span in scene.transcript_spans),
        visual_observations=(),
        ocr_spans=tuple(span for scene in scenes for span in scene.ocr_spans),
        temporal_events=(),
        world_state=world,
        provenance_artifacts=(),
        cache_records=(),
        performance=(),
        runtime_versions={"test": "fake"},
        processing_time_ms=0,
    )


def _records() -> tuple[HeldOutVideoGold, ...]:
    output = []
    actions: tuple[HeldOutAction, ...] = (
        "stationary",
        "appear",
        "move_right",
        "disappear",
    )
    companions = ("green circle", "yellow square", "blue square", "red circle")
    for video_index in range(8):
        scenes = tuple(
            HeldOutSceneGold(
                scene_index=index,
                nominal_time_range=TimeRange(
                    start_ms=index * 3000,
                    end_ms=(index + 1) * 3000,
                ),
                primary_entity="red square",
                companion_entity=companions[index],
                action=actions[index],
                ocr_text=f"M2B{video_index + 1:02d} S{index + 1}",
                transcript=f"Video {video_index + 1} scene {index + 1} marker.",
            )
            for index in range(4)
        )
        output.append(
            HeldOutVideoGold(
                source_group_id=f"m2b-heldout-source-{video_index + 1:02d}",
                seed=video_index,
                source_sha256=f"{video_index + 1:064x}",
                scenes=scenes,
            )
        )
    return tuple(output)


def test_validation_set_is_held_out_private_and_exact_size() -> None:
    records = _records()
    results = {
        record.source_group_id: _result(f"video-{index}") for index, record in enumerate(records)
    }
    cases = build_heldout_validation_cases(records, results)
    assert len(cases) == 96
    assert {case.gold.split for case in cases} == {DatasetSplit.VALIDATION}
    assert_source_group_disjoint(cases, {"m2a-generated-development-v1"})
    for case in cases:
        public = case.query.model_dump(mode="json")
        assert "query_type" not in public
        assert "source_group_id" not in public
        assert "split" not in public


def test_validation_source_group_overlap_is_rejected() -> None:
    records = _records()
    results = {
        record.source_group_id: _result(f"video-{index}") for index, record in enumerate(records)
    }
    cases = build_heldout_validation_cases(records, results)
    with pytest.raises(ValueError, match="overlap"):
        assert_source_group_disjoint(cases, {cases[0].gold.source_group_id})
