"""Deterministic construction of a policy-safe structured video world state."""

from __future__ import annotations

from collections.abc import Iterable

from cutagent.schemas.media import IngestionResult
from cutagent.schemas.perception import (
    EvidenceDescriptor,
    EvidenceRef,
    OCRSpan,
    ScenePerception,
    TemporalEvent,
    TranscriptSpan,
    VideoWorldState,
    VisualObservation,
)


def _overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def _unique_evidence(references: Iterable[EvidenceRef]) -> tuple[EvidenceRef, ...]:
    seen: set[tuple[object, ...]] = set()
    output: list[EvidenceRef] = []
    for reference in references:
        key = (
            reference.artifact_id,
            reference.evidence_kind,
            reference.segment_id,
            reference.observed_ms,
            None
            if reference.time_range is None
            else (reference.time_range.start_ms, reference.time_range.end_ms),
        )
        if key not in seen:
            seen.add(key)
            output.append(reference)
    return tuple(output)


def _unique_catalog(
    descriptors: Iterable[EvidenceDescriptor],
) -> tuple[EvidenceDescriptor, ...]:
    seen: set[tuple[object, ...]] = set()
    output: list[EvidenceDescriptor] = []
    for descriptor in descriptors:
        key = (
            descriptor.artifact_id,
            descriptor.evidence_kind,
            descriptor.segment_id,
            descriptor.observed_ms,
            None
            if descriptor.time_range is None
            else (descriptor.time_range.start_ms, descriptor.time_range.end_ms),
        )
        if key not in seen:
            seen.add(key)
            output.append(descriptor)
    return tuple(output)


class WorldStateBuilder:
    """Pure reducer: fixed backend outputs always produce identical state."""

    def build(
        self,
        *,
        ingestion: IngestionResult,
        transcripts: tuple[TranscriptSpan, ...],
        observations: tuple[VisualObservation, ...],
        ocr_spans: tuple[OCRSpan, ...],
        temporal_events: tuple[TemporalEvent, ...],
        evidence_catalog: tuple[EvidenceDescriptor, ...],
    ) -> VideoWorldState:
        observations_by_scene: dict[str, list[VisualObservation]] = {}
        for observation in observations:
            observations_by_scene.setdefault(observation.segment_id, []).append(observation)
        ocr_by_scene: dict[str, list[OCRSpan]] = {}
        for span in ocr_spans:
            ocr_by_scene.setdefault(span.segment_id, []).append(span)

        scenes: list[ScenePerception] = []
        for scene in ingestion.scenes:
            scene_observations = tuple(observations_by_scene.get(scene.segment_id, ()))
            scene_transcripts = tuple(
                span
                for span in transcripts
                if _overlaps(
                    span.time_range.start_ms,
                    span.time_range.end_ms,
                    scene.time_range.start_ms,
                    scene.time_range.end_ms,
                )
            )
            keyframe_evidence = _unique_evidence(
                EvidenceRef(
                    artifact_id=keyframe.artifact.artifact_id,
                    evidence_kind="keyframe",
                    segment_id=scene.segment_id,
                    observed_ms=(
                        keyframe.observed_timestamp_ms
                        if keyframe.observed_timestamp_ms is not None
                        else keyframe.requested_timestamp_ms
                    ),
                )
                for keyframe in ingestion.keyframes
                if keyframe.segment_id == scene.segment_id
            )
            summaries = tuple(
                observation.scene_summary
                for observation in scene_observations
                if observation.scene_summary is not None
            )
            scene_summary = " | ".join(summaries) if summaries else None
            summary_evidence = _unique_evidence(
                reference
                for observation in scene_observations
                for reference in observation.summary_evidence_refs
            )
            scene_events = tuple(
                event for observation in scene_observations for event in observation.temporal_events
            ) + tuple(
                event
                for event in temporal_events
                if event
                not in {
                    nested
                    for observation in scene_observations
                    for nested in observation.temporal_events
                }
                and _overlaps(
                    event.time_range.start_ms,
                    event.time_range.end_ms,
                    scene.time_range.start_ms,
                    scene.time_range.end_ms,
                )
            )
            scenes.append(
                ScenePerception(
                    segment_id=scene.segment_id,
                    time_range=scene.time_range,
                    keyframe_evidence=keyframe_evidence,
                    transcript_spans=scene_transcripts,
                    ocr_spans=tuple(ocr_by_scene.get(scene.segment_id, ())),
                    entities=tuple(
                        entity
                        for observation in scene_observations
                        for entity in observation.entities
                    ),
                    actions=tuple(
                        action
                        for observation in scene_observations
                        for action in observation.actions
                    ),
                    temporal_events=scene_events,
                    scene_summary=scene_summary,
                    summary_evidence_refs=summary_evidence,
                    visual_observation_ids=tuple(
                        observation.observation_id for observation in scene_observations
                    ),
                )
            )

        scene_summaries = tuple(
            scene.scene_summary for scene in scenes if scene.scene_summary is not None
        )
        global_summary = " ".join(scene_summaries) if scene_summaries else None
        global_evidence = _unique_evidence(
            reference for scene in scenes for reference in scene.summary_evidence_refs
        )
        return VideoWorldState(
            video_id=ingestion.video.video_id,
            duration_ms=ingestion.video.duration_ms,
            evidence_catalog=_unique_catalog(evidence_catalog),
            scenes=tuple(scenes),
            global_summary=global_summary,
            global_summary_evidence_refs=global_evidence,
        )
