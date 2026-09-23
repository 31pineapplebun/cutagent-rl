"""Deterministic, field-separated scene document construction."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field, model_validator

from cutagent.schemas.base import Identifier, SchemaModel
from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import EvidenceRef, ScenePerception, VideoWorldState
from cutagent.schemas.retrieval import DocumentField, RetrievalEvidenceType


class DocumentEvidence(SchemaModel):
    evidence_type: RetrievalEvidenceType
    reference: EvidenceRef
    text: str = ""


class SceneDocument(SchemaModel):
    """Inspectable scene text. Model-inferred event times are never indexed as facts."""

    video_id: Identifier
    scene_id: Identifier
    time_range: TimeRange
    transcript: str = ""
    ocr: str = ""
    structured_semantic: str = ""
    combined: str = ""
    field_evidence: dict[DocumentField, tuple[DocumentEvidence, ...]] = Field(default_factory=dict)
    keyframe_evidence: tuple[EvidenceRef, ...] = ()
    inferred_temporal_boundaries_authoritative: bool = False

    @model_validator(mode="after")
    def validate_evidence_ownership(self) -> SceneDocument:
        references = [
            evidence.reference
            for field_evidence in self.field_evidence.values()
            for evidence in field_evidence
        ] + list(self.keyframe_evidence)
        for reference in references:
            if reference.segment_id is not None and reference.segment_id != self.scene_id:
                raise ValueError("document evidence belongs to a different scene")
            if reference.observed_ms is not None and not (
                self.time_range.start_ms <= reference.observed_ms < self.time_range.end_ms
            ):
                raise ValueError("document evidence timestamp lies outside its scene")
            if reference.time_range is not None and not (
                reference.time_range.start_ms < self.time_range.end_ms
                and self.time_range.start_ms < reference.time_range.end_ms
            ):
                raise ValueError("document evidence interval does not overlap its scene")
        return self

    def text_for(self, field: DocumentField) -> str:
        fields = {
            "transcript": self.transcript,
            "ocr": self.ocr,
            "structured_semantic": self.structured_semantic,
            "combined": self.combined,
        }
        return fields[field]


def _unique_refs(references: Iterable[EvidenceRef]) -> tuple[EvidenceRef, ...]:
    return tuple(dict.fromkeys(references))


def _semantic_text(scene: ScenePerception) -> str:
    parts: list[str] = []
    if scene.scene_summary:
        parts.append(f"summary: {scene.scene_summary}")
    if scene.entities:
        entities = []
        for entity in scene.entities:
            attributes = " ".join(
                f"{key}={value}" for key, value in sorted(entity.attributes.items())
            )
            entities.append(" ".join(item for item in (entity.label, attributes) if item))
        parts.append("entities: " + "; ".join(entities))
    if scene.actions:
        actions = [
            " ".join(item for item in (action.subject, action.action, action.object) if item)
            for action in scene.actions
        ]
        parts.append("actions: " + "; ".join(actions))
    if scene.temporal_events:
        # Deliberately omit inferred boundaries: descriptions are retrieval hints, not timing truth.
        parts.append(
            "candidate events: " + "; ".join(event.description for event in scene.temporal_events)
        )
    return "\n".join(parts)


def _semantic_evidence(scene: ScenePerception) -> tuple[DocumentEvidence, ...]:
    evidence: list[DocumentEvidence] = []
    evidence.extend(
        DocumentEvidence(
            evidence_type="structured_text", reference=ref, text=scene.scene_summary or ""
        )
        for ref in scene.summary_evidence_refs
    )
    for entity in scene.entities:
        evidence.extend(
            DocumentEvidence(evidence_type="structured_text", reference=ref, text=entity.label)
            for ref in entity.evidence_refs
        )
    for action in scene.actions:
        action_text = " ".join(
            item for item in (action.subject, action.action, action.object) if item
        )
        evidence.extend(
            DocumentEvidence(evidence_type="structured_text", reference=ref, text=action_text)
            for ref in action.evidence_refs
        )
    for event in scene.temporal_events:
        evidence.extend(
            DocumentEvidence(evidence_type="structured_text", reference=ref, text=event.description)
            for ref in event.evidence_refs
        )
    return tuple(evidence)


class SceneDocumentBuilder:
    """Pure builder: fixed world states produce byte-equivalent documents."""

    template_version = "m2a-scene-document-v1"

    def build(self, world_states: tuple[VideoWorldState, ...]) -> tuple[SceneDocument, ...]:
        documents: list[SceneDocument] = []
        for world in sorted(world_states, key=lambda item: item.video_id):
            for scene in world.scenes:
                transcript = "\n".join(span.text for span in scene.transcript_spans)
                ocr = "\n".join(span.exact_text for span in scene.ocr_spans)
                semantic = _semantic_text(scene)
                combined_parts = [
                    f"[TRANSCRIPT]\n{transcript}" if transcript else "",
                    f"[OCR]\n{ocr}" if ocr else "",
                    f"[SEMANTIC]\n{semantic}" if semantic else "",
                ]
                transcript_evidence = tuple(
                    DocumentEvidence(evidence_type="transcript", reference=ref, text=span.text)
                    for span in scene.transcript_spans
                    for ref in span.evidence_refs
                )
                ocr_evidence = tuple(
                    DocumentEvidence(evidence_type="ocr", reference=ref, text=span.exact_text)
                    for span in scene.ocr_spans
                    for ref in span.evidence_refs
                )
                semantic_evidence = _semantic_evidence(scene)
                combined_evidence = transcript_evidence + ocr_evidence + semantic_evidence
                documents.append(
                    SceneDocument(
                        video_id=world.video_id,
                        scene_id=scene.segment_id,
                        time_range=scene.time_range,
                        transcript=transcript,
                        ocr=ocr,
                        structured_semantic=semantic,
                        combined="\n".join(part for part in combined_parts if part),
                        field_evidence={
                            "transcript": transcript_evidence,
                            "ocr": ocr_evidence,
                            "structured_semantic": semantic_evidence,
                            "combined": combined_evidence,
                        },
                        keyframe_evidence=_unique_refs(scene.keyframe_evidence),
                    )
                )
        return tuple(documents)
