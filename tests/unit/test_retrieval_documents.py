"""Scene representation separation and evidence ownership tests."""

import pytest
from pydantic import ValidationError

from cutagent.retrieval.documents import DocumentEvidence, SceneDocument, SceneDocumentBuilder
from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import EvidenceRef
from tests.retrieval_fixtures import sample_world_states


def test_scene_document_fields_remain_separate_and_deterministic() -> None:
    states = sample_world_states()
    builder = SceneDocumentBuilder()
    first = builder.build(states)
    second = builder.build(states)
    assert first == second
    red = first[0]
    assert red.transcript == "A red square moves right"
    assert red.ocr == "SALE 42"
    assert "red square" in red.structured_semantic
    assert "[TRANSCRIPT]" in red.combined
    assert "[OCR]" in red.combined
    assert red.inferred_temporal_boundaries_authoritative is False
    assert {item.evidence_type for item in red.field_evidence["combined"]} == {
        "transcript",
        "ocr",
        "structured_text",
    }


def test_scene_document_rejects_cross_scene_evidence() -> None:
    wrong = EvidenceRef(
        artifact_id="frame-wrong",
        evidence_kind="keyframe",
        segment_id="different-scene",
        observed_ms=50,
    )
    with pytest.raises(ValidationError, match="different scene"):
        SceneDocument(
            video_id="video-1",
            scene_id="scene-1",
            time_range=TimeRange(start_ms=0, end_ms=100),
            transcript="text",
            combined="text",
            field_evidence={
                "transcript": (
                    DocumentEvidence(evidence_type="transcript", reference=wrong, text="text"),
                )
            },
        )
