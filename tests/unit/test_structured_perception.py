"""Strict parsing, bounded repair, and evidence alias tests."""

import json

import pytest

from cutagent.core.errors import StructuredOutputError
from cutagent.perception.structured_output import (
    parse_with_bounded_repair,
    to_visual_observation,
)
from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import EvidenceRef


def _valid_payload(evidence_id: str = "frame_1") -> str:
    return json.dumps(
        {
            "scene_summary": {"text": "A red square.", "evidence_ids": [evidence_id]},
            "entities": [{"label": "Red Square", "attributes": {}, "evidence_ids": [evidence_id]}],
            "actions": [],
            "directly_visible_text": [],
            "inferred_semantic_text": [],
            "temporal_events": [],
            "uncertainties": [],
        }
    )


def test_bounded_repair_uses_valid_second_attempt() -> None:
    repairs: list[tuple[str, str]] = []

    def repair(previous: str, error: str) -> str:
        repairs.append((previous, error))
        return _valid_payload()

    output, attempts = parse_with_bounded_repair("not-json", maximum_repairs=1, repair=repair)
    assert output.scene_summary is not None
    assert len(attempts) == 2
    assert len(repairs) == 1


def test_malformed_output_fails_after_bound() -> None:
    with pytest.raises(StructuredOutputError, match="after 2 attempt"):
        parse_with_bounded_repair(
            "not-json",
            maximum_repairs=1,
            repair=lambda _previous, _error: "still-not-json",
        )


def test_unknown_evidence_alias_is_not_fabricated() -> None:
    output, _ = parse_with_bounded_repair(
        _valid_payload("invented-frame"), maximum_repairs=0, repair=lambda _a, _b: ""
    )
    with pytest.raises(StructuredOutputError, match="unknown evidence alias"):
        to_visual_observation(
            output,
            observation_id="visual-001",
            segment_id="scene-001",
            mode="keyframes",
            scene_range=TimeRange(start_ms=0, end_ms=1000),
            available_evidence={
                "frame_1": EvidenceRef(
                    artifact_id="frame-001",
                    evidence_kind="keyframe",
                    observed_ms=500,
                )
            },
            repair_count=0,
        )
