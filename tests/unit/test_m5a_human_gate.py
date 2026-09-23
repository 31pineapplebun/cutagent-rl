from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.m5a_human_gate_lib import (
    agreement_payload,
    load_packet,
    template_rows,
    validate_pair,
    write_csv,
)


def _packet(tmp_path: Path) -> Path:
    path = tmp_path / "packet.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "packet_version": "test-packet-v1",
                "required_independent_raters": 2,
                "case_ids": ["case-1", "case-2"],
                "ratings": [],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return path


def _submission(packet_path: Path, path: Path, rater_id: str, *, disagree: bool) -> None:
    packet = load_packet(packet_path)
    rows = [dict(row) for row in template_rows(packet)]
    for index, row in enumerate(rows):
        row.update(
            {
                "rater_id": rater_id,
                "task_success": "failure" if disagree and index == 1 else "success",
                "semantic_constraint_score": "2" if disagree and index == 1 else "4",
                "failure_type": "planning_error" if disagree and index == 1 else "none",
                "confidence": "4",
            }
        )
    write_csv(path, tuple(rows))


def test_blank_template_contains_no_fabricated_ratings(tmp_path: Path) -> None:
    packet = load_packet(_packet(tmp_path))
    rows = template_rows(packet)
    assert len(rows) == 2
    assert all(row["task_success"] == "" for row in rows)
    assert all(row["semantic_constraint_score"] == "" for row in rows)
    assert all(row["failure_type"] == "" for row in rows)


def test_validate_real_pair_and_compute_agreement(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path)
    first_path = tmp_path / "rater_1.csv"
    second_path = tmp_path / "rater_2.csv"
    _submission(packet_path, first_path, "reviewer-alpha", disagree=False)
    _submission(packet_path, second_path, "reviewer-beta", disagree=True)
    packet, first, second = validate_pair(packet_path, first_path, second_path)
    result = agreement_payload(packet, first, second)
    assert result["status"] == "validated_real_human_pair"
    assert result["disagreement_count"] == 1
    assert result["protected_agent_results_used"] is False


def test_duplicate_identity_is_rejected(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path)
    first_path = tmp_path / "rater_1.csv"
    second_path = tmp_path / "rater_2.csv"
    _submission(packet_path, first_path, "same-person", disagree=False)
    _submission(packet_path, second_path, "same-person", disagree=True)
    with pytest.raises(ValueError, match="distinct real rater"):
        validate_pair(packet_path, first_path, second_path)


def test_model_identity_and_incomplete_submission_are_rejected(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path)
    model_path = tmp_path / "model.csv"
    human_path = tmp_path / "human.csv"
    valid_path = tmp_path / "valid.csv"
    _submission(packet_path, model_path, "ChatGPT-rater", disagree=False)
    _submission(packet_path, human_path, "real-reviewer", disagree=True)
    _submission(packet_path, valid_path, "other-reviewer", disagree=False)
    with pytest.raises(ValueError, match="LLM/model"):
        validate_pair(packet_path, model_path, human_path)
    human_path.write_text("rater_id\nreal-reviewer\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid rating columns"):
        validate_pair(packet_path, valid_path, human_path)
