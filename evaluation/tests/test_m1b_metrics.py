"""Transparent offline metric regression tests."""

import pytest
from cutagent_evaluation.m1b_metrics import (
    character_error_rate,
    edit_distance,
    temporal_iou,
    word_error_rate,
)

from cutagent.schemas.media import TimeRange


def test_edit_distance_and_asr_error_rates() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert word_error_rate("red square", "red circle") == pytest.approx(0.5)
    assert word_error_rate("Red square.", "red square") == 0
    assert character_error_rate("CASE 01", "case01") == 0


def test_temporal_iou_uses_half_open_intervals() -> None:
    assert temporal_iou(
        TimeRange(start_ms=0, end_ms=1000),
        TimeRange(start_ms=500, end_ms=1500),
    ) == pytest.approx(1 / 3)
    assert (
        temporal_iou(
            TimeRange(start_ms=0, end_ms=500),
            TimeRange(start_ms=500, end_ms=1000),
        )
        == 0
    )
