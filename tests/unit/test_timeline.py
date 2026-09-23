"""Exact source-PTS and normalized-millisecond conversion tests."""

from itertools import pairwise

import pytest

from cutagent.ingestion.timeline import Timeline, decimal_seconds_to_ms, parse_rational
from cutagent.schemas.media import RationalValue, TimeRange


def test_cfr_timeline_uses_pts_timebase_not_frame_rate() -> None:
    timeline = Timeline(
        time_base=RationalValue(numerator=1, denominator=90_000),
        origin_pts=180_000,
        duration_ms=2_000,
    )
    assert timeline.pts_to_ms(180_000) == 0
    assert timeline.pts_to_ms(225_000) == 500
    assert timeline.ms_to_pts(500) == 225_000


def test_vfr_like_pts_keep_irregular_temporal_spacing() -> None:
    timeline = Timeline(
        time_base=RationalValue(numerator=1, denominator=1000),
        origin_pts=0,
        duration_ms=1000,
    )
    pts = (0, 40, 125, 166, 310)
    mapped = tuple(timeline.pts_to_ms(value) for value in pts)
    assert mapped == pts
    assert [right - left for left, right in pairwise(mapped)] == [
        40,
        85,
        41,
        144,
    ]


def test_nonzero_start_is_normalized_to_zero() -> None:
    timeline = Timeline(
        time_base=RationalValue(numerator=1, denominator=1000),
        origin_pts=2500,
        duration_ms=1000,
    )
    assert timeline.pts_to_ms(2500) == 0
    assert timeline.pts_to_ms(2750) == 250
    assert timeline.ms_to_pts(0) == 2500


def test_millisecond_rounding_boundaries_are_deterministic() -> None:
    timeline = Timeline(
        time_base=RationalValue(numerator=1, denominator=2000),
        origin_pts=0,
        duration_ms=10,
    )
    assert timeline.pts_to_ms(1, rounding="floor") == 0
    assert timeline.pts_to_ms(1, rounding="ceil") == 1
    assert timeline.pts_to_ms(1, rounding="nearest") == 1
    assert decimal_seconds_to_ms("0.0004") == 0
    assert decimal_seconds_to_ms("0.0005") == 1
    assert decimal_seconds_to_ms("1.2345") == 1235


def test_interval_clipping_and_rejection() -> None:
    timeline = Timeline(
        time_base=RationalValue(numerator=1, denominator=1000),
        origin_pts=0,
        duration_ms=1000,
    )
    assert timeline.clip_range(-100, 1200) == TimeRange(start_ms=0, end_ms=1000)
    with pytest.raises(ValueError, match="exceeds"):
        timeline.validate_range(TimeRange(start_ms=900, end_ms=1001))
    with pytest.raises(ValueError, match="empty"):
        timeline.clip_range(1500, 2000)


def test_ffprobe_rational_parser_rejects_invalid_and_ignores_unknown() -> None:
    assert parse_rational("30000/1001") == RationalValue(numerator=30000, denominator=1001)
    assert parse_rational("0/0") is None
    with pytest.raises(ValueError, match="invalid rational"):
        parse_rational("not-a-rate")
