"""Deterministic conversion between source PTS and normalized milliseconds."""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Literal

from cutagent.schemas.media import RationalValue, SourceTimestampRange, TimeRange

RoundingMode = Literal["floor", "ceil", "nearest"]


def _divide(numerator: int, denominator: int, mode: RoundingMode) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if mode == "floor":
        return numerator // denominator
    if mode == "ceil":
        return -((-numerator) // denominator)
    absolute = abs(numerator)
    quotient, remainder = divmod(absolute, denominator)
    rounded = quotient + int(remainder * 2 >= denominator)
    return rounded if numerator >= 0 else -rounded


def parse_rational(value: str | None) -> RationalValue | None:
    """Parse a positive ffprobe rational, treating 0/0 and N/A as absent."""

    if not value or value in {"N/A", "0/0"}:
        return None
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"invalid rational value: {value}") from error
    if fraction <= 0:
        return None
    return RationalValue(numerator=fraction.numerator, denominator=fraction.denominator)


def decimal_seconds_to_ms(value: str, *, rounding: RoundingMode = "nearest") -> int:
    """Convert a decimal seconds string to milliseconds without binary floats."""

    fraction = Fraction(Decimal(value)) * 1000
    return _divide(fraction.numerator, fraction.denominator, rounding)


@dataclass(frozen=True, slots=True)
class Timeline:
    """Affine mapping from source PTS to a zero-based CutAgent timeline.

    Point timestamps use nearest rounding with ties away from zero. Interval
    starts use floor and interval ends use ceil so conversion cannot silently
    remove source coverage.
    """

    time_base: RationalValue
    origin_pts: int
    duration_ms: int

    def __post_init__(self) -> None:
        if self.duration_ms <= 0:
            raise ValueError("timeline duration must be positive")

    def pts_to_ms(self, pts: int, *, rounding: RoundingMode = "nearest") -> int:
        ticks = pts - self.origin_pts
        numerator = ticks * self.time_base.numerator * 1000
        return _divide(numerator, self.time_base.denominator, rounding)

    def ms_to_pts(self, milliseconds: int, *, rounding: RoundingMode = "nearest") -> int:
        numerator = milliseconds * self.time_base.denominator
        denominator = self.time_base.numerator * 1000
        return self.origin_pts + _divide(numerator, denominator, rounding)

    def validate_range(self, time_range: TimeRange) -> TimeRange:
        if time_range.end_ms > self.duration_ms:
            raise ValueError("interval exceeds known video duration")
        return time_range

    def clip_range(self, start_ms: int, end_ms: int) -> TimeRange:
        clipped_start = max(0, min(start_ms, self.duration_ms))
        clipped_end = max(0, min(end_ms, self.duration_ms))
        if clipped_start >= clipped_end:
            raise ValueError("clipped interval is empty")
        return TimeRange(start_ms=clipped_start, end_ms=clipped_end)

    def source_range(self, time_range: TimeRange) -> SourceTimestampRange:
        valid = self.validate_range(time_range)
        start_pts = self.ms_to_pts(valid.start_ms, rounding="floor")
        end_pts = self.ms_to_pts(valid.end_ms, rounding="ceil")
        return SourceTimestampRange(
            start_pts=start_pts,
            end_pts=end_pts,
            time_base=self.time_base,
        )
