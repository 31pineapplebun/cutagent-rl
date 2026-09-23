"""Keyframe sampling identities are timestamp based."""

from cutagent.ingestion.keyframes import requested_timestamps
from cutagent.schemas.media import (
    KeyframeExtractionConfig,
    RationalValue,
    SceneDetectionProvenance,
    SceneSegment,
    SourceTimestampRange,
    TimeRange,
)


def _scene(start_ms: int, end_ms: int) -> SceneSegment:
    time_base = RationalValue(numerator=1, denominator=1000)
    return SceneSegment(
        segment_id="scene-test",
        parent_video_id="video-test",
        time_range=TimeRange(start_ms=start_ms, end_ms=end_ms),
        source_timestamps=SourceTimestampRange(
            start_pts=start_ms,
            end_pts=end_ms,
            time_base=time_base,
        ),
        provenance=SceneDetectionProvenance(
            threshold=0.3,
            minimum_scene_duration_ms=100,
            ffmpeg_version="test",
            cache_key="a" * 64,
        ),
    )


def test_midpoint_sampling_is_inside_half_open_scene() -> None:
    assert requested_timestamps(_scene(100, 200), KeyframeExtractionConfig()) == (150,)
    assert requested_timestamps(_scene(100, 101), KeyframeExtractionConfig()) == (100,)


def test_uniform_sampling_is_deterministic_and_interior() -> None:
    config = KeyframeExtractionConfig(strategy="uniform", frames_per_scene=3)
    assert requested_timestamps(_scene(1000, 2000), config) == (1166, 1500, 1833)
