"""M1A public media schema contracts."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.media import (
    AudioStreamInfo,
    RationalValue,
    TimeRange,
    VideoAsset,
    VideoStreamInfo,
)


def _artifact(identifier: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=identifier,
        uri=Path(f"{identifier}.bin").resolve().as_uri(),
        sha256="a" * 64,
        media_type="application/octet-stream",
        size_bytes=1,
    )


def test_time_range_is_half_open_and_positive() -> None:
    value = TimeRange(start_ms=0, end_ms=1)
    assert value.duration_ms == 1
    with pytest.raises(ValidationError, match="start_ms < end_ms"):
        TimeRange(start_ms=5, end_ms=5)
    with pytest.raises(ValidationError):
        TimeRange(start_ms=-1, end_ms=1)


def test_stream_dimensions_rates_and_time_base_are_validated() -> None:
    with pytest.raises(ValidationError):
        RationalValue(numerator=0, denominator=1)
    with pytest.raises(ValidationError):
        VideoStreamInfo(
            stream_index=0,
            codec_name="h264",
            time_base=RationalValue(numerator=1, denominator=1000),
            source_start_pts=0,
            source_start_time_ms=0,
            width=0,
            height=90,
        )


def test_video_asset_records_audio_presence_without_paths_in_flat_fields() -> None:
    time_base = RationalValue(numerator=1, denominator=1000)
    video = VideoStreamInfo(
        stream_index=0,
        codec_name="h264",
        time_base=time_base,
        source_start_pts=2000,
        source_start_time_ms=2000,
        duration_ts=1000,
        duration_ms=1000,
        width=160,
        height=90,
        average_frame_rate=RationalValue(numerator=10, denominator=1),
        real_frame_rate=RationalValue(numerator=10, denominator=1),
    )
    audio = AudioStreamInfo(
        stream_index=1,
        codec_name="aac",
        time_base=RationalValue(numerator=1, denominator=16000),
        source_start_pts=32000,
        source_start_time_ms=2000,
        duration_ms=1000,
        sample_rate_hz=16000,
        channels=1,
    )
    asset = VideoAsset(
        video_id="video-test",
        source=_artifact("source"),
        raw_ffprobe=_artifact("probe"),
        container_formats=("mov", "mp4"),
        duration_ms=1000,
        source_start_time_ms=2000,
        video_stream=video,
        audio_streams=(audio,),
    )
    assert asset.normalized_start_ms == 0
    assert asset.audio_streams[0].sample_rate_hz == 16000
    assert "path" not in VideoAsset.model_fields


def test_media_schemas_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TimeRange.model_validate({"start_ms": 0, "end_ms": 1, "path": "/private"})
