"""Real FFmpeg/ffprobe M1A integration tests on generated tiny media."""

from pathlib import Path

from cutagent.core.artifacts import ArtifactRef
from cutagent.ingestion.ffprobe import FFprobeAdapter
from cutagent.ingestion.pipeline import VideoIngestionPipeline
from cutagent.ingestion.timeline import Timeline
from cutagent.schemas.media import (
    IngestionConfig,
    KeyframeExtractionConfig,
    NormalizationConfig,
)
from tests.media_fixtures import (
    generate_low_frame_rate,
    generate_multiscene_with_audio,
    generate_nonzero_start,
    generate_vfr_like,
    require_ffmpeg,
)


def test_ffprobe_extracts_cfr_audio_and_raw_provenance(tmp_path: Path) -> None:
    _, ffprobe = require_ffmpeg()
    source_path = generate_multiscene_with_audio(tmp_path / "multiscene.mp4")
    source = ArtifactRef.from_path(
        source_path,
        artifact_id="source-test",
        media_type="video/mp4",
    )
    adapter = FFprobeAdapter(ffprobe)
    video = adapter.probe(
        source_path,
        source=source,
        raw_output_path=tmp_path / "raw.ffprobe.json",
    )
    assert 2900 <= video.duration_ms <= 3100
    assert (video.video_stream.width, video.video_stream.height) == (160, 90)
    assert video.video_stream.average_frame_rate is not None
    assert video.video_stream.time_base.denominator > 0
    assert video.audio_streams[0].sample_rate_hz == 16000
    assert (
        video.raw_ffprobe.sha256
        == ArtifactRef.from_path(
            tmp_path / "raw.ffprobe.json",
            artifact_id="verify",
            media_type="application/json",
        ).sha256
    )


def test_nonzero_start_is_preserved_but_runtime_timeline_is_zero_based(tmp_path: Path) -> None:
    _, ffprobe = require_ffmpeg()
    source_path = generate_nonzero_start(tmp_path / "offset.mp4")
    source = ArtifactRef.from_path(
        source_path,
        artifact_id="source-offset",
        media_type="video/mp4",
    )
    video = FFprobeAdapter(ffprobe).probe(
        source_path,
        source=source,
        raw_output_path=tmp_path / "offset.ffprobe.json",
    )
    assert video.source_start_time_ms >= 1900
    assert video.video_stream.source_start_pts > 0
    assert video.normalized_start_ms == 0

    pipeline = VideoIngestionPipeline(
        cache_root=tmp_path / "offset-cache",
        ffmpeg_executable=require_ffmpeg()[0],
        ffprobe_executable=ffprobe,
    )
    result = pipeline.ingest(source_path)
    assert len(result.scenes) == 1
    assert result.scenes[0].time_range.start_ms == 0
    assert result.scenes[0].time_range.end_ms == video.duration_ms
    assert result.keyframes[0].observed_timestamp_ms is not None
    assert result.keyframes[0].observed_timestamp_ms >= 500


def test_vfr_like_fixture_is_indicated_by_rate_metadata(tmp_path: Path) -> None:
    _, ffprobe = require_ffmpeg()
    source_path = generate_vfr_like(tmp_path / "vfr.mp4")
    source = ArtifactRef.from_path(
        source_path,
        artifact_id="source-vfr",
        media_type="video/mp4",
    )
    video = FFprobeAdapter(ffprobe).probe(
        source_path,
        source=source,
        raw_output_path=tmp_path / "vfr.ffprobe.json",
    )
    assert video.video_stream.variable_frame_rate is True
    assert "avg_frame_rate_differs_from_r_frame_rate" in video.video_stream.vfr_indicators


def test_pipeline_detects_scenes_extracts_keyframes_and_hits_cache(tmp_path: Path) -> None:
    ffmpeg, ffprobe = require_ffmpeg()
    source_path = generate_multiscene_with_audio(tmp_path / "multiscene.mp4")
    pipeline = VideoIngestionPipeline(
        cache_root=tmp_path / "cache",
        ffmpeg_executable=ffmpeg,
        ffprobe_executable=ffprobe,
    )
    config = IngestionConfig(
        scene_threshold=0.1,
        minimum_scene_duration_ms=300,
        keyframes=KeyframeExtractionConfig(strategy="uniform", frames_per_scene=2),
    )

    first = pipeline.ingest(source_path, config=config)
    second = pipeline.ingest(source_path, config=config)

    assert len(first.scenes) == 3
    assert [(scene.time_range.start_ms, scene.time_range.end_ms) for scene in first.scenes] == [
        (0, 1000),
        (1000, 2000),
        (2000, 3000),
    ]
    assert len(first.keyframes) == 6
    assert all(not record.hit for record in first.cache_records)
    assert all(record.hit for record in second.cache_records)
    assert first.scenes == second.scenes
    assert first.keyframes == second.keyframes
    assert all(keyframe.artifact.size_bytes > 0 for keyframe in first.keyframes)
    assert all(keyframe.observed_timestamp_ms is not None for keyframe in first.keyframes)
    assert all(
        keyframe.observed_timestamp_ms is not None
        and keyframe.observed_timestamp_ms >= keyframe.requested_timestamp_ms
        for keyframe in first.keyframes
    )
    timeline = Timeline(
        time_base=first.video.video_stream.time_base,
        origin_pts=first.video.video_stream.source_start_pts,
        duration_ms=first.video.duration_ms,
    )
    assert all(
        keyframe.observed_source_pts is not None
        and keyframe.observed_timestamp_ms is not None
        and abs(timeline.pts_to_ms(keyframe.observed_source_pts) - keyframe.observed_timestamp_ms)
        <= 1
        for keyframe in first.keyframes
    )


def test_keyframe_tail_falls_back_to_last_frame_covering_requested_time(
    tmp_path: Path,
) -> None:
    """A request in the last frame's display interval must still produce evidence."""

    ffmpeg, ffprobe = require_ffmpeg()
    source_path = generate_low_frame_rate(tmp_path / "low-fps.mp4")
    result = VideoIngestionPipeline(
        cache_root=tmp_path / "cache",
        ffmpeg_executable=ffmpeg,
        ffprobe_executable=ffprobe,
    ).ingest(
        source_path,
        config=IngestionConfig(
            scene_threshold=0.99,
            minimum_scene_duration_ms=300,
            keyframes=KeyframeExtractionConfig(strategy="uniform", frames_per_scene=3),
        ),
    )

    assert [frame.requested_timestamp_ms for frame in result.keyframes] == [333, 1000, 1666]
    assert [frame.observed_timestamp_ms for frame in result.keyframes] == [500, 1000, 1500]
    assert result.keyframes[-1].artifact.size_bytes > 0


def test_auto_normalization_creates_traceable_vfr_preserving_proxy(tmp_path: Path) -> None:
    ffmpeg, ffprobe = require_ffmpeg()
    source_path = generate_multiscene_with_audio(tmp_path / "multiscene.mp4")
    pipeline = VideoIngestionPipeline(
        cache_root=tmp_path / "cache",
        ffmpeg_executable=ffmpeg,
        ffprobe_executable=ffprobe,
    )
    config = IngestionConfig(
        scene_threshold=0.1,
        minimum_scene_duration_ms=300,
        normalization=NormalizationConfig(policy="auto"),
    )

    result = pipeline.ingest(source_path, config=config)
    cached = pipeline.ingest(source_path, config=config)

    assert result.normalization.proxy_required is True
    proxy = result.normalization.analysis_proxy
    assert proxy is not None
    assert proxy.source.sha256 == result.video.source.sha256
    assert proxy.proxy.sha256 != proxy.source.sha256
    assert proxy.timestamp_mapping.preserves_frame_timestamps is True
    assert proxy.transformation_config["vsync"] == "passthrough"
    assert proxy.transformation_config["audio"] == "omitted_from_visual_analysis_proxy"
    assert all(record.hit for record in cached.cache_records)
    assert cached.normalization == result.normalization
