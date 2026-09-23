"""M1B.5 temporal-prompt ablation contracts without loading model weights."""

from pathlib import Path

from cutagent.core.artifacts import ArtifactRef
from cutagent.perception.backends.qwen import Qwen3VLVLMBackend
from cutagent.perception.protocols import VisualInput, VLMRequest
from cutagent.schemas.media import (
    RationalValue,
    SceneDetectionProvenance,
    SceneSegment,
    SourceTimestampRange,
    TimeRange,
    VideoAsset,
    VideoStreamInfo,
)
from cutagent.schemas.perception import EvidenceRef, PerceptionConfig


def _request(tmp_path: Path, *, explicit: bool) -> VLMRequest:
    image_paths = tuple(tmp_path / f"frame-{index}.jpg" for index in range(3))
    for image_path in image_paths:
        image_path.write_bytes(b"fixture")
    source = ArtifactRef(
        artifact_id="source-1",
        uri=(tmp_path / "source.mp4").resolve().as_uri(),
        sha256="a" * 64,
        size_bytes=1,
        media_type="video/mp4",
    )
    video = VideoAsset(
        video_id="video-1",
        source=source,
        raw_ffprobe=source.model_copy(update={"artifact_id": "probe-1"}),
        container_formats=("mov", "mp4"),
        duration_ms=3000,
        source_start_time_ms=0,
        video_stream=VideoStreamInfo(
            stream_index=0,
            codec_name="h264",
            time_base=RationalValue(numerator=1, denominator=1000),
            source_start_pts=0,
            source_start_time_ms=0,
            duration_ts=3000,
            duration_ms=3000,
            width=384,
            height=256,
            average_frame_rate=RationalValue(numerator=8, denominator=1),
            real_frame_rate=RationalValue(numerator=8, denominator=1),
            pixel_format="yuv420p",
        ),
    )
    scene = SceneSegment(
        segment_id="scene-1",
        parent_video_id=video.video_id,
        time_range=TimeRange(start_ms=0, end_ms=3000),
        source_timestamps=SourceTimestampRange(
            start_pts=0,
            end_pts=3000,
            time_base=RationalValue(numerator=1, denominator=1000),
        ),
        provenance=SceneDetectionProvenance(
            threshold=0.3,
            minimum_scene_duration_ms=500,
            ffmpeg_version="fixture-1",
            cache_key="b" * 64,
        ),
    )
    inputs = tuple(
        VisualInput(
            evidence_alias=f"keyframe_{index + 1}",
            path=image_path,
            evidence_ref=EvidenceRef(
                artifact_id=f"keyframe-{index + 1}",
                evidence_kind="keyframe",
                segment_id=scene.segment_id,
                observed_ms=index * 1000,
            ),
        )
        for index, image_path in enumerate(image_paths)
    )
    return VLMRequest(
        video=video,
        scene=scene,
        visual_inputs=inputs,
        output_directory=tmp_path,
        config=PerceptionConfig(
            temporal_prompt_style="explicit_comparison" if explicit else "baseline",
            label_temporal_phases=explicit,
        ),
    )


def test_explicit_temporal_prompt_labels_phases_and_requests_comparison(
    tmp_path: Path,
) -> None:
    backend = Qwen3VLVLMBackend(model_cache=str(tmp_path / "models"))
    messages = backend._messages(_request(tmp_path, explicit=True), "base prompt")
    text = str(messages[0]["content"][-1]["text"])

    assert "chronological phase=before" in text
    assert "chronological phase=middle" in text
    assert "chronological phase=after" in text
    assert "object motion from camera motion" in text
    assert "which motion starts first" in text


def test_baseline_prompt_does_not_receive_ablation_wording(tmp_path: Path) -> None:
    backend = Qwen3VLVLMBackend(model_cache=str(tmp_path / "models"))
    messages = backend._messages(_request(tmp_path, explicit=False), "base prompt")
    text = str(messages[0]["content"][-1]["text"])

    assert "chronological phase=" not in text
    assert "which motion starts first" not in text
