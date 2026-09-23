"""Public runtime media contracts for timestamp-preserving ingestion."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel


class RationalValue(SchemaModel):
    """A positive rational used for stream time bases and frame rates."""

    numerator: int = Field(gt=0)
    denominator: int = Field(gt=0)


class TimeRange(SchemaModel):
    """A normalized, half-open interval: [start_ms, end_ms)."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "TimeRange":
        if self.start_ms >= self.end_ms:
            raise ValueError("time range must satisfy start_ms < end_ms")
        return self

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class SourceTimestampRange(SchemaModel):
    """Half-open source-PTS interval in a declared stream time base."""

    start_pts: int
    end_pts: int
    time_base: RationalValue

    @model_validator(mode="after")
    def validate_order(self) -> "SourceTimestampRange":
        if self.start_pts >= self.end_pts:
            raise ValueError("source timestamp range must satisfy start_pts < end_pts")
        return self


class MediaStreamInfo(SchemaModel):
    """Timing metadata shared by decoded media streams."""

    stream_index: int = Field(ge=0)
    codec_type: Literal["video", "audio"]
    codec_name: NonEmptyStr
    time_base: RationalValue
    source_start_pts: int
    source_start_time_ms: int
    duration_ts: int | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, gt=0)


class VideoStreamInfo(MediaStreamInfo):
    codec_type: Literal["video"] = "video"
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    average_frame_rate: RationalValue | None = None
    real_frame_rate: RationalValue | None = None
    pixel_format: NonEmptyStr | None = None
    rotation_degrees: int | None = Field(default=None, ge=-360, le=360)
    frame_count: int | None = Field(default=None, gt=0)
    variable_frame_rate: bool | None = None
    vfr_indicators: tuple[NonEmptyStr, ...] = ()


class AudioStreamInfo(MediaStreamInfo):
    codec_type: Literal["audio"] = "audio"
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(gt=0)


StreamInfo = Annotated[
    VideoStreamInfo | AudioStreamInfo,
    Field(discriminator="codec_type"),
]


class VideoAsset(SchemaModel):
    """Immutable source media plus parsed timing metadata.

    Artifact URIs remain runtime-internal and are never copied into a policy
    view. The normalized CutAgent timeline always begins at zero milliseconds.
    """

    video_id: Identifier
    source: ArtifactRef
    raw_ffprobe: ArtifactRef
    container_formats: tuple[NonEmptyStr, ...]
    duration_ms: int = Field(gt=0)
    normalized_start_ms: Literal[0] = 0
    source_start_time_ms: int
    video_stream: VideoStreamInfo
    audio_streams: tuple[AudioStreamInfo, ...] = ()

    @field_validator("container_formats")
    @classmethod
    def validate_formats(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one container format is required")
        if len(value) != len(set(value)):
            raise ValueError("container formats must be unique")
        return value

    @model_validator(mode="after")
    def validate_stream_duration(self) -> "VideoAsset":
        if self.video_stream.duration_ms is not None and self.video_stream.duration_ms <= 0:
            raise ValueError("video stream duration must be positive")
        return self


class SceneDetectionProvenance(SchemaModel):
    backend: Literal["ffmpeg_scene"] = "ffmpeg_scene"
    threshold: float = Field(gt=0, lt=1)
    minimum_scene_duration_ms: int = Field(gt=0)
    ffmpeg_version: NonEmptyStr
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")


class SceneSegment(SchemaModel):
    segment_id: Identifier
    parent_video_id: Identifier
    time_range: TimeRange
    source_timestamps: SourceTimestampRange
    boundary_score: float | None = Field(default=None, ge=0, le=1)
    provenance: SceneDetectionProvenance


class KeyframeExtractionConfig(SchemaModel):
    strategy: Literal["midpoint", "uniform"] = "midpoint"
    frames_per_scene: int = Field(default=1, ge=1, le=32)
    image_format: Literal["png", "jpg"] = "jpg"

    @model_validator(mode="after")
    def validate_strategy_count(self) -> "KeyframeExtractionConfig":
        if self.strategy == "midpoint" and self.frames_per_scene != 1:
            raise ValueError("midpoint strategy requires frames_per_scene=1")
        return self


class KeyframeRef(SchemaModel):
    keyframe_id: Identifier
    parent_video_id: Identifier
    segment_id: Identifier
    requested_timestamp_ms: int = Field(ge=0)
    observed_timestamp_ms: int | None = Field(default=None, ge=0)
    observed_source_pts: int | None = None
    artifact: ArtifactRef
    extraction_config: KeyframeExtractionConfig
    ffmpeg_version: NonEmptyStr
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")


class NormalizationConfig(SchemaModel):
    policy: Literal["source_only", "auto"] = "source_only"
    maximum_analysis_width: int = Field(default=1280, gt=0)
    accepted_video_codecs: tuple[NonEmptyStr, ...] = ("h264", "hevc", "vp9", "av1")
    accepted_pixel_formats: tuple[NonEmptyStr, ...] = ("yuv420p", "yuv420p10le")
    proxy_video_codec: NonEmptyStr = "libx264"
    proxy_pixel_format: NonEmptyStr = "yuv420p"


class TimestampMapping(SchemaModel):
    """Mapping between source/proxy PTS and the common normalized timeline."""

    rule: Literal["subtract_origin_then_scale_to_ms"] = "subtract_origin_then_scale_to_ms"
    source_time_base: RationalValue
    source_origin_pts: int
    analysis_time_base: RationalValue
    analysis_origin_pts: int
    preserves_frame_timestamps: bool


class AnalysisProxy(SchemaModel):
    source: ArtifactRef
    proxy: ArtifactRef
    raw_ffprobe: ArtifactRef
    transformation_config: dict[str, JsonValue]
    ffmpeg_version: NonEmptyStr
    timestamp_mapping: TimestampMapping


class NormalizationResult(SchemaModel):
    proxy_required: bool
    reasons: tuple[NonEmptyStr, ...]
    analysis_proxy: AnalysisProxy | None = None

    @model_validator(mode="after")
    def validate_proxy(self) -> "NormalizationResult":
        if self.proxy_required != (self.analysis_proxy is not None):
            raise ValueError("proxy_required must match analysis_proxy presence")
        return self


class IngestionConfig(SchemaModel):
    scene_threshold: float = Field(default=0.3, gt=0, lt=1)
    minimum_scene_duration_ms: int = Field(default=500, gt=0)
    keyframes: KeyframeExtractionConfig = Field(default_factory=KeyframeExtractionConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    command_timeout_seconds: int = Field(default=120, gt=0, le=3600)


class OperationCacheRecord(SchemaModel):
    operation: Literal["ffprobe", "normalization", "scene_split", "keyframe_extract"]
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    hit: bool
    tool_version: NonEmptyStr


class IngestionResult(SchemaModel):
    video: VideoAsset
    config: IngestionConfig
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization: NormalizationResult
    scenes: tuple[SceneSegment, ...]
    keyframes: tuple[KeyframeRef, ...]
    cache_records: tuple[OperationCacheRecord, ...]
    tool_versions: dict[str, NonEmptyStr]
    processing_time_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_timeline(self) -> "IngestionResult":
        if not self.scenes:
            raise ValueError("ingestion must contain at least one scene")
        segment_ids: set[str] = set()
        expected_start = 0
        for scene in self.scenes:
            if scene.segment_id in segment_ids:
                raise ValueError("scene segment identifiers must be unique")
            segment_ids.add(scene.segment_id)
            if scene.parent_video_id != self.video.video_id:
                raise ValueError("scene parent does not match video")
            if scene.time_range.start_ms != expected_start:
                raise ValueError("scenes must be ordered, non-overlapping, and contiguous")
            if scene.time_range.end_ms > self.video.duration_ms:
                raise ValueError("scene interval exceeds known video duration")
            expected_start = scene.time_range.end_ms
        if expected_start != self.video.duration_ms:
            raise ValueError("scenes must cover the complete normalized video timeline")

        scenes_by_id = {scene.segment_id: scene for scene in self.scenes}
        keyframe_ids: set[str] = set()
        for keyframe in self.keyframes:
            if keyframe.keyframe_id in keyframe_ids:
                raise ValueError("keyframe identifiers must be unique")
            keyframe_ids.add(keyframe.keyframe_id)
            if keyframe.parent_video_id != self.video.video_id:
                raise ValueError("keyframe parent does not match video")
            parent_scene = scenes_by_id.get(keyframe.segment_id)
            if parent_scene is None:
                raise ValueError("keyframe references an unknown scene")
            if not (
                parent_scene.time_range.start_ms
                <= keyframe.requested_timestamp_ms
                < parent_scene.time_range.end_ms
            ):
                raise ValueError("requested keyframe timestamp lies outside its scene")
            if (
                keyframe.observed_timestamp_ms is not None
                and keyframe.observed_timestamp_ms >= self.video.duration_ms
            ):
                raise ValueError("observed keyframe timestamp exceeds the video duration")
        return self
