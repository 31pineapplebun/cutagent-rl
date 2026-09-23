"""Public typed contracts for the safe M3A tool environment."""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter, field_validator, model_validator

from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.event import ToolObservation
from cutagent.schemas.media import TimeRange
from cutagent.schemas.retrieval import RetrievalEvidenceType

ToolName = Literal[
    "search_video",
    "inspect_media",
    "trim_video",
    "concat_videos",
    "change_speed",
    "add_subtitles",
    "reframe_video",
    "normalize_audio",
    "validate_media",
]
ToolErrorCategory = Literal[
    "invalid_call",
    "unknown_tool",
    "capability_denied",
    "artifact_not_allowed",
    "artifact_not_found",
    "filesystem_violation",
    "invalid_interval",
    "incompatible_media",
    "invalid_subtitle",
    "timeout",
    "ffmpeg_error",
    "output_validation_failed",
    "output_too_large",
    "corrupt_media",
    "internal_error",
]
ToolCapabilityName = Literal[
    "retrieval.read",
    "media.inspect",
    "media.decode",
    "media.write",
    "audio.write",
]


class ToolCapability(SchemaModel):
    capability: ToolCapabilityName
    description: NonEmptyStr
    read_only: bool


class ToolSpec(SchemaModel):
    name: ToolName
    version: NonEmptyStr
    description: NonEmptyStr
    capabilities: tuple[ToolCapabilityName, ...] = Field(min_length=1)
    argument_schema: dict[str, JsonValue]
    deterministic: bool
    produces_artifact: bool

    @field_validator("capabilities")
    @classmethod
    def unique_capabilities(
        cls, value: tuple[ToolCapabilityName, ...]
    ) -> tuple[ToolCapabilityName, ...]:
        if len(value) != len(set(value)):
            raise ValueError("tool capabilities must be unique")
        return value


class ToolExecutionContext(SchemaModel):
    """Execution policy with opaque IDs only; no host path is model-visible."""

    execution_id: Identifier
    allowed_output_root_id: Identifier
    allowed_artifact_ids: tuple[Identifier, ...]
    allowed_capabilities: tuple[ToolCapabilityName, ...]
    timeout_ms: int = Field(default=120_000, ge=1, le=3_600_000)
    maximum_output_bytes: int = Field(default=1_073_741_824, ge=1)

    @field_validator("allowed_artifact_ids", "allowed_capabilities")
    @classmethod
    def unique_policy_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("execution policy values must be unique")
        return value


class SearchVideoArgs(SchemaModel):
    query: NonEmptyStr
    top_k: int = Field(default=5, ge=1, le=20)
    video_id: Identifier | None = None
    approximate_time_range: TimeRange | None = None
    required_evidence_types: tuple[RetrievalEvidenceType, ...] = ()

    @field_validator("required_evidence_types")
    @classmethod
    def unique_evidence(
        cls, value: tuple[RetrievalEvidenceType, ...]
    ) -> tuple[RetrievalEvidenceType, ...]:
        if len(value) != len(set(value)):
            raise ValueError("required evidence types must be unique")
        return value


class InspectMediaArgs(SchemaModel):
    input_artifact_id: Identifier


class TrimVideoArgs(SchemaModel):
    input_artifact_id: Identifier
    time_range: TimeRange


class ConcatVideosArgs(SchemaModel):
    input_artifact_ids: tuple[Identifier, ...] = Field(min_length=2, max_length=32)

    @field_validator("input_artifact_ids")
    @classmethod
    def distinct_inputs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("concat inputs must be distinct artifacts")
        return value


class ChangeSpeedArgs(SchemaModel):
    input_artifact_id: Identifier
    speed_factor: float = Field(gt=0.24, le=4.0)


class SubtitleCue(SchemaModel):
    cue_id: Identifier
    time_range: TimeRange
    text: NonEmptyStr

    @field_validator("text")
    @classmethod
    def safe_text(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("subtitle text cannot contain NUL")
        return value


class SubtitleStyle(SchemaModel):
    font_size: int = Field(default=24, ge=8, le=96)
    alignment: Literal["bottom", "center", "top"] = "bottom"
    text_color: Literal["white", "yellow"] = "white"
    outline_color: Literal["black", "white"] = "black"


class AddSubtitlesArgs(SchemaModel):
    input_artifact_id: Identifier
    cues: tuple[SubtitleCue, ...] = Field(min_length=1, max_length=500)
    style: SubtitleStyle = Field(default_factory=SubtitleStyle)

    @field_validator("cues")
    @classmethod
    def ordered_non_overlapping(cls, value: tuple[SubtitleCue, ...]) -> tuple[SubtitleCue, ...]:
        ids = [item.cue_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("subtitle cue identifiers must be unique")
        for prior, current in pairwise(value):
            if current.time_range.start_ms < prior.time_range.end_ms:
                raise ValueError("subtitle cues must be ordered and non-overlapping")
        return value


class ReframeVideoArgs(SchemaModel):
    input_artifact_id: Identifier
    width: int = Field(ge=64, le=4096)
    height: int = Field(ge=64, le=4096)
    fit: Literal["crop", "pad"] = "crop"

    @model_validator(mode="after")
    def require_even_dimensions(self) -> ReframeVideoArgs:
        if self.width % 2 or self.height % 2:
            raise ValueError("encoded output dimensions must be even")
        return self


class NormalizeAudioArgs(SchemaModel):
    input_artifact_id: Identifier
    target_lufs: float = Field(default=-16.0, ge=-30.0, le=-5.0)
    loudness_range: float = Field(default=11.0, ge=1.0, le=20.0)
    true_peak_db: float = Field(default=-1.5, ge=-9.0, le=0.0)


class ValidateMediaArgs(SchemaModel):
    input_artifact_id: Identifier
    require_audio: bool = False
    decode_entire_video: bool = True


class SearchVideoCall(SchemaModel):
    tool_name: Literal["search_video"] = "search_video"
    tool_call_id: Identifier
    arguments: SearchVideoArgs


class InspectMediaCall(SchemaModel):
    tool_name: Literal["inspect_media"] = "inspect_media"
    tool_call_id: Identifier
    arguments: InspectMediaArgs


class TrimVideoCall(SchemaModel):
    tool_name: Literal["trim_video"] = "trim_video"
    tool_call_id: Identifier
    arguments: TrimVideoArgs


class ConcatVideosCall(SchemaModel):
    tool_name: Literal["concat_videos"] = "concat_videos"
    tool_call_id: Identifier
    arguments: ConcatVideosArgs


class ChangeSpeedCall(SchemaModel):
    tool_name: Literal["change_speed"] = "change_speed"
    tool_call_id: Identifier
    arguments: ChangeSpeedArgs


class AddSubtitlesCall(SchemaModel):
    tool_name: Literal["add_subtitles"] = "add_subtitles"
    tool_call_id: Identifier
    arguments: AddSubtitlesArgs


class ReframeVideoCall(SchemaModel):
    tool_name: Literal["reframe_video"] = "reframe_video"
    tool_call_id: Identifier
    arguments: ReframeVideoArgs


class NormalizeAudioCall(SchemaModel):
    tool_name: Literal["normalize_audio"] = "normalize_audio"
    tool_call_id: Identifier
    arguments: NormalizeAudioArgs


class ValidateMediaCall(SchemaModel):
    tool_name: Literal["validate_media"] = "validate_media"
    tool_call_id: Identifier
    arguments: ValidateMediaArgs


ToolCall = Annotated[
    SearchVideoCall
    | InspectMediaCall
    | TrimVideoCall
    | ConcatVideosCall
    | ChangeSpeedCall
    | AddSubtitlesCall
    | ReframeVideoCall
    | NormalizeAudioCall
    | ValidateMediaCall,
    Field(discriminator="tool_name"),
]
TOOL_CALL_ADAPTER: TypeAdapter[ToolCall] = TypeAdapter(ToolCall)


class ToolManifest(SchemaModel):
    manifest_id: Identifier
    registry_version: NonEmptyStr
    tools: tuple[ToolSpec, ...]
    capabilities: tuple[ToolCapability, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> ToolManifest:
        names = [item.name for item in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool manifest names must be unique")
        declared = {item.capability for item in self.capabilities}
        if any(set(item.capabilities) - declared for item in self.tools):
            raise ValueError("tool spec references an undeclared capability")
        return self


class TraceArtifact(SchemaModel):
    """Trace-safe artifact identity without a host URI."""

    artifact_id: Identifier
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: NonEmptyStr
    size_bytes: int = Field(ge=0)


class ToolValidationCheck(SchemaModel):
    check_name: Identifier
    passed: bool
    observed: JsonValue
    expected: JsonValue | None = None
    tolerance: float | None = Field(default=None, ge=0)


class ToolTrace(SchemaModel):
    trace_id: Identifier
    execution_id: Identifier
    tool_call_id: Identifier
    tool_name: Identifier
    tool_version: NonEmptyStr
    normalized_arguments: dict[str, JsonValue]
    parent_artifacts: tuple[TraceArtifact, ...]
    output_artifact: TraceArtifact | None = None
    started_at: datetime
    ended_at: datetime
    latency_ms: int = Field(ge=0)
    ffmpeg_return_code: int | None = None
    validation_results: tuple[ToolValidationCheck, ...]
    status: Literal["success", "invalid", "timeout", "error"]
    error_category: ToolErrorCategory | None = None
    cache_hit: bool
    cache_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("started_at", "ended_at")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("tool trace timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_trace_status(self) -> ToolTrace:
        if self.ended_at < self.started_at:
            raise ValueError("tool trace cannot end before it starts")
        if self.status == "success" and self.error_category is not None:
            raise ValueError("successful trace cannot have an error category")
        if self.status != "success" and self.error_category is None:
            raise ValueError("failed trace requires an error category")
        return self


class ToolExecutionRecord(SchemaModel):
    observation: ToolObservation
    trace: ToolTrace
