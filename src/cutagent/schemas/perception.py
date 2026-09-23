"""Public, evidence-linked contracts for M1B multimodal perception."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.media import TimeRange

EvidenceKind = Literal[
    "source_media",
    "source_audio_pcm",
    "keyframe",
    "scene_clip",
    "asr_raw_output",
    "vlm_raw_output",
]
PerceptionMode = Literal["keyframes", "native_video"]
TemporalPromptStyle = Literal["baseline", "explicit_comparison"]
PerceptionFailureType = Literal[
    "missed_entity",
    "hallucinated_entity",
    "wrong_action",
    "wrong_visible_text",
    "temporal_misalignment",
    "unsupported_event",
    "asr_substitution",
    "asr_deletion",
    "asr_insertion",
    "malformed_structured_output",
]


class EvidenceRef(SchemaModel):
    """A claim-local pointer to an entry in the world-state evidence catalog."""

    artifact_id: Identifier
    evidence_kind: EvidenceKind
    segment_id: Identifier | None = None
    observed_ms: int | None = Field(default=None, ge=0)
    time_range: TimeRange | None = None

    @model_validator(mode="after")
    def validate_temporal_selector(self) -> EvidenceRef:
        if self.observed_ms is not None and self.time_range is not None:
            raise ValueError("evidence cannot have both observed_ms and time_range")
        return self


class EvidenceDescriptor(EvidenceRef):
    """Policy-safe evidence catalog entry; hashes and URIs remain private."""

    media_type: NonEmptyStr


class TranscriptSpan(SchemaModel):
    span_id: Identifier
    text: NonEmptyStr
    time_range: TimeRange
    language: NonEmptyStr | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_refs: tuple[EvidenceRef, ...] = Field(min_length=1)


class BoundingBox(SchemaModel):
    """Normalized [0, 1] left/top/right/bottom box."""

    left: float = Field(ge=0, le=1)
    top: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)
    bottom: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_order(self) -> BoundingBox:
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("bounding box must have positive width and height")
        return self


class OCRSpan(SchemaModel):
    span_id: Identifier
    exact_text: NonEmptyStr
    normalized_text: NonEmptyStr | None = None
    text_source: Literal["directly_visible_text"] = "directly_visible_text"
    segment_id: Identifier
    observed_ms: int | None = Field(default=None, ge=0)
    time_range: TimeRange | None = None
    bounding_box: BoundingBox | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_refs: tuple[EvidenceRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_time(self) -> OCRSpan:
        if (self.observed_ms is None) == (self.time_range is None):
            raise ValueError("OCR span requires exactly one of observed_ms or time_range")
        return self


class VisualEntity(SchemaModel):
    label: NonEmptyStr
    attributes: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    evidence_refs: tuple[EvidenceRef, ...] = Field(min_length=1)

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        normalized = " ".join(value.casefold().split())
        if not normalized:
            raise ValueError("entity label cannot normalize to empty")
        return normalized


class VisualAction(SchemaModel):
    subject: NonEmptyStr
    action: NonEmptyStr
    object: NonEmptyStr | None = None
    evidence_refs: tuple[EvidenceRef, ...] = Field(min_length=1)


class DirectlyVisibleText(SchemaModel):
    exact_text: NonEmptyStr
    normalized_text: NonEmptyStr | None = None
    evidence_refs: tuple[EvidenceRef, ...] = Field(min_length=1)


class TemporalEvent(SchemaModel):
    event_id: Identifier
    description: NonEmptyStr
    time_range: TimeRange
    evidence_refs: tuple[EvidenceRef, ...] = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    uncertainty: NonEmptyStr | None = None


class VisualObservation(SchemaModel):
    observation_id: Identifier
    segment_id: Identifier
    mode: PerceptionMode
    time_range: TimeRange
    scene_summary: NonEmptyStr | None = None
    summary_evidence_refs: tuple[EvidenceRef, ...] = ()
    entities: tuple[VisualEntity, ...] = ()
    actions: tuple[VisualAction, ...] = ()
    directly_visible_text: tuple[DirectlyVisibleText, ...] = ()
    inferred_semantic_text: tuple[NonEmptyStr, ...] = ()
    temporal_events: tuple[TemporalEvent, ...] = ()
    uncertainties: tuple[NonEmptyStr, ...] = ()
    repair_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_summary_evidence(self) -> VisualObservation:
        if self.scene_summary is not None and not self.summary_evidence_refs:
            raise ValueError("scene summary requires evidence")
        if self.scene_summary is None and self.summary_evidence_refs:
            raise ValueError("summary evidence requires a scene summary")
        for event in self.temporal_events:
            if (
                event.time_range.start_ms < self.time_range.start_ms
                or event.time_range.end_ms > self.time_range.end_ms
            ):
                raise ValueError("temporal event lies outside its scene")
        return self


class ScenePerception(SchemaModel):
    segment_id: Identifier
    time_range: TimeRange
    keyframe_evidence: tuple[EvidenceRef, ...]
    transcript_spans: tuple[TranscriptSpan, ...]
    ocr_spans: tuple[OCRSpan, ...]
    entities: tuple[VisualEntity, ...]
    actions: tuple[VisualAction, ...]
    temporal_events: tuple[TemporalEvent, ...]
    scene_summary: NonEmptyStr | None = None
    summary_evidence_refs: tuple[EvidenceRef, ...] = ()
    visual_observation_ids: tuple[Identifier, ...] = ()


class VideoWorldState(SchemaModel):
    """Policy-safe structured video state with a closed evidence catalog."""

    video_id: Identifier
    duration_ms: int = Field(gt=0)
    evidence_catalog: tuple[EvidenceDescriptor, ...] = Field(min_length=1)
    scenes: tuple[ScenePerception, ...] = Field(min_length=1)
    global_summary: NonEmptyStr | None = None
    global_summary_evidence_refs: tuple[EvidenceRef, ...] = ()

    @model_validator(mode="after")
    def validate_world(self) -> VideoWorldState:
        def evidence_key(reference: EvidenceRef) -> tuple[object, ...]:
            return (
                reference.artifact_id,
                reference.evidence_kind,
                reference.segment_id,
                reference.observed_ms,
                None
                if reference.time_range is None
                else (reference.time_range.start_ms, reference.time_range.end_ms),
            )

        catalog = {evidence_key(entry): entry for entry in self.evidence_catalog}
        if len(catalog) != len(self.evidence_catalog):
            raise ValueError("evidence catalog occurrences must be unique")
        catalog_by_artifact: dict[tuple[str, str], list[EvidenceDescriptor]] = {}
        for entry in self.evidence_catalog:
            catalog_by_artifact.setdefault((entry.artifact_id, entry.evidence_kind), []).append(
                entry
            )

        def descriptor_supports(descriptor: EvidenceDescriptor, reference: EvidenceRef) -> bool:
            if descriptor.segment_id is not None and descriptor.segment_id != reference.segment_id:
                return False
            if reference.observed_ms is not None:
                if descriptor.observed_ms is not None:
                    return descriptor.observed_ms == reference.observed_ms
                if descriptor.time_range is not None:
                    return (
                        descriptor.time_range.start_ms
                        <= reference.observed_ms
                        < descriptor.time_range.end_ms
                    )
            if reference.time_range is not None:
                if descriptor.time_range is not None:
                    return (
                        descriptor.time_range.start_ms <= reference.time_range.start_ms
                        and descriptor.time_range.end_ms >= reference.time_range.end_ms
                    )
                if descriptor.observed_ms is not None:
                    return False
            return True

        expected_start = 0
        segment_ids: set[str] = set()
        references: list[EvidenceRef] = list(self.global_summary_evidence_refs)
        for scene in self.scenes:
            if scene.segment_id in segment_ids:
                raise ValueError("scene identifiers must be unique")
            segment_ids.add(scene.segment_id)
            if scene.time_range.start_ms != expected_start:
                raise ValueError("world-state scenes must be ordered and contiguous")
            expected_start = scene.time_range.end_ms
            references.extend(scene.keyframe_evidence)
            references.extend(scene.summary_evidence_refs)
            for transcript in scene.transcript_spans:
                references.extend(transcript.evidence_refs)
            for ocr_span in scene.ocr_spans:
                references.extend(ocr_span.evidence_refs)
            for entity in scene.entities:
                references.extend(entity.evidence_refs)
            for action in scene.actions:
                references.extend(action.evidence_refs)
            for event in scene.temporal_events:
                references.extend(event.evidence_refs)
        if expected_start != self.duration_ms:
            raise ValueError("world-state scenes must cover the video timeline")
        if self.global_summary is not None and not self.global_summary_evidence_refs:
            raise ValueError("global summary requires scene-level evidence")
        if self.global_summary is None and self.global_summary_evidence_refs:
            raise ValueError("global summary evidence requires a summary")
        for reference in references:
            candidates = catalog_by_artifact.get(
                (reference.artifact_id, reference.evidence_kind), ()
            )
            if not any(descriptor_supports(candidate, reference) for candidate in candidates):
                raise ValueError(
                    f"claim references unknown evidence occurrence: {reference.artifact_id}"
                )
            if reference.observed_ms is not None and reference.observed_ms >= self.duration_ms:
                raise ValueError("evidence timestamp exceeds video duration")
            if reference.time_range is not None and reference.time_range.end_ms > self.duration_ms:
                raise ValueError("evidence interval exceeds video duration")
        return self


class QwenModelSpec(SchemaModel):
    model_id: Literal["Qwen/Qwen3-VL-4B-Instruct"] = "Qwen/Qwen3-VL-4B-Instruct"
    revision: Literal["ebb281ec70b05090aa6165b016eac8ec08e71b17"] = (
        "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    )
    license: Literal["apache-2.0"] = "apache-2.0"
    dtype: Literal["bfloat16"] = "bfloat16"


class WhisperModelSpec(SchemaModel):
    model_id: Literal["openai/whisper-large-v3-turbo"] = "openai/whisper-large-v3-turbo"
    revision: Literal["41f01f3fe87f28c78e2fbf8b568835947dd65ed9"] = (
        "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    )
    license: Literal["mit"] = "mit"
    dtype: Literal["float16"] = "float16"


class PerceptionConfig(SchemaModel):
    qwen: QwenModelSpec = Field(default_factory=QwenModelSpec)
    whisper: WhisperModelSpec = Field(default_factory=WhisperModelSpec)
    visual_mode: PerceptionMode = "keyframes"
    prompt_template_version: NonEmptyStr = "m1b-structured-v1.2"
    temporal_prompt_style: TemporalPromptStyle = "baseline"
    label_temporal_phases: bool = False
    maximum_repair_attempts: int = Field(default=1, ge=0, le=2)
    maximum_new_tokens: int = Field(default=384, ge=64, le=1024)
    native_video_fps: float = Field(default=2.0, gt=0, le=8)
    maximum_image_pixels: int = Field(default=512 * 32 * 32, gt=0)
    maximum_video_frame_pixels: int = Field(default=256 * 32 * 32, gt=0)
    maximum_video_total_pixels: int = Field(default=8192 * 32 * 32, gt=0)
    asr_enabled: bool = True
    asr_language: NonEmptyStr | None = None
    asr_silence_rms_threshold: float = Field(default=1e-5, ge=0, le=1)


class ModelPerformance(SchemaModel):
    operation: Identifier
    model_id: NonEmptyStr
    model_revision: NonEmptyStr
    dtype: NonEmptyStr
    device: NonEmptyStr
    latency_ms: int = Field(ge=0)
    peak_allocated_bytes: int | None = Field(default=None, ge=0)
    peak_reserved_bytes: int | None = Field(default=None, ge=0)
    frames: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    repair_count: int = Field(default=0, ge=0)


class PerceptionCacheRecord(SchemaModel):
    operation: Literal["audio_extract", "asr", "scene_clip", "vlm_perception"]
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    hit: bool
    model_id: NonEmptyStr | None = None
    model_revision: NonEmptyStr | None = None


class PerceptionResult(SchemaModel):
    source_video_id: Identifier
    config: PerceptionConfig
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcript_spans: tuple[TranscriptSpan, ...]
    visual_observations: tuple[VisualObservation, ...]
    ocr_spans: tuple[OCRSpan, ...]
    temporal_events: tuple[TemporalEvent, ...]
    world_state: VideoWorldState
    provenance_artifacts: tuple[ArtifactRef, ...]
    cache_records: tuple[PerceptionCacheRecord, ...]
    performance: tuple[ModelPerformance, ...]
    runtime_versions: dict[NonEmptyStr, NonEmptyStr]
    processing_time_ms: int = Field(ge=0)
    metadata: dict[NonEmptyStr, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_video(self) -> PerceptionResult:
        if self.world_state.video_id != self.source_video_id:
            raise ValueError("world state and perception result video IDs differ")
        return self
