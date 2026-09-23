"""M1A-to-M1B multimodal perception pipeline with heavy-model caching."""

from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import JsonValue

from cutagent.core.artifacts import ArtifactRef
from cutagent.core.errors import CacheError, PerceptionError
from cutagent.core.run_context import config_sha256
from cutagent.ingestion.cache import CacheKeyBuilder, ContentAddressedCache
from cutagent.ingestion.ffmpeg import FFmpegRunner
from cutagent.perception.backends.qwen import Qwen3VLVLMBackend
from cutagent.perception.backends.whisper import WhisperLargeV3TurboBackend
from cutagent.perception.ocr import VLMVisibleTextOCRBackend
from cutagent.perception.protocols import (
    ASRBackend,
    ASRBackendResult,
    ASRRequest,
    OCRBackend,
    VisualInput,
    VLMBackend,
    VLMBackendResult,
    VLMRequest,
)
from cutagent.perception.world_state import WorldStateBuilder
from cutagent.schemas.media import IngestionResult, SceneSegment, TimeRange
from cutagent.schemas.perception import (
    EvidenceDescriptor,
    EvidenceRef,
    ModelPerformance,
    OCRSpan,
    PerceptionCacheRecord,
    PerceptionConfig,
    PerceptionResult,
    TranscriptSpan,
    VisualObservation,
)


def _file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise PerceptionError("M1B local inference requires file:// artifact URIs")
    return Path(url2pathname(parsed.path)).resolve(strict=True)


def _artifact_from_path(path: Path, *, prefix: str, media_type: str) -> ArtifactRef:
    provisional = ArtifactRef.from_path(
        path,
        artifact_id=f"{prefix}-provisional",
        media_type=media_type,
    )
    return ArtifactRef(
        **{
            **provisional.model_dump(),
            "artifact_id": f"{prefix}-{provisional.sha256[:20]}",
        }
    )


def _validate_artifact(reference: ArtifactRef) -> None:
    path = _file_uri_path(reference.uri)
    observed = ArtifactRef.from_path(
        path,
        artifact_id=reference.artifact_id,
        media_type=reference.media_type,
    )
    if observed.sha256 != reference.sha256 or observed.size_bytes != reference.size_bytes:
        raise CacheError(f"cached artifact digest mismatch: {reference.artifact_id}")


def build_asr_cache_key(
    *, ingestion: IngestionResult, audio_sha256: str, config: PerceptionConfig, backend: str
) -> str:
    return CacheKeyBuilder.build(
        source_sha256=ingestion.video.source.sha256,
        operation="asr",
        config={
            "audio_sha256": audio_sha256,
            "model": config.whisper.model_dump(mode="json"),
            "language": config.asr_language,
            "silence_rms_threshold": config.asr_silence_rms_threshold,
            "timeline_contract": "m1a-audio-offset-v1",
        },
        tool_versions={"backend": backend},
    )


def build_vlm_cache_key(
    *,
    ingestion: IngestionResult,
    scene: SceneSegment,
    inputs: tuple[VisualInput, ...],
    config: PerceptionConfig,
    backend: str,
) -> str:
    return CacheKeyBuilder.build(
        source_sha256=ingestion.video.source.sha256,
        operation="vlm_perception",
        config={
            "scene": scene.model_dump(mode="json"),
            "input_artifacts": [
                {
                    "alias": item.evidence_alias,
                    "evidence_ref": item.evidence_ref.model_dump(mode="json"),
                    "sha256": ArtifactRef.from_path(
                        item.path,
                        artifact_id="cache-key",
                        media_type=mimetypes.guess_type(item.path.name)[0]
                        or "application/octet-stream",
                    ).sha256,
                }
                for item in inputs
            ],
            "visual_mode": config.visual_mode,
            "model": config.qwen.model_dump(mode="json"),
            "prompt_template_version": config.prompt_template_version,
            "temporal_prompt_style": config.temporal_prompt_style,
            "label_temporal_phases": config.label_temporal_phases,
            "maximum_repair_attempts": config.maximum_repair_attempts,
            "maximum_new_tokens": config.maximum_new_tokens,
            "native_video_fps": config.native_video_fps,
            "pixel_budgets": {
                "image": config.maximum_image_pixels,
                "video_frame": config.maximum_video_frame_pixels,
                "video_total": config.maximum_video_total_pixels,
            },
        },
        tool_versions={"backend": backend},
    )


class MultimodalPerceptionPipeline:
    """Run Whisper, Qwen, OCR projection, and deterministic world-state assembly."""

    def __init__(
        self,
        *,
        cache_root: Path,
        model_cache: Path | None = None,
        asr_backend: ASRBackend | None = None,
        vlm_backend: VLMBackend | None = None,
        ocr_backend: OCRBackend | None = None,
        ffmpeg_executable: str = "ffmpeg",
        timeout_seconds: int = 600,
    ) -> None:
        if (asr_backend is None or vlm_backend is None) and model_cache is None:
            raise ValueError("model_cache is required for primary real model backends")
        self.cache = ContentAddressedCache(cache_root)
        self.ffmpeg = FFmpegRunner(ffmpeg_executable, timeout_seconds=timeout_seconds)
        cache = "" if model_cache is None else str(model_cache.resolve())
        self.asr = asr_backend or WhisperLargeV3TurboBackend(model_cache=cache)
        self.vlm = vlm_backend or Qwen3VLVLMBackend(model_cache=cache)
        self.ocr = ocr_backend or VLMVisibleTextOCRBackend()
        self.world_state_builder = WorldStateBuilder()

    def _extract_audio(
        self, source_path: Path, ingestion: IngestionResult
    ) -> tuple[ArtifactRef, int, PerceptionCacheRecord] | None:
        if not ingestion.video.audio_streams:
            return None
        audio = ingestion.video.audio_streams[0]
        key = CacheKeyBuilder.build(
            source_sha256=ingestion.video.source.sha256,
            operation="audio_extract",
            config={
                "stream_index": audio.stream_index,
                "sample_rate": 16000,
                "channels": 1,
                "codec": "pcm_s16le",
            },
            tool_versions={"ffmpeg": self.ffmpeg.version},
        )
        output = self.cache.path("audio_extract", key, "source_audio.wav")
        hit = output.is_file()
        if not hit:
            self.ffmpeg.run(
                (
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_path),
                    "-map",
                    f"0:{audio.stream_index}",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(output),
                ),
                operation="M1B source-audio extraction",
            )
        artifact = _artifact_from_path(output, prefix="audio-pcm", media_type="audio/wav")
        offset_ms = audio.source_start_time_ms - ingestion.video.video_stream.source_start_time_ms
        return (
            artifact,
            offset_ms,
            PerceptionCacheRecord(operation="audio_extract", cache_key=key, hit=hit),
        )

    def _asr(
        self,
        *,
        ingestion: IngestionResult,
        audio: ArtifactRef,
        offset_ms: int,
        config: PerceptionConfig,
    ) -> tuple[ASRBackendResult, PerceptionCacheRecord]:
        key = build_asr_cache_key(
            ingestion=ingestion,
            audio_sha256=audio.sha256,
            config=config,
            backend=self.asr.backend_version,
        )
        cached = self.cache.read_json("asr", key, "result.json")
        if cached is not None:
            if not isinstance(cached, dict):
                raise CacheError("cached ASR result must be an object")
            spans = tuple(
                TranscriptSpan.model_validate(item) for item in cast(list[Any], cached["spans"])
            )
            raw_output = ArtifactRef.model_validate(cached["raw_output"])
            performance = ModelPerformance.model_validate(cached["performance"])
            _validate_artifact(raw_output)
            result = ASRBackendResult(
                spans=spans,
                raw_output=raw_output,
                performance=performance,
            )
            hit = True
        else:
            result = self.asr.transcribe(
                ASRRequest(
                    audio_path=_file_uri_path(audio.uri),
                    audio_artifact=audio,
                    video=ingestion.video,
                    timeline_offset_ms=offset_ms,
                    output_directory=self.cache.entry_dir("asr", key),
                    config=config,
                )
            )
            self.cache.write_json(
                "asr",
                key,
                "result.json",
                cast(
                    JsonValue,
                    {
                        "spans": [span.model_dump(mode="json") for span in result.spans],
                        "raw_output": result.raw_output.model_dump(mode="json"),
                        "performance": result.performance.model_dump(mode="json"),
                    },
                ),
            )
            hit = False
        spec = config.whisper
        return result, PerceptionCacheRecord(
            operation="asr",
            cache_key=key,
            hit=hit,
            model_id=spec.model_id,
            model_revision=spec.revision,
        )

    def _scene_clip(
        self, source_path: Path, ingestion: IngestionResult, scene: SceneSegment
    ) -> tuple[ArtifactRef, PerceptionCacheRecord]:
        key = CacheKeyBuilder.build(
            source_sha256=ingestion.video.source.sha256,
            operation="scene_clip",
            config={
                "segment_id": scene.segment_id,
                "time_range": scene.time_range.model_dump(mode="json"),
                "codec": "mpeg4",
                "pixel_format": "yuv420p",
                "audio": False,
                "timestamp_mapping": "clip_zero_maps_to_scene_start_ms",
            },
            tool_versions={"ffmpeg": self.ffmpeg.version},
        )
        output = self.cache.path("scene_clip", key, "scene.mp4")
        hit = output.is_file()
        if not hit:
            start = f"{scene.time_range.start_ms / 1000:.3f}"
            duration = f"{scene.time_range.duration_ms / 1000:.3f}"
            self.ffmpeg.run(
                (
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    start,
                    "-i",
                    str(source_path),
                    "-t",
                    duration,
                    "-an",
                    "-c:v",
                    "mpeg4",
                    "-q:v",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    str(output),
                ),
                operation=f"M1B scene clip {scene.segment_id}",
            )
        artifact = _artifact_from_path(output, prefix="scene-clip", media_type="video/mp4")
        return artifact, PerceptionCacheRecord(operation="scene_clip", cache_key=key, hit=hit)

    def _visual_inputs(
        self,
        *,
        source_path: Path,
        ingestion: IngestionResult,
        scene: SceneSegment,
        config: PerceptionConfig,
    ) -> tuple[tuple[VisualInput, ...], tuple[ArtifactRef, ...], tuple[PerceptionCacheRecord, ...]]:
        if config.visual_mode == "keyframes":
            items = tuple(
                VisualInput(
                    evidence_alias=f"keyframe_{index + 1}",
                    path=_file_uri_path(keyframe.artifact.uri),
                    evidence_ref=EvidenceRef(
                        artifact_id=keyframe.artifact.artifact_id,
                        evidence_kind="keyframe",
                        segment_id=scene.segment_id,
                        observed_ms=(
                            keyframe.observed_timestamp_ms
                            if keyframe.observed_timestamp_ms is not None
                            else keyframe.requested_timestamp_ms
                        ),
                    ),
                )
                for index, keyframe in enumerate(
                    item for item in ingestion.keyframes if item.segment_id == scene.segment_id
                )
            )
            if not items:
                raise PerceptionError(f"scene has no keyframe evidence: {scene.segment_id}")
            return items, (), ()
        clip, record = self._scene_clip(source_path, ingestion, scene)
        return (
            (
                VisualInput(
                    evidence_alias="scene_clip",
                    path=_file_uri_path(clip.uri),
                    evidence_ref=EvidenceRef(
                        artifact_id=clip.artifact_id,
                        evidence_kind="scene_clip",
                        segment_id=scene.segment_id,
                        time_range=scene.time_range,
                    ),
                ),
            ),
            (clip,),
            (record,),
        )

    def _vlm(
        self,
        *,
        ingestion: IngestionResult,
        scene: SceneSegment,
        inputs: tuple[VisualInput, ...],
        config: PerceptionConfig,
    ) -> tuple[VLMBackendResult, PerceptionCacheRecord]:
        key = build_vlm_cache_key(
            ingestion=ingestion,
            scene=scene,
            inputs=inputs,
            config=config,
            backend=self.vlm.backend_version,
        )
        cached = self.cache.read_json("vlm_perception", key, "result.json")
        if cached is not None:
            if not isinstance(cached, dict):
                raise CacheError("cached VLM result must be an object")
            result = VLMBackendResult(
                observation=VisualObservation.model_validate(cached["observation"]),
                raw_output=ArtifactRef.model_validate(cached["raw_output"]),
                performance=ModelPerformance.model_validate(cached["performance"]),
            )
            _validate_artifact(result.raw_output)
            hit = True
        else:
            result = self.vlm.perceive(
                VLMRequest(
                    video=ingestion.video,
                    scene=scene,
                    visual_inputs=inputs,
                    output_directory=self.cache.entry_dir("vlm_perception", key),
                    config=config,
                )
            )
            self.cache.write_json(
                "vlm_perception",
                key,
                "result.json",
                cast(
                    JsonValue,
                    {
                        "observation": result.observation.model_dump(mode="json"),
                        "raw_output": result.raw_output.model_dump(mode="json"),
                        "performance": result.performance.model_dump(mode="json"),
                    },
                ),
            )
            hit = False
        spec = config.qwen
        return result, PerceptionCacheRecord(
            operation="vlm_perception",
            cache_key=key,
            hit=hit,
            model_id=spec.model_id,
            model_revision=spec.revision,
        )

    def run(
        self,
        source_path: Path,
        *,
        ingestion: IngestionResult,
        config: PerceptionConfig | None = None,
    ) -> PerceptionResult:
        started = time.perf_counter_ns()
        source = source_path.resolve(strict=True)
        observed_source = ArtifactRef.from_path(
            source,
            artifact_id=ingestion.video.source.artifact_id,
            media_type=ingestion.video.source.media_type,
        )
        if observed_source.sha256 != ingestion.video.source.sha256:
            raise PerceptionError("source path does not match M1A ingestion artifact")
        effective = config or PerceptionConfig()
        cache_records: list[PerceptionCacheRecord] = []
        provenance: list[ArtifactRef] = []
        performance: list[ModelPerformance] = []
        transcripts: tuple[TranscriptSpan, ...] = ()
        audio_result = self._extract_audio(source, ingestion) if effective.asr_enabled else None
        if audio_result is not None:
            audio, offset_ms, audio_record = audio_result
            cache_records.append(audio_record)
            provenance.append(audio)
            asr_result, asr_record = self._asr(
                ingestion=ingestion,
                audio=audio,
                offset_ms=offset_ms,
                config=effective,
            )
            transcripts = asr_result.spans
            provenance.append(asr_result.raw_output)
            performance.append(asr_result.performance)
            cache_records.append(asr_record)

        observations: list[VisualObservation] = []
        visual_evidence: list[VisualInput] = []
        for scene in ingestion.scenes:
            inputs, artifacts, records = self._visual_inputs(
                source_path=source,
                ingestion=ingestion,
                scene=scene,
                config=effective,
            )
            visual_evidence.extend(inputs)
            provenance.extend(artifacts)
            cache_records.extend(records)
            vlm_result, vlm_record = self._vlm(
                ingestion=ingestion,
                scene=scene,
                inputs=inputs,
                config=effective,
            )
            observations.append(vlm_result.observation)
            provenance.append(vlm_result.raw_output)
            performance.append(vlm_result.performance)
            cache_records.append(vlm_record)

        observation_tuple = tuple(observations)
        ocr_spans: tuple[OCRSpan, ...] = self.ocr.extract(observation_tuple)
        events = tuple(
            event for observation in observations for event in observation.temporal_events
        )
        catalog: list[EvidenceDescriptor] = [
            EvidenceDescriptor(
                artifact_id=ingestion.video.source.artifact_id,
                evidence_kind="source_media",
                media_type=ingestion.video.source.media_type,
                time_range=TimeRange(start_ms=0, end_ms=ingestion.video.duration_ms),
            )
        ]
        catalog.extend(
            EvidenceDescriptor(
                artifact_id=keyframe.artifact.artifact_id,
                evidence_kind="keyframe",
                media_type=keyframe.artifact.media_type,
                segment_id=keyframe.segment_id,
                observed_ms=(
                    keyframe.observed_timestamp_ms
                    if keyframe.observed_timestamp_ms is not None
                    else keyframe.requested_timestamp_ms
                ),
            )
            for keyframe in ingestion.keyframes
        )
        for artifact in provenance:
            if artifact.artifact_id.startswith("audio-pcm-"):
                catalog.append(
                    EvidenceDescriptor(
                        artifact_id=artifact.artifact_id,
                        evidence_kind="source_audio_pcm",
                        media_type=artifact.media_type,
                        time_range=TimeRange(start_ms=0, end_ms=ingestion.video.duration_ms),
                    )
                )
            elif artifact.artifact_id.startswith("scene-clip-"):
                matching = next(
                    item.evidence_ref
                    for item in visual_evidence
                    if item.evidence_ref.artifact_id == artifact.artifact_id
                )
                catalog.append(
                    EvidenceDescriptor(
                        **matching.model_dump(),
                        media_type=artifact.media_type,
                    )
                )
            elif artifact.artifact_id.startswith("asr-raw-"):
                catalog.append(
                    EvidenceDescriptor(
                        artifact_id=artifact.artifact_id,
                        evidence_kind="asr_raw_output",
                        media_type=artifact.media_type,
                    )
                )
            elif artifact.artifact_id.startswith("vlm-raw-"):
                catalog.append(
                    EvidenceDescriptor(
                        artifact_id=artifact.artifact_id,
                        evidence_kind="vlm_raw_output",
                        media_type=artifact.media_type,
                    )
                )
        world_state = self.world_state_builder.build(
            ingestion=ingestion,
            transcripts=transcripts,
            observations=observation_tuple,
            ocr_spans=ocr_spans,
            temporal_events=events,
            evidence_catalog=tuple(catalog),
        )
        elapsed_ms = (time.perf_counter_ns() - started) // 1_000_000
        return PerceptionResult(
            source_video_id=ingestion.video.video_id,
            config=effective,
            config_sha256=config_sha256(effective.model_dump(mode="json")),
            transcript_spans=transcripts,
            visual_observations=observation_tuple,
            ocr_spans=ocr_spans,
            temporal_events=events,
            world_state=world_state,
            provenance_artifacts=tuple(provenance),
            cache_records=tuple(cache_records),
            performance=tuple(performance),
            runtime_versions={
                "ffmpeg": self.ffmpeg.version,
                "asr_backend": self.asr.backend_version,
                "vlm_backend": self.vlm.backend_version,
                "ocr_backend": self.ocr.backend_version,
                "model_runtime": self.vlm.backend_version,
            },
            processing_time_ms=elapsed_ms,
            metadata={
                "audio_source": "immutable_source_media",
                "audio_video_offset_contract": "audio_start_ms-video_start_ms",
                "model_outputs_are_observations_not_ground_truth": True,
            },
        )
