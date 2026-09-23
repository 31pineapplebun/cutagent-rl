"""End-to-end M1A video ingestion pipeline."""

import json
import mimetypes
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import JsonValue

from cutagent.core.artifacts import ArtifactRef
from cutagent.core.errors import CacheError
from cutagent.core.run_context import config_sha256
from cutagent.ingestion.cache import CacheKeyBuilder, ContentAddressedCache
from cutagent.ingestion.ffmpeg import FFmpegRunner
from cutagent.ingestion.ffprobe import FFprobeAdapter
from cutagent.ingestion.keyframes import FFmpegKeyframeExtractor
from cutagent.ingestion.normalization import MediaNormalizer
from cutagent.ingestion.scene_splitter import FFmpegSceneSplitter, SceneSplitter
from cutagent.schemas.media import (
    IngestionConfig,
    IngestionResult,
    KeyframeRef,
    NormalizationResult,
    OperationCacheRecord,
    SceneSegment,
    VideoAsset,
)


def _file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise CacheError("M1A cache currently requires file:// artifact URIs")
    return Path(url2pathname(parsed.path))


def _validate_cached_artifact(reference: ArtifactRef) -> None:
    path = _file_uri_path(reference.uri)
    if not path.is_file():
        raise CacheError(f"cached artifact is missing: {reference.artifact_id}")
    observed = ArtifactRef.from_path(
        path,
        artifact_id=reference.artifact_id,
        media_type=reference.media_type,
    )
    if observed.sha256 != reference.sha256 or observed.size_bytes != reference.size_bytes:
        raise CacheError(f"cached artifact digest mismatch: {reference.artifact_id}")


def _json_object(value: JsonValue | None, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CacheError(f"cached {label} must be a JSON object")
    return cast(dict[str, Any], value)


class VideoIngestionPipeline:
    """Probe, map, segment and sample immutable source video artifacts."""

    def __init__(
        self,
        *,
        cache_root: Path,
        ffmpeg_executable: str = "ffmpeg",
        ffprobe_executable: str = "ffprobe",
        scene_splitter: SceneSplitter | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.cache = ContentAddressedCache(cache_root)
        self.ffmpeg = FFmpegRunner(ffmpeg_executable, timeout_seconds=timeout_seconds)
        self.ffprobe = FFprobeAdapter(ffprobe_executable, timeout_seconds=timeout_seconds)
        self.scene_splitter = scene_splitter or FFmpegSceneSplitter(self.ffmpeg)
        self.normalizer = MediaNormalizer(self.ffmpeg, self.ffprobe)
        self.keyframe_extractor = FFmpegKeyframeExtractor(self.ffmpeg)

    def _probe_source(
        self, source_path: Path, source: ArtifactRef
    ) -> tuple[VideoAsset, OperationCacheRecord]:
        key = CacheKeyBuilder.build(
            source_sha256=source.sha256,
            operation="ffprobe",
            config={"output": "show_format+show_streams", "schema": "m1a-v1"},
            tool_versions={"ffprobe": self.ffprobe.version},
        )
        raw_path = self.cache.path("ffprobe", key, "raw.json")
        hit = raw_path.is_file()
        if hit:
            try:
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise CacheError("cached raw ffprobe JSON is invalid") from error
            if not isinstance(payload, dict):
                raise CacheError("cached raw ffprobe JSON must be an object")
            raw_ref = ArtifactRef.from_path(
                raw_path,
                artifact_id="ffprobe-provisional",
                media_type="application/json",
            )
            raw_ref = ArtifactRef(
                **{
                    **raw_ref.model_dump(),
                    "artifact_id": f"ffprobe-{raw_ref.sha256[:20]}",
                }
            )
            video = self.ffprobe.parse(
                cast(dict[str, Any], payload),
                source=source,
                raw_ffprobe=raw_ref,
            )
        else:
            video = self.ffprobe.probe(
                source_path,
                source=source,
                raw_output_path=raw_path,
            )
        return video, OperationCacheRecord(
            operation="ffprobe",
            cache_key=key,
            hit=hit,
            tool_version=self.ffprobe.version,
        )

    def _normalize(
        self,
        source_path: Path,
        source_video: VideoAsset,
        config: IngestionConfig,
    ) -> tuple[NormalizationResult, Path, VideoAsset, OperationCacheRecord]:
        key = CacheKeyBuilder.build(
            source_sha256=source_video.source.sha256,
            operation="normalization",
            config=config.normalization.model_dump(mode="json"),
            tool_versions={"ffmpeg": self.ffmpeg.version, "ffprobe": self.ffprobe.version},
        )
        cached = _json_object(
            self.cache.read_json("normalization", key, "result.json"),
            label="normalization result",
        )
        if cached is not None:
            result = NormalizationResult.model_validate(cached)
            if result.analysis_proxy is None:
                analysis_path = source_path
                analysis_video = source_video
            else:
                proxy = result.analysis_proxy
                _validate_cached_artifact(proxy.proxy)
                _validate_cached_artifact(proxy.raw_ffprobe)
                analysis_path = _file_uri_path(proxy.proxy.uri)
                raw_payload = json.loads(
                    _file_uri_path(proxy.raw_ffprobe.uri).read_text(encoding="utf-8")
                )
                if not isinstance(raw_payload, dict):
                    raise CacheError("proxy ffprobe artifact must contain an object")
                analysis_video = self.ffprobe.parse(
                    cast(dict[str, Any], raw_payload),
                    source=proxy.proxy,
                    raw_ffprobe=proxy.raw_ffprobe,
                    video_id=source_video.video_id,
                )
            hit = True
        else:
            result, analysis_path, analysis_video = self.normalizer.normalize(
                source_path,
                source_video=source_video,
                config=config,
                output_directory=self.cache.entry_dir("normalization", key),
            )
            self.cache.write_json(
                "normalization",
                key,
                "result.json",
                cast(JsonValue, result.model_dump(mode="json")),
            )
            hit = False
        return (
            result,
            analysis_path,
            analysis_video,
            OperationCacheRecord(
                operation="normalization",
                cache_key=key,
                hit=hit,
                tool_version=self.ffmpeg.version,
            ),
        )

    def _scenes(
        self,
        analysis_path: Path,
        *,
        source_video: VideoAsset,
        analysis_video: VideoAsset,
        config: IngestionConfig,
    ) -> tuple[tuple[SceneSegment, ...], OperationCacheRecord]:
        key = CacheKeyBuilder.build(
            source_sha256=source_video.source.sha256,
            operation="scene_split",
            config={
                "scene_threshold": config.scene_threshold,
                "minimum_scene_duration_ms": config.minimum_scene_duration_ms,
                "analysis_sha256": analysis_video.source.sha256,
                "backend": "ffmpeg_scene",
            },
            tool_versions={"ffmpeg": self.ffmpeg.version},
        )
        cached = self.cache.read_json("scene_split", key, "result.json")
        if cached is not None:
            if not isinstance(cached, list):
                raise CacheError("cached scene result must be a JSON list")
            scenes = tuple(SceneSegment.model_validate(item) for item in cached)
            hit = True
        else:
            scenes = self.scene_splitter.split(
                analysis_path,
                source_video=source_video,
                analysis_video=analysis_video,
                config=config,
                cache_key=key,
            )
            self.cache.write_json(
                "scene_split",
                key,
                "result.json",
                cast(JsonValue, [scene.model_dump(mode="json") for scene in scenes]),
            )
            hit = False
        return scenes, OperationCacheRecord(
            operation="scene_split",
            cache_key=key,
            hit=hit,
            tool_version=self.ffmpeg.version,
        )

    def _keyframes(
        self,
        analysis_path: Path,
        *,
        source_video: VideoAsset,
        analysis_video: VideoAsset,
        scenes: tuple[SceneSegment, ...],
        config: IngestionConfig,
    ) -> tuple[tuple[KeyframeRef, ...], OperationCacheRecord]:
        key = CacheKeyBuilder.build(
            source_sha256=source_video.source.sha256,
            operation="keyframe_extract",
            config={
                "keyframes": config.keyframes.model_dump(mode="json"),
                "analysis_sha256": analysis_video.source.sha256,
                "scenes": [
                    {
                        "segment_id": scene.segment_id,
                        "start_ms": scene.time_range.start_ms,
                        "end_ms": scene.time_range.end_ms,
                    }
                    for scene in scenes
                ],
            },
            tool_versions={
                "ffmpeg": self.ffmpeg.version,
                "cutagent_keyframe_extractor": "timestamp-select-v2",
            },
        )
        cached = self.cache.read_json("keyframe_extract", key, "result.json")
        if cached is not None:
            if not isinstance(cached, list):
                raise CacheError("cached keyframe result must be a JSON list")
            keyframes = tuple(KeyframeRef.model_validate(item) for item in cached)
            for keyframe in keyframes:
                _validate_cached_artifact(keyframe.artifact)
            hit = True
        else:
            keyframes = self.keyframe_extractor.extract(
                analysis_path,
                source_video=source_video,
                analysis_video=analysis_video,
                scenes=scenes,
                config=config.keyframes,
                cache_key=key,
                output_directory=self.cache.entry_dir("keyframe_extract", key),
            )
            self.cache.write_json(
                "keyframe_extract",
                key,
                "result.json",
                cast(JsonValue, [keyframe.model_dump(mode="json") for keyframe in keyframes]),
            )
            hit = False
        return keyframes, OperationCacheRecord(
            operation="keyframe_extract",
            cache_key=key,
            hit=hit,
            tool_version=self.ffmpeg.version,
        )

    def ingest(
        self,
        source_path: Path,
        *,
        config: IngestionConfig | None = None,
    ) -> IngestionResult:
        started = time.perf_counter_ns()
        resolved = source_path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("video source must be a regular file")
        media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        provisional = ArtifactRef.from_path(
            resolved,
            artifact_id="source-provisional",
            media_type=media_type,
        )
        source = ArtifactRef(
            **{
                **provisional.model_dump(),
                "artifact_id": f"source-{provisional.sha256[:20]}",
            }
        )
        effective_config = config or IngestionConfig()
        source_video, probe_record = self._probe_source(resolved, source)
        normalization, analysis_path, analysis_video, normalization_record = self._normalize(
            resolved,
            source_video,
            effective_config,
        )
        scenes, scene_record = self._scenes(
            analysis_path,
            source_video=source_video,
            analysis_video=analysis_video,
            config=effective_config,
        )
        keyframes, keyframe_record = self._keyframes(
            analysis_path,
            source_video=source_video,
            analysis_video=analysis_video,
            scenes=scenes,
            config=effective_config,
        )
        elapsed_ms = (time.perf_counter_ns() - started) // 1_000_000
        return IngestionResult(
            video=source_video,
            config=effective_config,
            config_sha256=config_sha256(effective_config.model_dump(mode="json")),
            normalization=normalization,
            scenes=scenes,
            keyframes=keyframes,
            cache_records=(probe_record, normalization_record, scene_record, keyframe_record),
            tool_versions={"ffmpeg": self.ffmpeg.version, "ffprobe": self.ffprobe.version},
            processing_time_ms=elapsed_ms,
        )
