"""Timestamp-based deterministic keyframe selection and extraction."""

import hashlib
import re
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

from cutagent.core.artifacts import ArtifactRef
from cutagent.core.errors import MediaProcessingError
from cutagent.ingestion.ffmpeg import FFmpegRunner
from cutagent.ingestion.scene_splitter import _filter_pts_to_normalized_ms
from cutagent.ingestion.timeline import _divide, parse_rational
from cutagent.schemas.media import (
    KeyframeExtractionConfig,
    KeyframeRef,
    SceneSegment,
    VideoAsset,
)

_SHOWINFO_TIME_BASE = re.compile(r"config in time_base:\s*(?P<value>\d+/\d+)")
_SHOWINFO_PTS = re.compile(r"\bpts:\s*(?P<pts>-?\d+)\s+pts_time:")


def requested_timestamps(scene: SceneSegment, config: KeyframeExtractionConfig) -> tuple[int, ...]:
    """Select interior millisecond timestamps without using frame indexes."""

    start = scene.time_range.start_ms
    duration = scene.time_range.duration_ms
    count = config.frames_per_scene
    return tuple(
        min(scene.time_range.end_ms - 1, start + ((2 * index + 1) * duration) // (2 * count))
        for index in range(count)
    )


def _seconds_text(value: Fraction) -> str:
    getcontext().prec = 24
    return format(Decimal(value.numerator) / Decimal(value.denominator), ".12f")


class FFmpegKeyframeExtractor:
    def __init__(self, runner: FFmpegRunner) -> None:
        self.runner = runner

    def _extract_one(
        self,
        source_path: Path,
        output_path: Path,
        *,
        analysis_video: VideoAsset,
        source_video: VideoAsset,
        requested_timestamp_ms: int,
        image_format: str,
    ) -> tuple[int | None, int | None]:
        stream = analysis_video.video_stream
        origin_seconds = Fraction(
            stream.source_start_pts * stream.time_base.numerator,
            stream.time_base.denominator,
        )
        target_seconds = origin_seconds + Fraction(requested_timestamp_ms, 1000)
        select_filter = f"select=gte(t\\,{_seconds_text(target_seconds)}),showinfo"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = output_path.with_name(f".{output_path.stem}.partial.{image_format}")
        partial_path.unlink(missing_ok=True)
        arguments = [
            "-hide_banner",
            "-nostdin",
            "-copyts",
            "-i",
            str(source_path),
            "-map",
            f"0:{stream.stream_index}",
            "-vf",
            select_filter,
            "-frames:v",
            "1",
            "-vsync",
            "0",
        ]
        if image_format == "jpg":
            arguments.extend(("-q:v", "2"))
        arguments.extend(("-y", str(partial_path)))
        try:
            completed = None
            use_last_observed_pts = False
            try:
                completed = self.runner.run(arguments, operation="keyframe extraction")
            except MediaProcessingError:
                # Some FFmpeg builds report an encoder initialization error instead
                # of a successful empty output when no frame satisfies ``gte``.
                partial_path.unlink(missing_ok=True)
            if not partial_path.is_file() or partial_path.stat().st_size == 0:
                # A valid timestamp can fall inside the display interval of the final
                # decoded frame, leaving no later PTS for ``gte`` to select.  Decode the
                # last frame at or before the requested timestamp as a deterministic
                # fallback. ``-update 1`` keeps only that final selected image.
                partial_path.unlink(missing_ok=True)
                fallback_filter = f"select=lte(t\\,{_seconds_text(target_seconds)}),showinfo"
                fallback_arguments = [
                    "-hide_banner",
                    "-nostdin",
                    "-copyts",
                    "-i",
                    str(source_path),
                    "-map",
                    f"0:{stream.stream_index}",
                    "-vf",
                    fallback_filter,
                    "-vsync",
                    "0",
                    "-update",
                    "1",
                ]
                if image_format == "jpg":
                    fallback_arguments.extend(("-q:v", "2"))
                fallback_arguments.extend(("-y", str(partial_path)))
                completed = self.runner.run(
                    fallback_arguments,
                    operation="keyframe extraction final-frame fallback",
                )
                use_last_observed_pts = True
            if not partial_path.is_file() or partial_path.stat().st_size == 0:
                raise MediaProcessingError("keyframe extraction produced no image")
            if completed is None:
                raise MediaProcessingError("keyframe extraction did not return process metadata")
            partial_path.replace(output_path)
        finally:
            partial_path.unlink(missing_ok=True)

        time_base_match = _SHOWINFO_TIME_BASE.search(completed.stderr)
        pts_matches = _SHOWINFO_PTS.findall(completed.stderr)
        if time_base_match is None or not pts_matches:
            return None, None
        filter_time_base = parse_rational(time_base_match.group("value"))
        if filter_time_base is None:
            return None, None
        observed_pts = int(pts_matches[-1] if use_last_observed_pts else pts_matches[0])
        observed_ms = _filter_pts_to_normalized_ms(
            observed_pts,
            filter_time_base_numerator=filter_time_base.numerator,
            filter_time_base_denominator=filter_time_base.denominator,
            video=analysis_video,
        )
        source_stream = source_video.video_stream
        absolute_seconds = Fraction(
            observed_pts * filter_time_base.numerator,
            filter_time_base.denominator,
        )
        source_tick = absolute_seconds / Fraction(
            source_stream.time_base.numerator,
            source_stream.time_base.denominator,
        )
        source_pts = _divide(source_tick.numerator, source_tick.denominator, "nearest")
        return max(0, observed_ms), source_pts

    def extract(
        self,
        source_path: Path,
        *,
        source_video: VideoAsset,
        analysis_video: VideoAsset,
        scenes: tuple[SceneSegment, ...],
        config: KeyframeExtractionConfig,
        cache_key: str,
        output_directory: Path,
    ) -> tuple[KeyframeRef, ...]:
        keyframes: list[KeyframeRef] = []
        for scene in scenes:
            for sample_index, requested_ms in enumerate(requested_timestamps(scene, config)):
                extension = config.image_format
                filename = f"{scene.segment_id}-{sample_index:03d}-{requested_ms:012d}.{extension}"
                output_path = output_directory / filename
                observed_ms, observed_pts = self._extract_one(
                    source_path,
                    output_path,
                    analysis_video=analysis_video,
                    source_video=source_video,
                    requested_timestamp_ms=requested_ms,
                    image_format=extension,
                )
                provisional_artifact = ArtifactRef.from_path(
                    output_path,
                    artifact_id="frame-provisional",
                    media_type="image/jpeg" if extension == "jpg" else "image/png",
                )
                artifact = ArtifactRef(
                    **{
                        **provisional_artifact.model_dump(),
                        "artifact_id": f"frame-{provisional_artifact.sha256[:20]}",
                    }
                )
                identity = hashlib.sha256(
                    (
                        f"{source_video.source.sha256}:{scene.segment_id}:{requested_ms}:"
                        f"{sample_index}:{cache_key}"
                    ).encode("ascii")
                ).hexdigest()[:20]
                keyframes.append(
                    KeyframeRef(
                        keyframe_id=f"keyframe-{identity}",
                        parent_video_id=source_video.video_id,
                        segment_id=scene.segment_id,
                        requested_timestamp_ms=requested_ms,
                        observed_timestamp_ms=observed_ms,
                        observed_source_pts=observed_pts,
                        artifact=artifact,
                        extraction_config=config,
                        ffmpeg_version=self.runner.version,
                        cache_key=cache_key,
                    )
                )
        return tuple(keyframes)
