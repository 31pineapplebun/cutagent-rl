"""Deterministic FFmpeg scene-change baseline with complete coverage."""

import hashlib
import re
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from cutagent.ingestion.ffmpeg import FFmpegRunner
from cutagent.ingestion.timeline import Timeline, _divide, parse_rational
from cutagent.schemas.media import (
    IngestionConfig,
    SceneDetectionProvenance,
    SceneSegment,
    TimeRange,
    VideoAsset,
)

_SHOWINFO_TIME_BASE = re.compile(r"config in time_base:\s*(?P<value>\d+/\d+)")
_SHOWINFO_PTS = re.compile(r"\bpts:\s*(?P<pts>-?\d+)\s+pts_time:")


class SceneSplitter(Protocol):
    def split(
        self,
        source_path: Path,
        *,
        source_video: VideoAsset,
        analysis_video: VideoAsset,
        config: IngestionConfig,
        cache_key: str,
    ) -> tuple[SceneSegment, ...]: ...


def _filter_pts_to_normalized_ms(
    pts: int,
    *,
    filter_time_base_numerator: int,
    filter_time_base_denominator: int,
    video: VideoAsset,
) -> int:
    absolute_numerator = pts * filter_time_base_numerator
    absolute_denominator = filter_time_base_denominator
    source = video.video_stream
    origin = Fraction(
        source.source_start_pts * source.time_base.numerator,
        source.time_base.denominator,
    )
    delta = Fraction(absolute_numerator, absolute_denominator) - origin
    return _divide(delta.numerator * 1000, delta.denominator, "nearest")


def merge_scene_boundaries(
    boundaries_ms: list[int],
    *,
    duration_ms: int,
    minimum_scene_duration_ms: int,
) -> tuple[TimeRange, ...]:
    """Merge short neighbors deterministically and retain full timeline coverage."""

    candidates = sorted({value for value in boundaries_ms if 0 < value < duration_ms})
    starts = [0]
    for boundary in candidates:
        if boundary - starts[-1] >= minimum_scene_duration_ms:
            starts.append(boundary)
    if len(starts) > 1 and duration_ms - starts[-1] < minimum_scene_duration_ms:
        starts.pop()
    ends = [*starts[1:], duration_ms]
    return tuple(
        TimeRange(start_ms=start, end_ms=end) for start, end in zip(starts, ends, strict=True)
    )


class FFmpegSceneSplitter:
    """Use FFmpeg's `scene` score and decoded PTS; never infer time from FPS."""

    def __init__(self, runner: FFmpegRunner) -> None:
        self.runner = runner

    def _detect_boundaries(
        self,
        source_path: Path,
        *,
        video: VideoAsset,
        threshold: float,
    ) -> list[int]:
        select_filter = f"select=gt(scene\\,{threshold:.8f}),showinfo"
        completed = self.runner.run(
            (
                "-hide_banner",
                "-nostdin",
                "-copyts",
                "-i",
                str(source_path),
                "-map",
                f"0:{video.video_stream.stream_index}",
                "-vf",
                select_filter,
                "-an",
                "-vsync",
                "0",
                "-f",
                "null",
                "-",
            ),
            operation="scene detection",
        )
        time_base_match = _SHOWINFO_TIME_BASE.search(completed.stderr)
        if time_base_match is None:
            return []
        time_base = parse_rational(time_base_match.group("value"))
        if time_base is None:
            return []
        boundaries: list[int] = []
        for match in _SHOWINFO_PTS.finditer(completed.stderr):
            timestamp_ms = _filter_pts_to_normalized_ms(
                int(match.group("pts")),
                filter_time_base_numerator=time_base.numerator,
                filter_time_base_denominator=time_base.denominator,
                video=video,
            )
            boundaries.append(timestamp_ms)
        return boundaries

    def split(
        self,
        source_path: Path,
        *,
        source_video: VideoAsset,
        analysis_video: VideoAsset,
        config: IngestionConfig,
        cache_key: str,
    ) -> tuple[SceneSegment, ...]:
        boundaries = self._detect_boundaries(
            source_path,
            video=analysis_video,
            threshold=config.scene_threshold,
        )
        ranges = merge_scene_boundaries(
            boundaries,
            duration_ms=source_video.duration_ms,
            minimum_scene_duration_ms=config.minimum_scene_duration_ms,
        )
        source_timeline = Timeline(
            time_base=source_video.video_stream.time_base,
            origin_pts=source_video.video_stream.source_start_pts,
            duration_ms=source_video.duration_ms,
        )
        provenance = SceneDetectionProvenance(
            threshold=config.scene_threshold,
            minimum_scene_duration_ms=config.minimum_scene_duration_ms,
            ffmpeg_version=self.runner.version,
            cache_key=cache_key,
        )
        segments: list[SceneSegment] = []
        for index, time_range in enumerate(ranges):
            identity = hashlib.sha256(
                (
                    f"{source_video.source.sha256}:{cache_key}:{index}:"
                    f"{time_range.start_ms}:{time_range.end_ms}"
                ).encode("ascii")
            ).hexdigest()[:20]
            segments.append(
                SceneSegment(
                    segment_id=f"scene-{identity}",
                    parent_video_id=source_video.video_id,
                    time_range=time_range,
                    source_timestamps=source_timeline.source_range(time_range),
                    provenance=provenance,
                )
            )
        return tuple(segments)
