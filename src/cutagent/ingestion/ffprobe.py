"""Real ffprobe JSON adapter and media metadata parser."""

import hashlib
import json
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from cutagent.core.artifacts import ArtifactRef
from cutagent.core.errors import CacheError, MediaProbeError, MediaProbeTimeoutError
from cutagent.ingestion.timeline import Timeline, _divide, decimal_seconds_to_ms, parse_rational
from cutagent.schemas.media import (
    AudioStreamInfo,
    RationalValue,
    VideoAsset,
    VideoStreamInfo,
)


def _required_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip() or value == "N/A":
        raise MediaProbeError(f"ffprobe field {key!r} is missing or invalid")
    return value.strip()


def _optional_int(value: object) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        return int(cast(str | int, value))
    except (TypeError, ValueError) as error:
        raise MediaProbeError(f"invalid integer reported by ffprobe: {value!r}") from error


def _duration_ms(stream: dict[str, Any], time_base: RationalValue) -> int | None:
    duration_ts = _optional_int(stream.get("duration_ts"))
    if duration_ts is not None and duration_ts > 0:
        return Timeline(time_base=time_base, origin_pts=0, duration_ms=1).pts_to_ms(duration_ts)
    duration = stream.get("duration")
    if isinstance(duration, str) and duration not in {"", "N/A"}:
        value = decimal_seconds_to_ms(duration)
        return value if value > 0 else None
    tags = stream.get("tags")
    if isinstance(tags, dict):
        tagged_duration = tags.get("DURATION") or tags.get("duration")
        if isinstance(tagged_duration, str):
            parts = tagged_duration.split(":")
            if len(parts) == 3:
                try:
                    total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + Fraction(parts[2])
                except (ValueError, ZeroDivisionError):
                    pass
                else:
                    value = _divide(
                        total_seconds.numerator * 1000,
                        total_seconds.denominator,
                        "nearest",
                    )
                    return value if value > 0 else None
    return None


def _start_fields(stream: dict[str, Any], time_base: RationalValue) -> tuple[int, int]:
    start_time = stream.get("start_time")
    start_time_ms = 0
    if isinstance(start_time, str) and start_time not in {"", "N/A"}:
        start_time_ms = decimal_seconds_to_ms(start_time)
    start_pts = _optional_int(stream.get("start_pts"))
    if start_pts is None:
        start_pts = Timeline(time_base=time_base, origin_pts=0, duration_ms=1).ms_to_pts(
            start_time_ms
        )
    return start_pts, start_time_ms


def _rotation(stream: dict[str, Any]) -> int | None:
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict) and item.get("rotation") is not None:
                try:
                    return round(float(item["rotation"]))
                except (TypeError, ValueError):
                    continue
    tags = stream.get("tags")
    if isinstance(tags, dict) and tags.get("rotate") is not None:
        try:
            return round(float(tags["rotate"]))
        except (TypeError, ValueError):
            return None
    return None


def _vfr_status(
    average: RationalValue | None, real: RationalValue | None
) -> tuple[bool | None, tuple[str, ...]]:
    if average is None or real is None:
        return None, ("insufficient_rate_metadata",)
    average_fraction = Fraction(average.numerator, average.denominator)
    real_fraction = Fraction(real.numerator, real.denominator)
    if average_fraction != real_fraction:
        return True, ("avg_frame_rate_differs_from_r_frame_rate",)
    return False, ()


def _parse_video_stream(stream: dict[str, Any]) -> VideoStreamInfo:
    time_base = parse_rational(_required_text(stream, "time_base"))
    if time_base is None:
        raise MediaProbeError("video time_base must be a positive rational")
    average = parse_rational(cast(str | None, stream.get("avg_frame_rate")))
    real = parse_rational(cast(str | None, stream.get("r_frame_rate")))
    variable, indicators = _vfr_status(average, real)
    start_pts, start_time_ms = _start_fields(stream, time_base)
    duration_ts = _optional_int(stream.get("duration_ts"))
    return VideoStreamInfo(
        stream_index=int(stream["index"]),
        codec_name=_required_text(stream, "codec_name"),
        time_base=time_base,
        source_start_pts=start_pts,
        source_start_time_ms=start_time_ms,
        duration_ts=duration_ts if duration_ts is None or duration_ts >= 0 else None,
        duration_ms=_duration_ms(stream, time_base),
        width=int(stream["width"]),
        height=int(stream["height"]),
        average_frame_rate=average,
        real_frame_rate=real,
        pixel_format=cast(str | None, stream.get("pix_fmt")),
        rotation_degrees=_rotation(stream),
        frame_count=_optional_int(stream.get("nb_frames")),
        variable_frame_rate=variable,
        vfr_indicators=indicators,
    )


def _parse_audio_stream(stream: dict[str, Any]) -> AudioStreamInfo:
    time_base = parse_rational(_required_text(stream, "time_base"))
    if time_base is None:
        raise MediaProbeError("audio time_base must be a positive rational")
    start_pts, start_time_ms = _start_fields(stream, time_base)
    duration_ts = _optional_int(stream.get("duration_ts"))
    return AudioStreamInfo(
        stream_index=int(stream["index"]),
        codec_name=_required_text(stream, "codec_name"),
        time_base=time_base,
        source_start_pts=start_pts,
        source_start_time_ms=start_time_ms,
        duration_ts=duration_ts if duration_ts is None or duration_ts >= 0 else None,
        duration_ms=_duration_ms(stream, time_base),
        sample_rate_hz=int(_required_text(stream, "sample_rate")),
        channels=int(stream["channels"]),
    )


class FFprobeAdapter:
    """Execute ffprobe and parse only its JSON output."""

    def __init__(self, executable: str = "ffprobe", *, timeout_seconds: int = 120) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.version = self._detect_version()

    def _detect_version(self) -> str:
        try:
            completed = subprocess.run(
                [self.executable, "-version"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise MediaProbeError(f"cannot execute ffprobe: {error}") from error
        if completed.returncode != 0 or not completed.stdout.strip():
            raise MediaProbeError(f"ffprobe version probe failed: {completed.stderr.strip()}")
        return completed.stdout.splitlines()[0].strip()

    def run_json(
        self,
        source_path: Path,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("ffprobe timeout_seconds must be positive")
        command = [
            self.executable,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(source_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds or self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise MediaProbeTimeoutError("ffprobe execution timed out") from error
        except OSError as error:
            raise MediaProbeError(f"ffprobe execution failed: {error}") from error
        if completed.returncode != 0:
            raise MediaProbeError(
                f"ffprobe failed with exit code {completed.returncode}: {completed.stderr.strip()}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise MediaProbeError("ffprobe did not return valid JSON") from error
        if not isinstance(payload, dict):
            raise MediaProbeError("ffprobe root JSON value must be an object")
        return cast(dict[str, Any], payload)

    def save_raw_json(self, payload: dict[str, Any], output_path: Path) -> ArtifactRef:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            if output_path.read_bytes() != serialized:
                raise CacheError("immutable raw ffprobe artifact has conflicting content")
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.", dir=output_path.parent
            )
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary_path.replace(output_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        return ArtifactRef(
            artifact_id=f"ffprobe-{hashlib.sha256(serialized).hexdigest()[:20]}",
            uri=output_path.resolve().as_uri(),
            sha256=hashlib.sha256(serialized).hexdigest(),
            media_type="application/json",
            size_bytes=len(serialized),
        )

    @staticmethod
    def parse(
        payload: dict[str, Any],
        *,
        source: ArtifactRef,
        raw_ffprobe: ArtifactRef,
        video_id: str | None = None,
    ) -> VideoAsset:
        raw_streams = payload.get("streams")
        raw_format = payload.get("format")
        if not isinstance(raw_streams, list) or not isinstance(raw_format, dict):
            raise MediaProbeError("ffprobe JSON requires streams and format")
        streams = [cast(dict[str, Any], value) for value in raw_streams if isinstance(value, dict)]
        video_candidates = [stream for stream in streams if stream.get("codec_type") == "video"]
        if not video_candidates:
            raise MediaProbeError("media contains no video stream")
        video_stream = _parse_video_stream(video_candidates[0])
        audio_streams = tuple(
            _parse_audio_stream(stream) for stream in streams if stream.get("codec_type") == "audio"
        )

        duration_ms = video_stream.duration_ms
        duration_text = raw_format.get("duration")
        if (
            (duration_ms is None or duration_ms <= 0)
            and isinstance(duration_text, str)
            and duration_text not in {"", "N/A"}
        ):
            duration_ms = decimal_seconds_to_ms(duration_text)
        if duration_ms is None or duration_ms <= 0:
            raise MediaProbeError("media duration is missing or non-positive")

        format_name = _required_text(cast(dict[str, Any], raw_format), "format_name")
        container_formats = tuple(
            dict.fromkeys(part.strip() for part in format_name.split(",") if part.strip())
        )
        start_text = raw_format.get("start_time")
        format_start_ms = video_stream.source_start_time_ms
        if isinstance(start_text, str) and start_text not in {"", "N/A"}:
            format_start_ms = decimal_seconds_to_ms(start_text)
        return VideoAsset(
            video_id=video_id or f"video-{source.sha256[:20]}",
            source=source,
            raw_ffprobe=raw_ffprobe,
            container_formats=container_formats,
            duration_ms=duration_ms,
            source_start_time_ms=format_start_ms,
            video_stream=video_stream,
            audio_streams=audio_streams,
        )

    def probe(
        self,
        source_path: Path,
        *,
        source: ArtifactRef,
        raw_output_path: Path,
        video_id: str | None = None,
    ) -> VideoAsset:
        payload = self.run_json(source_path)
        raw_ref = self.save_raw_json(payload, raw_output_path)
        return self.parse(payload, source=source, raw_ffprobe=raw_ref, video_id=video_id)
