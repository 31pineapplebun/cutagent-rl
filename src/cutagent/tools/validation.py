"""Independent ffprobe/decode validation for M3A media tool outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cutagent.core.artifacts import ArtifactRef
from cutagent.core.errors import MediaProbeError, MediaProbeTimeoutError
from cutagent.ingestion.ffprobe import FFprobeAdapter
from cutagent.schemas.media import VideoAsset
from cutagent.schemas.tools import ToolExecutionContext, ToolValidationCheck
from cutagent.tools.artifacts import ArtifactStore
from cutagent.tools.errors import ToolFailure, ToolTimeout
from cutagent.tools.executor import FFmpegExecutor


@dataclass(frozen=True, slots=True)
class ValidatedMedia:
    asset: VideoAsset
    raw_probe: ArtifactRef
    checks: tuple[ToolValidationCheck, ...]

    @property
    def has_audio(self) -> bool:
        return bool(self.asset.audio_streams)


class MediaValidator:
    version = "m3a-media-validator-v1"

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        executor: FFmpegExecutor,
        ffprobe_executable: str = "ffprobe",
    ) -> None:
        self.artifact_store = artifact_store
        self.executor = executor
        self.ffprobe = FFprobeAdapter(ffprobe_executable)

    def probe(
        self,
        reference: ArtifactRef,
        path: Path,
        *,
        context: ToolExecutionContext,
        decode_entire_video: bool,
        require_audio: bool = False,
    ) -> ValidatedMedia:
        try:
            payload = self.ffprobe.run_json(
                path,
                timeout_seconds=context.timeout_ms / 1000,
            )
            raw_probe = self.artifact_store.put_json(
                payload,
                artifact_prefix="tool-ffprobe",
            )
            asset = self.ffprobe.parse(
                payload,
                source=reference,
                raw_ffprobe=raw_probe,
            )
        except MediaProbeTimeoutError as exc:
            raise ToolTimeout("media probe exceeded the configured timeout") from exc
        except (MediaProbeError, OSError, ValueError) as exc:
            raise ToolFailure("corrupt_media", "media probe failed") from exc
        checks: list[ToolValidationCheck] = [
            ToolValidationCheck(
                check_name="ffprobe_integrity",
                passed=True,
                observed=True,
                expected=True,
            ),
            ToolValidationCheck(
                check_name="positive_duration",
                passed=asset.duration_ms > 0,
                observed=asset.duration_ms,
                expected=">0ms",
            ),
            ToolValidationCheck(
                check_name="valid_dimensions",
                passed=asset.video_stream.width > 0 and asset.video_stream.height > 0,
                observed={
                    "width": asset.video_stream.width,
                    "height": asset.video_stream.height,
                },
                expected="positive dimensions",
            ),
        ]
        if require_audio:
            checks.append(
                ToolValidationCheck(
                    check_name="audio_required",
                    passed=bool(asset.audio_streams),
                    observed=bool(asset.audio_streams),
                    expected=True,
                )
            )
        if decode_entire_video:
            decoded = self.executor.run(
                (
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a?",
                    "-f",
                    "null",
                    "-",
                ),
                context=context,
                operation="media decode validation",
            )
            checks.append(
                ToolValidationCheck(
                    check_name="full_decode",
                    passed=decoded.return_code == 0,
                    observed=decoded.return_code,
                    expected=0,
                )
            )
        if not all(item.passed for item in checks):
            raise ToolFailure("output_validation_failed", "media validation checks failed")
        return ValidatedMedia(asset=asset, raw_probe=raw_probe, checks=tuple(checks))


def duration_check(
    *,
    observed_ms: int,
    expected_ms: int,
    tolerance_ms: int,
    check_name: str = "duration_expectation",
) -> ToolValidationCheck:
    error = abs(observed_ms - expected_ms)
    return ToolValidationCheck(
        check_name=check_name,
        passed=error <= tolerance_ms,
        observed={"duration_ms": observed_ms, "absolute_error_ms": error},
        expected=expected_ms,
        tolerance=float(tolerance_ms),
    )


def media_contract_equivalence(
    left: VideoAsset,
    right: VideoAsset,
    *,
    duration_tolerance_ms: int = 200,
) -> ToolValidationCheck:
    """Compare decoded-media contracts without claiming byte-identical encoding."""

    same_dimensions = (
        left.video_stream.width,
        left.video_stream.height,
    ) == (
        right.video_stream.width,
        right.video_stream.height,
    )
    same_stream_presence = bool(left.audio_streams) == bool(right.audio_streams)
    duration_error = abs(left.duration_ms - right.duration_ms)
    return ToolValidationCheck(
        check_name="media_contract_equivalence",
        passed=(
            same_dimensions and same_stream_presence and duration_error <= duration_tolerance_ms
        ),
        observed={
            "left_duration_ms": left.duration_ms,
            "right_duration_ms": right.duration_ms,
            "duration_error_ms": duration_error,
            "left_dimensions": {
                "width": left.video_stream.width,
                "height": left.video_stream.height,
            },
            "right_dimensions": {
                "width": right.video_stream.width,
                "height": right.video_stream.height,
            },
            "left_has_audio": bool(left.audio_streams),
            "right_has_audio": bool(right.audio_streams),
        },
        expected="equal dimensions/stream presence and duration within tolerance",
        tolerance=float(duration_tolerance_ms),
    )
