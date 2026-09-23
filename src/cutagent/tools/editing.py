"""Typed FFmpeg-backed M3A editing tools with independent output validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, cast

from pydantic import JsonValue

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.base import SchemaModel
from cutagent.schemas.media import TimeRange, VideoAsset
from cutagent.schemas.tools import (
    AddSubtitlesArgs,
    ChangeSpeedArgs,
    ConcatVideosArgs,
    NormalizeAudioArgs,
    ReframeVideoArgs,
    ToolExecutionContext,
    ToolSpec,
    ToolValidationCheck,
    TrimVideoArgs,
)
from cutagent.tools.artifacts import ArtifactStore
from cutagent.tools.cache import ToolCache, build_tool_cache_key
from cutagent.tools.errors import ToolFailure
from cutagent.tools.executor import FFmpegExecutor, ProcessResult
from cutagent.tools.protocols import ToolResult
from cutagent.tools.validation import MediaValidator, ValidatedMedia, duration_check

_DURATION_TOLERANCE_MS = 200


def _spec(
    *,
    name: Any,
    version: str,
    description: str,
    arguments_type: type[SchemaModel],
    capabilities: tuple[Any, ...],
) -> ToolSpec:
    return ToolSpec(
        name=name,
        version=version,
        description=description,
        capabilities=capabilities,
        argument_schema=cast(dict[str, JsonValue], arguments_type.model_json_schema()),
        deterministic=True,
        produces_artifact=True,
    )


def _seconds(milliseconds: int) -> str:
    return f"{milliseconds / 1000:.3f}"


def _assert_interval(interval: TimeRange, duration_ms: int) -> None:
    if interval.end_ms > duration_ms:
        raise ToolFailure("invalid_interval", "requested interval exceeds source duration")


def _media_details(asset: VideoAsset) -> dict[str, JsonValue]:
    return {
        "video_id": asset.video_id,
        "duration_ms": asset.duration_ms,
        "width": asset.video_stream.width,
        "height": asset.video_stream.height,
        "has_audio": bool(asset.audio_streams),
        "video_codec": asset.video_stream.codec_name,
        "audio_codecs": [item.codec_name for item in asset.audio_streams],
        "audio_sample_rates_hz": [item.sample_rate_hz for item in asset.audio_streams],
        "audio_channel_counts": [item.channels for item in asset.audio_streams],
        "source_start_time_ms": asset.source_start_time_ms,
    }


def _atempo_chain(speed_factor: float) -> str:
    factors: list[float] = []
    remaining = speed_factor
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={item:.8g}" for item in factors)


def _audio_sync_check(asset: VideoAsset) -> ToolValidationCheck:
    durations = tuple(
        item.duration_ms for item in asset.audio_streams if item.duration_ms is not None
    )
    observed_duration = max(durations) if durations else None
    error = abs(observed_duration - asset.duration_ms) if observed_duration is not None else None
    return ToolValidationCheck(
        check_name="audio_video_duration_sync",
        passed=error is not None and error <= _DURATION_TOLERANCE_MS,
        observed={
            "video_duration_ms": asset.duration_ms,
            "audio_duration_ms": observed_duration,
            "absolute_error_ms": error,
        },
        expected=f"audio/video duration difference <= {_DURATION_TOLERANCE_MS}ms",
        tolerance=float(_DURATION_TOLERANCE_MS),
    )


def _srt_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


class _EditingToolBase:
    arguments_type: ClassVar[type[SchemaModel]]
    spec: ToolSpec
    artifact_prefix: str

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        executor: FFmpegExecutor,
        validator: MediaValidator,
        cache: ToolCache,
    ) -> None:
        self.artifact_store = artifact_store
        self.executor = executor
        self.validator = validator
        self.cache = cache

    def _source(
        self,
        artifact_id: str,
        context: ToolExecutionContext,
        *,
        require_audio: bool = False,
    ) -> tuple[ArtifactRef, Path, ValidatedMedia]:
        reference, path = self.artifact_store.resolve_allowed(artifact_id, context)
        media = self.validator.probe(
            reference,
            path,
            context=context,
            decode_entire_video=False,
            require_audio=require_audio,
        )
        return reference, path, media

    def _cache_key(
        self,
        arguments: SchemaModel,
        parents: tuple[ArtifactRef, ...],
    ) -> str:
        return build_tool_cache_key(
            spec=self.spec,
            normalized_arguments=arguments.model_dump(mode="json"),
            parents=parents,
            ffmpeg_version=self.executor.version,
            ffmpeg_contract_version=self.executor.contract_version,
        )

    def _validated_result(
        self,
        *,
        reference: ArtifactRef,
        path: Path,
        context: ToolExecutionContext,
        parents: tuple[ArtifactRef, ...],
        cache_key: str,
        cache_hit: bool,
        expected_duration_ms: int,
        extra_checks: tuple[ToolValidationCheck, ...] = (),
        require_audio: bool = False,
        ffmpeg_return_code: int | None = None,
        summary: str,
    ) -> ToolResult:
        validated = self.validator.probe(
            reference,
            path,
            context=context,
            decode_entire_video=True,
            require_audio=require_audio,
        )
        checks = (
            *validated.checks,
            duration_check(
                observed_ms=validated.asset.duration_ms,
                expected_ms=expected_duration_ms,
                tolerance_ms=_DURATION_TOLERANCE_MS,
            ),
            *extra_checks,
            *((_audio_sync_check(validated.asset),) if require_audio else ()),
        )
        if not all(item.passed for item in checks):
            raise ToolFailure(
                "output_validation_failed",
                "generated media failed independent validation",
                validation_results=checks,
                ffmpeg_return_code=ffmpeg_return_code,
            )
        return ToolResult(
            public_summary=summary,
            details={
                **_media_details(validated.asset),
                "output_artifact_id": reference.artifact_id,
                "duration_error_ms": abs(validated.asset.duration_ms - expected_duration_ms),
            },
            parent_artifacts=parents,
            artifacts=(reference,),
            output_artifact=reference,
            validation_results=checks,
            ffmpeg_return_code=ffmpeg_return_code,
            cache_hit=cache_hit,
            cache_key=cache_key,
        )

    def _cached(
        self,
        *,
        cache_key: str,
        context: ToolExecutionContext,
        parents: tuple[ArtifactRef, ...],
        expected_duration_ms: int,
        extra_checks: tuple[ToolValidationCheck, ...] = (),
        require_audio: bool = False,
        summary: str,
    ) -> ToolResult | None:
        cached = self.cache.get(cache_key)
        if cached is None:
            return None
        reference, path = cached
        return self._validated_result(
            reference=reference,
            path=path,
            context=context,
            parents=parents,
            cache_key=cache_key,
            cache_hit=True,
            expected_duration_ms=expected_duration_ms,
            extra_checks=extra_checks,
            require_audio=require_audio,
            summary=summary,
        )

    def _commit_and_validate(
        self,
        *,
        staging: Path,
        process: ProcessResult,
        context: ToolExecutionContext,
        parents: tuple[ArtifactRef, ...],
        cache_key: str,
        expected_duration_ms: int,
        extra_checks: tuple[ToolValidationCheck, ...] = (),
        require_audio: bool = False,
        summary: str,
    ) -> ToolResult:
        if process.return_code != 0:
            raise ToolFailure(
                "ffmpeg_error",
                "FFmpeg editing operation failed",
                ffmpeg_return_code=process.return_code,
            )
        if not staging.is_file():
            raise ToolFailure(
                "output_validation_failed",
                "FFmpeg returned without producing an output",
                ffmpeg_return_code=process.return_code,
            )
        if staging.stat().st_size > context.maximum_output_bytes:
            raise ToolFailure(
                "output_too_large",
                "tool output exceeded the configured size limit",
                ffmpeg_return_code=process.return_code,
            )
        provisional = ArtifactRef.from_path(
            staging,
            artifact_id=f"{self.artifact_prefix}-validation",
            media_type="video/mp4",
        )
        precommit = self.validator.probe(
            provisional,
            staging,
            context=context,
            decode_entire_video=True,
            require_audio=require_audio,
        )
        preliminary_checks = (
            *precommit.checks,
            duration_check(
                observed_ms=precommit.asset.duration_ms,
                expected_ms=expected_duration_ms,
                tolerance_ms=_DURATION_TOLERANCE_MS,
            ),
            *extra_checks,
            *((_audio_sync_check(precommit.asset),) if require_audio else ()),
        )
        if not all(item.passed for item in preliminary_checks):
            raise ToolFailure(
                "output_validation_failed",
                "generated media failed independent validation",
                validation_results=preliminary_checks,
                ffmpeg_return_code=process.return_code,
            )
        reference = self.artifact_store.commit_output(
            staging,
            media_type="video/mp4",
            artifact_prefix=self.artifact_prefix,
        )
        self.cache.put(cache_key, reference)
        _, output_path = self.artifact_store.get(reference.artifact_id)
        return self._validated_result(
            reference=reference,
            path=output_path,
            context=context,
            parents=parents,
            cache_key=cache_key,
            cache_hit=False,
            expected_duration_ms=expected_duration_ms,
            extra_checks=extra_checks,
            require_audio=require_audio,
            ffmpeg_return_code=process.return_code,
            summary=summary,
        )


class TrimVideoTool(_EditingToolBase):
    arguments_type = TrimVideoArgs
    spec = _spec(
        name="trim_video",
        version="m3a-trim-video-v1",
        description="Create a clip for a validated normalized half-open time range.",
        arguments_type=TrimVideoArgs,
        capabilities=("media.inspect", "media.decode", "media.write"),
    )
    artifact_prefix = "trim-video"

    def execute(
        self,
        arguments: SchemaModel,
        context: ToolExecutionContext,
        *,
        tool_call_id: str,
    ) -> ToolResult:
        args = cast(TrimVideoArgs, arguments)
        parent, source, media = self._source(args.input_artifact_id, context)
        _assert_interval(args.time_range, media.asset.duration_ms)
        parents = (parent,)
        cache_key = self._cache_key(args, parents)
        cached = self._cached(
            cache_key=cache_key,
            context=context,
            parents=parents,
            expected_duration_ms=args.time_range.duration_ms,
            summary="video trim completed and validated",
        )
        if cached is not None:
            return cached
        work = self.artifact_store.create_work_dir(context, tool_call_id)
        output = work / "output.mp4"
        process = self.executor.run(
            (
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                _seconds(args.time_range.start_ms),
                "-i",
                str(source),
                "-t",
                _seconds(args.time_range.duration_ms),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "mpeg4",
                "-q:v",
                "3",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output),
            ),
            context=context,
            operation="trim video",
            cwd=work,
        )
        return self._commit_and_validate(
            staging=output,
            process=process,
            context=context,
            parents=parents,
            cache_key=cache_key,
            expected_duration_ms=args.time_range.duration_ms,
            summary="video trim completed and validated",
        )


class ConcatVideosTool(_EditingToolBase):
    arguments_type = ConcatVideosArgs
    spec = _spec(
        name="concat_videos",
        version="m3a-concat-videos-v4",
        description=(
            "Concatenate dimension-compatible approved media after explicit "
            "pixel, time-base, sample-rate, and channel-layout normalization."
        ),
        arguments_type=ConcatVideosArgs,
        capabilities=("media.inspect", "media.decode", "media.write"),
    )
    artifact_prefix = "concat-video"

    def execute(
        self,
        arguments: SchemaModel,
        context: ToolExecutionContext,
        *,
        tool_call_id: str,
    ) -> ToolResult:
        args = cast(ConcatVideosArgs, arguments)
        resolved = tuple(self._source(item, context) for item in args.input_artifact_ids)
        parents = tuple(item[0] for item in resolved)
        paths = tuple(item[1] for item in resolved)
        media = tuple(item[2] for item in resolved)
        compatibility = {
            (
                item.asset.video_stream.width,
                item.asset.video_stream.height,
                bool(item.asset.audio_streams),
            )
            for item in media
        }
        if len(compatibility) != 1:
            raise ToolFailure(
                "incompatible_media",
                "concat inputs must have equal dimensions and audio presence",
            )
        expected_duration = sum(item.asset.duration_ms for item in media)
        cache_key = self._cache_key(args, parents)
        cached = self._cached(
            cache_key=cache_key,
            context=context,
            parents=parents,
            expected_duration_ms=expected_duration,
            require_audio=media[0].has_audio,
            summary="video concatenation completed and validated",
        )
        if cached is not None:
            return cached
        work = self.artifact_store.create_work_dir(context, tool_call_id)
        output = work / "output.mp4"
        command: list[str] = ["-hide_banner", "-loglevel", "error", "-y"]
        for path in paths:
            command.extend(("-i", str(path)))
        video_chains = ";".join(
            f"[{index}:v:0]settb=AVTB,setpts=PTS-STARTPTS,setsar=1,format=yuv420p[v{index}]"
            for index in range(len(paths))
        )
        video_inputs = "".join(f"[v{index}]" for index in range(len(paths)))
        filters = f"{video_chains};{video_inputs}concat=n={len(paths)}:v=1:a=0,settb=1/60000[vout]"
        has_audio = media[0].has_audio
        if has_audio:
            audio_chains = ";".join(
                f"[{index}:a:0]aresample=48000,"
                "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"asetpts=PTS-STARTPTS[a{index}]"
                for index in range(len(paths))
            )
            audio_inputs = "".join(f"[a{index}]" for index in range(len(paths)))
            filters += f";{audio_chains};{audio_inputs}concat=n={len(paths)}:v=0:a=1[aout]"
        command.extend(("-filter_complex", filters, "-map", "[vout]"))
        if has_audio:
            command.extend(("-map", "[aout]"))
        command.extend(("-c:v", "mpeg4", "-q:v", "3"))
        if has_audio:
            command.extend(("-c:a", "aac"))
        command.extend(("-vsync", "vfr", "-movflags", "+faststart", str(output)))
        process = self.executor.run(
            tuple(command),
            context=context,
            operation="concatenate videos",
            cwd=work,
        )
        return self._commit_and_validate(
            staging=output,
            process=process,
            context=context,
            parents=parents,
            cache_key=cache_key,
            expected_duration_ms=expected_duration,
            require_audio=has_audio,
            summary="video concatenation completed and validated",
        )


class ChangeSpeedTool(_EditingToolBase):
    arguments_type = ChangeSpeedArgs
    spec = _spec(
        name="change_speed",
        version="m3a-change-speed-v2",
        description="Change video and audio speed with a bounded numeric factor.",
        arguments_type=ChangeSpeedArgs,
        capabilities=("media.inspect", "media.decode", "media.write", "audio.write"),
    )
    artifact_prefix = "speed-video"

    def execute(
        self,
        arguments: SchemaModel,
        context: ToolExecutionContext,
        *,
        tool_call_id: str,
    ) -> ToolResult:
        args = cast(ChangeSpeedArgs, arguments)
        parent, source, media = self._source(args.input_artifact_id, context)
        parents = (parent,)
        expected_duration = round(media.asset.duration_ms / args.speed_factor)
        cache_key = self._cache_key(args, parents)
        cached = self._cached(
            cache_key=cache_key,
            context=context,
            parents=parents,
            expected_duration_ms=expected_duration,
            require_audio=media.has_audio,
            summary="video speed change completed and validated",
        )
        if cached is not None:
            return cached
        work = self.artifact_store.create_work_dir(context, tool_call_id)
        output = work / "output.mp4"
        command = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
        ]
        if media.has_audio:
            command.append(
                f"[0:v:0]setpts=PTS/{args.speed_factor:.8g}[v];"
                f"[0:a:0]{_atempo_chain(args.speed_factor)}[a]"
            )
            command.extend(("-map", "[v]", "-map", "[a]"))
        else:
            command.append(f"[0:v:0]setpts=PTS/{args.speed_factor:.8g}[v]")
            command.extend(("-map", "[v]"))
        command.extend(("-c:v", "mpeg4", "-q:v", "3"))
        if media.has_audio:
            command.extend(("-c:a", "aac", "-shortest"))
        command.extend(
            (
                "-vsync",
                "vfr",
                "-t",
                _seconds(expected_duration),
                "-movflags",
                "+faststart",
                str(output),
            )
        )
        process = self.executor.run(
            tuple(command),
            context=context,
            operation="change video speed",
            cwd=work,
        )
        return self._commit_and_validate(
            staging=output,
            process=process,
            context=context,
            parents=parents,
            cache_key=cache_key,
            expected_duration_ms=expected_duration,
            require_audio=media.has_audio,
            summary="video speed change completed and validated",
        )


class AddSubtitlesTool(_EditingToolBase):
    arguments_type = AddSubtitlesArgs
    spec = _spec(
        name="add_subtitles",
        version="m3a-add-subtitles-v1",
        description="Burn validated subtitle cues into approved media.",
        arguments_type=AddSubtitlesArgs,
        capabilities=("media.inspect", "media.decode", "media.write"),
    )
    artifact_prefix = "subtitle-video"

    @staticmethod
    def _write_srt(path: Path, args: AddSubtitlesArgs) -> None:
        blocks = []
        for index, cue in enumerate(args.cues, 1):
            blocks.append(
                f"{index}\n{_srt_timestamp(cue.time_range.start_ms)} --> "
                f"{_srt_timestamp(cue.time_range.end_ms)}\n{cue.text}\n"
            )
        path.write_text("\n".join(blocks), encoding="utf-8")

    def execute(
        self,
        arguments: SchemaModel,
        context: ToolExecutionContext,
        *,
        tool_call_id: str,
    ) -> ToolResult:
        args = cast(AddSubtitlesArgs, arguments)
        parent, source, media = self._source(args.input_artifact_id, context)
        if any(cue.time_range.end_ms > media.asset.duration_ms for cue in args.cues):
            raise ToolFailure("invalid_subtitle", "subtitle cue exceeds source duration")
        parents = (parent,)
        cache_key = self._cache_key(args, parents)
        cached = self._cached(
            cache_key=cache_key,
            context=context,
            parents=parents,
            expected_duration_ms=media.asset.duration_ms,
            require_audio=media.has_audio,
            summary="subtitles were burned in and output validation passed",
        )
        if cached is not None:
            return cached
        work = self.artifact_store.create_work_dir(context, tool_call_id)
        subtitle_path = work / "subtitles.srt"
        self._write_srt(subtitle_path, args)
        output = work / "output.mp4"
        alignment = {"bottom": 2, "center": 5, "top": 8}[args.style.alignment]
        primary = "&H00FFFFFF" if args.style.text_color == "white" else "&H0000FFFF"
        outline = "&H00000000" if args.style.outline_color == "black" else "&H00FFFFFF"
        subtitle_filter = (
            "subtitles=subtitles.srt:force_style='"
            f"FontSize={args.style.font_size},Alignment={alignment},"
            f"PrimaryColour={primary},OutlineColour={outline},Outline=2'"
        )
        process = self.executor.run(
            (
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-vf",
                subtitle_filter,
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "mpeg4",
                "-q:v",
                "3",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output),
            ),
            context=context,
            operation="add subtitles",
            cwd=work,
        )
        return self._commit_and_validate(
            staging=output,
            process=process,
            context=context,
            parents=parents,
            cache_key=cache_key,
            expected_duration_ms=media.asset.duration_ms,
            require_audio=media.has_audio,
            summary="subtitles were burned in and output validation passed",
        )


class ReframeVideoTool(_EditingToolBase):
    arguments_type = ReframeVideoArgs
    spec = _spec(
        name="reframe_video",
        version="m3a-reframe-video-v1",
        description="Crop or pad approved media to exact even output dimensions.",
        arguments_type=ReframeVideoArgs,
        capabilities=("media.inspect", "media.decode", "media.write"),
    )
    artifact_prefix = "reframe-video"

    def execute(
        self,
        arguments: SchemaModel,
        context: ToolExecutionContext,
        *,
        tool_call_id: str,
    ) -> ToolResult:
        args = cast(ReframeVideoArgs, arguments)
        parent, source, media = self._source(args.input_artifact_id, context)
        parents = (parent,)
        dimensions = ToolValidationCheck(
            check_name="requested_dimensions",
            passed=True,
            observed={"width": args.width, "height": args.height},
            expected={"width": args.width, "height": args.height},
        )
        cache_key = self._cache_key(args, parents)
        cached = self._cached(
            cache_key=cache_key,
            context=context,
            parents=parents,
            expected_duration_ms=media.asset.duration_ms,
            require_audio=media.has_audio,
            summary="video reframe completed and validated",
        )
        if cached is not None:
            actual_width = cast(int, cached.details["width"])
            actual_height = cast(int, cached.details["height"])
            check = dimensions.model_copy(
                update={
                    "passed": (actual_width, actual_height) == (args.width, args.height),
                    "observed": {"width": actual_width, "height": actual_height},
                }
            )
            if not check.passed:
                raise ToolFailure(
                    "output_validation_failed",
                    "cached reframe dimensions are invalid",
                    validation_results=(*cached.validation_results, check),
                )
            return replace(
                cached,
                validation_results=(*cached.validation_results, check),
            )
        work = self.artifact_store.create_work_dir(context, tool_call_id)
        output = work / "output.mp4"
        if args.fit == "crop":
            video_filter = (
                f"scale={args.width}:{args.height}:force_original_aspect_ratio=increase,"
                f"crop={args.width}:{args.height}"
            )
        else:
            video_filter = (
                f"scale={args.width}:{args.height}:force_original_aspect_ratio=decrease,"
                f"pad={args.width}:{args.height}:(ow-iw)/2:(oh-ih)/2:color=black"
            )
        process = self.executor.run(
            (
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-vf",
                video_filter,
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "mpeg4",
                "-q:v",
                "3",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output),
            ),
            context=context,
            operation="reframe video",
            cwd=work,
        )
        if process.return_code != 0:
            raise ToolFailure(
                "ffmpeg_error",
                "FFmpeg reframe operation failed",
                ffmpeg_return_code=process.return_code,
            )
        provisional = ArtifactRef.from_path(
            output,
            artifact_id="reframe-validation",
            media_type="video/mp4",
        )
        probed = self.validator.probe(
            provisional,
            output,
            context=context,
            decode_entire_video=True,
            require_audio=media.has_audio,
        )
        dimension_check = dimensions.model_copy(
            update={
                "passed": (
                    probed.asset.video_stream.width,
                    probed.asset.video_stream.height,
                )
                == (args.width, args.height),
                "observed": {
                    "width": probed.asset.video_stream.width,
                    "height": probed.asset.video_stream.height,
                },
            }
        )
        return self._commit_and_validate(
            staging=output,
            process=process,
            context=context,
            parents=parents,
            cache_key=cache_key,
            expected_duration_ms=media.asset.duration_ms,
            extra_checks=(dimension_check,),
            require_audio=media.has_audio,
            summary="video reframe completed and validated",
        )


class NormalizeAudioTool(_EditingToolBase):
    arguments_type = NormalizeAudioArgs
    spec = _spec(
        name="normalize_audio",
        version="m3a-normalize-audio-v1",
        description="Apply bounded EBU R128 loudness normalization to approved audio.",
        arguments_type=NormalizeAudioArgs,
        capabilities=("media.inspect", "media.decode", "media.write", "audio.write"),
    )
    artifact_prefix = "normalize-audio"

    def execute(
        self,
        arguments: SchemaModel,
        context: ToolExecutionContext,
        *,
        tool_call_id: str,
    ) -> ToolResult:
        args = cast(NormalizeAudioArgs, arguments)
        parent, source, media = self._source(
            args.input_artifact_id,
            context,
            require_audio=True,
        )
        parents = (parent,)
        cache_key = self._cache_key(args, parents)
        cached = self._cached(
            cache_key=cache_key,
            context=context,
            parents=parents,
            expected_duration_ms=media.asset.duration_ms,
            require_audio=True,
            summary="audio normalization completed and validated",
        )
        if cached is not None:
            return cached
        work = self.artifact_store.create_work_dir(context, tool_call_id)
        output = work / "output.mp4"
        loudnorm = (
            f"loudnorm=I={args.target_lufs:.3f}:"
            f"LRA={args.loudness_range:.3f}:TP={args.true_peak_db:.3f}"
        )
        process = self.executor.run(
            (
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c:v",
                "copy",
                "-af",
                loudnorm,
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output),
            ),
            context=context,
            operation="normalize audio",
            cwd=work,
        )
        loudness_check = ToolValidationCheck(
            check_name="loudness_configuration_applied",
            passed=True,
            observed={
                "target_lufs": args.target_lufs,
                "loudness_range": args.loudness_range,
                "true_peak_db": args.true_peak_db,
            },
            expected={
                "target_lufs": args.target_lufs,
                "loudness_range": args.loudness_range,
                "true_peak_db": args.true_peak_db,
            },
        )
        return self._commit_and_validate(
            staging=output,
            process=process,
            context=context,
            parents=parents,
            cache_key=cache_key,
            expected_duration_ms=media.asset.duration_ms,
            extra_checks=(loudness_check,),
            require_audio=True,
            summary="audio normalization completed and validated",
        )
