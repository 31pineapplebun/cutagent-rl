"""Conservative source-versus-analysis-proxy normalization policy."""

from pathlib import Path

from pydantic import JsonValue

from cutagent.core.artifacts import ArtifactRef
from cutagent.core.errors import MediaProcessingError
from cutagent.ingestion.ffmpeg import FFmpegRunner
from cutagent.ingestion.ffprobe import FFprobeAdapter
from cutagent.schemas.media import (
    AnalysisProxy,
    IngestionConfig,
    NormalizationResult,
    TimestampMapping,
    VideoAsset,
)


def normalization_reasons(video: VideoAsset, config: IngestionConfig) -> tuple[str, ...]:
    policy = config.normalization
    if policy.policy == "source_only":
        return ()
    reasons: list[str] = []
    if video.video_stream.width > policy.maximum_analysis_width:
        reasons.append("width_exceeds_analysis_limit")
    if video.video_stream.codec_name not in policy.accepted_video_codecs:
        reasons.append("video_codec_not_accepted_for_analysis")
    if (
        video.video_stream.pixel_format is not None
        and video.video_stream.pixel_format not in policy.accepted_pixel_formats
    ):
        reasons.append("pixel_format_not_accepted_for_analysis")
    return tuple(reasons)


class MediaNormalizer:
    def __init__(self, ffmpeg: FFmpegRunner, ffprobe: FFprobeAdapter) -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    def normalize(
        self,
        source_path: Path,
        *,
        source_video: VideoAsset,
        config: IngestionConfig,
        output_directory: Path,
    ) -> tuple[NormalizationResult, Path, VideoAsset]:
        reasons = normalization_reasons(source_video, config)
        if not reasons:
            return (
                NormalizationResult(proxy_required=False, reasons=("source_media_accepted",)),
                source_path,
                source_video,
            )

        policy = config.normalization
        output_directory.mkdir(parents=True, exist_ok=True)
        proxy_path = output_directory / "analysis-proxy.mkv"
        partial_path = output_directory / ".analysis-proxy.partial.mkv"
        partial_path.unlink(missing_ok=True)
        arguments = [
            "-hide_banner",
            "-nostdin",
            "-copyts",
            "-i",
            str(source_path),
            "-map",
            f"0:{source_video.video_stream.stream_index}",
            "-an",
        ]
        if source_video.video_stream.width > policy.maximum_analysis_width:
            arguments.extend(("-vf", f"scale=w={policy.maximum_analysis_width}:h=-2"))
        arguments.extend(
            (
                "-c:v",
                policy.proxy_video_codec,
                "-pix_fmt",
                policy.proxy_pixel_format,
                "-vsync",
                "0",
                "-y",
                str(partial_path),
            )
        )
        try:
            self.ffmpeg.run(arguments, operation="analysis proxy generation")
            if not partial_path.is_file() or partial_path.stat().st_size == 0:
                raise MediaProcessingError("analysis proxy generation produced no media")
            partial_path.replace(proxy_path)
        finally:
            partial_path.unlink(missing_ok=True)

        provisional_proxy = ArtifactRef.from_path(
            proxy_path,
            artifact_id="proxy-provisional",
            media_type="video/x-matroska",
        )
        proxy_ref = ArtifactRef(
            **{
                **provisional_proxy.model_dump(),
                "artifact_id": f"proxy-{provisional_proxy.sha256[:20]}",
            }
        )
        raw_probe_path = output_directory / "analysis-proxy.ffprobe.json"
        proxy_video = self.ffprobe.probe(
            proxy_path,
            source=proxy_ref,
            raw_output_path=raw_probe_path,
            video_id=source_video.video_id,
        )
        if abs(proxy_video.duration_ms - source_video.duration_ms) > 2:
            raise MediaProcessingError(
                "analysis proxy duration differs from source by more than 2 ms"
            )

        transformation: dict[str, JsonValue] = {
            "video_codec": policy.proxy_video_codec,
            "pixel_format": policy.proxy_pixel_format,
            "maximum_width": policy.maximum_analysis_width,
            "audio": "omitted_from_visual_analysis_proxy",
            "vsync": "passthrough",
            "copy_timestamps": True,
        }
        proxy = AnalysisProxy(
            source=source_video.source,
            proxy=proxy_ref,
            raw_ffprobe=proxy_video.raw_ffprobe,
            transformation_config=transformation,
            ffmpeg_version=self.ffmpeg.version,
            timestamp_mapping=TimestampMapping(
                source_time_base=source_video.video_stream.time_base,
                source_origin_pts=source_video.video_stream.source_start_pts,
                analysis_time_base=proxy_video.video_stream.time_base,
                analysis_origin_pts=proxy_video.video_stream.source_start_pts,
                preserves_frame_timestamps=True,
            ),
        )
        return (
            NormalizationResult(proxy_required=True, reasons=reasons, analysis_proxy=proxy),
            proxy_path,
            proxy_video,
        )
