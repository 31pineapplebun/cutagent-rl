"""Deterministic runtime and media helpers for M3A tests."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.tools import ToolExecutionContext
from cutagent.tools.artifacts import ArtifactStore
from cutagent.tools.cache import ToolCache
from cutagent.tools.editing import (
    AddSubtitlesTool,
    ChangeSpeedTool,
    ConcatVideosTool,
    NormalizeAudioTool,
    ReframeVideoTool,
    TrimVideoTool,
)
from cutagent.tools.executor import FFmpegExecutor
from cutagent.tools.readonly import InspectMediaTool, ValidateMediaTool
from cutagent.tools.registry import ToolRegistry
from cutagent.tools.trace import ToolTraceRecorder
from cutagent.tools.validation import MediaValidator
from tests.media_fixtures import require_ffmpeg


@dataclass(frozen=True, slots=True)
class ToolRuntime:
    registry: ToolRegistry
    store: ArtifactStore
    context: ToolExecutionContext
    source: ArtifactRef


def generate_tool_media(
    path: Path,
    *,
    color: str = "red",
    width: int = 160,
    height: int = 90,
    duration_seconds: float = 3.0,
    with_audio: bool = True,
) -> Path:
    ffmpeg, _ = require_ffmpeg()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={width}x{height}:r=10:d={duration_seconds}",
    ]
    if with_audio:
        command.extend(
            (
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=16000:duration={duration_seconds}",
            )
        )
    command.extend(("-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"))
    if with_audio:
        command.extend(("-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"))
    command.extend(("-movflags", "+faststart", "-y", str(path)))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"tool media fixture failed: {completed.stderr}")
    return path


def build_tool_runtime(
    root: Path,
    source_path: Path,
    *,
    timeout_ms: int = 120_000,
    maximum_output_bytes: int = 100_000_000,
) -> ToolRuntime:
    ffmpeg, ffprobe = require_ffmpeg()
    store = ArtifactStore(root / "store")
    source = store.import_file(source_path, media_type="video/mp4")
    executor = FFmpegExecutor(ffmpeg)
    validator = MediaValidator(
        artifact_store=store,
        executor=executor,
        ffprobe_executable=ffprobe,
    )
    cache = ToolCache(root / "cache", store)
    registry = ToolRegistry(
        artifact_store=store,
        trace_recorder=ToolTraceRecorder(root / "traces"),
    )
    tools = (
        InspectMediaTool(artifact_store=store, validator=validator),
        TrimVideoTool(artifact_store=store, executor=executor, validator=validator, cache=cache),
        ConcatVideosTool(artifact_store=store, executor=executor, validator=validator, cache=cache),
        ChangeSpeedTool(artifact_store=store, executor=executor, validator=validator, cache=cache),
        AddSubtitlesTool(artifact_store=store, executor=executor, validator=validator, cache=cache),
        ReframeVideoTool(artifact_store=store, executor=executor, validator=validator, cache=cache),
        NormalizeAudioTool(
            artifact_store=store, executor=executor, validator=validator, cache=cache
        ),
        ValidateMediaTool(artifact_store=store, validator=validator),
    )
    for tool in tools:
        registry.register(tool)
    context = ToolExecutionContext(
        execution_id="m3a-test-execution",
        allowed_output_root_id="pytest",
        allowed_artifact_ids=(source.artifact_id,),
        allowed_capabilities=(
            "retrieval.read",
            "media.inspect",
            "media.decode",
            "media.write",
            "audio.write",
        ),
        timeout_ms=timeout_ms,
        maximum_output_bytes=maximum_output_bytes,
    )
    return ToolRuntime(registry=registry, store=store, context=context, source=source)
