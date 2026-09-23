"""Explicit construction of the static M3A tool environment."""

from __future__ import annotations

from pathlib import Path

from cutagent.retrieval.store import RetrievalMemory
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
from cutagent.tools.readonly import (
    FrozenAdaptiveRetriever,
    InspectMediaTool,
    SearchVideoTool,
    ValidateMediaTool,
)
from cutagent.tools.registry import ToolRegistry
from cutagent.tools.trace import ToolTraceRecorder
from cutagent.tools.validation import MediaValidator


def create_m3a_registry(
    *,
    root: Path,
    retriever: FrozenAdaptiveRetriever,
    retrieval_memory: RetrievalMemory,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
) -> ToolRegistry:
    """Construct only audited, statically known M3A tools."""

    registry = create_media_tool_registry(
        root=root,
        ffmpeg_executable=ffmpeg_executable,
        ffprobe_executable=ffprobe_executable,
    )
    registry.register(SearchVideoTool(retriever=retriever, memory=retrieval_memory))
    return registry


def create_media_tool_registry(
    *,
    root: Path,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
) -> ToolRegistry:
    """Construct the read-only and editing subset without a retrieval backend."""

    artifact_store = ArtifactStore(root / "artifact_store")
    executor = FFmpegExecutor(ffmpeg_executable)
    validator = MediaValidator(
        artifact_store=artifact_store,
        executor=executor,
        ffprobe_executable=ffprobe_executable,
    )
    cache = ToolCache(root / "cache", artifact_store)
    registry = ToolRegistry(
        artifact_store=artifact_store,
        trace_recorder=ToolTraceRecorder(root / "traces"),
    )
    tools = (
        InspectMediaTool(artifact_store=artifact_store, validator=validator),
        TrimVideoTool(
            artifact_store=artifact_store,
            executor=executor,
            validator=validator,
            cache=cache,
        ),
        ConcatVideosTool(
            artifact_store=artifact_store,
            executor=executor,
            validator=validator,
            cache=cache,
        ),
        ChangeSpeedTool(
            artifact_store=artifact_store,
            executor=executor,
            validator=validator,
            cache=cache,
        ),
        AddSubtitlesTool(
            artifact_store=artifact_store,
            executor=executor,
            validator=validator,
            cache=cache,
        ),
        ReframeVideoTool(
            artifact_store=artifact_store,
            executor=executor,
            validator=validator,
            cache=cache,
        ),
        NormalizeAudioTool(
            artifact_store=artifact_store,
            executor=executor,
            validator=validator,
            cache=cache,
        ),
        ValidateMediaTool(artifact_store=artifact_store, validator=validator),
    )
    for tool in tools:
        registry.register(tool)
    return registry
