"""Read-only M3A tools over frozen retrieval and media-probe foundations."""

from __future__ import annotations

from typing import Protocol, cast

from pydantic import JsonValue

from cutagent.retrieval.store import RetrievalMemory
from cutagent.schemas.base import SchemaModel
from cutagent.schemas.retrieval import AdaptiveRetrievalResult, RetrievalQuery
from cutagent.schemas.tools import (
    InspectMediaArgs,
    SearchVideoArgs,
    ToolExecutionContext,
    ToolSpec,
    ValidateMediaArgs,
)
from cutagent.tools.artifacts import ArtifactStore
from cutagent.tools.errors import ToolFailure
from cutagent.tools.protocols import ToolResult
from cutagent.tools.validation import MediaValidator


class FrozenAdaptiveRetriever(Protocol):
    """The exact M2B trace-bearing interface wrapped by ``search_video``."""

    def search_with_trace(
        self,
        query: RetrievalQuery,
        memory_ref: RetrievalMemory,
    ) -> AdaptiveRetrievalResult: ...


class SearchVideoTool:
    arguments_type = SearchVideoArgs
    spec = ToolSpec(
        name="search_video",
        version="m3a-search-video-v1",
        description=(
            "Search the frozen M2B query-aware weighted, evidence-reranked "
            "BM25, DenseText, and Visual index. Native-video verification is disabled."
        ),
        capabilities=("retrieval.read",),
        argument_schema=cast(dict[str, JsonValue], SearchVideoArgs.model_json_schema()),
        deterministic=True,
        produces_artifact=False,
    )

    def __init__(
        self,
        *,
        retriever: FrozenAdaptiveRetriever,
        memory: RetrievalMemory,
    ) -> None:
        self.retriever = retriever
        self.memory = memory

    def execute(
        self,
        arguments: SchemaModel,
        context: ToolExecutionContext,
        *,
        tool_call_id: str,
    ) -> ToolResult:
        del context
        args = cast(SearchVideoArgs, arguments)
        query = RetrievalQuery(
            query_id=f"query-{tool_call_id}",
            text=args.query,
            top_k=args.top_k,
            video_id=args.video_id,
            approximate_time_range=args.approximate_time_range,
            required_evidence_types=args.required_evidence_types,
        )
        result = self.retriever.search_with_trace(query, self.memory)
        if result.plan.native_video_policy != "disabled" or result.native_verifications:
            raise ToolFailure(
                "internal_error",
                "M3A search attempted to enable native-video verification",
            )
        return ToolResult(
            public_summary=f"retrieval returned {len(result.response.candidates)} scenes",
            details=cast(
                dict[str, JsonValue],
                {
                    "response": result.response.model_dump(mode="json"),
                    "query_analysis": result.query_analysis.model_dump(mode="json"),
                    "retrieval_plan": result.plan.model_dump(mode="json"),
                    "rerank_trace": [item.model_dump(mode="json") for item in result.rerank_trace],
                },
            ),
            cache_hit=result.response.cache_hit,
        )


def _safe_media_details(media: object) -> dict[str, JsonValue]:
    from cutagent.schemas.media import VideoAsset

    asset = cast(VideoAsset, media)
    return {
        "video_id": asset.video_id,
        "container_formats": list(asset.container_formats),
        "duration_ms": asset.duration_ms,
        "source_start_time_ms": asset.source_start_time_ms,
        "video": {
            "codec": asset.video_stream.codec_name,
            "width": asset.video_stream.width,
            "height": asset.video_stream.height,
            "average_frame_rate": (
                asset.video_stream.average_frame_rate.model_dump(mode="json")
                if asset.video_stream.average_frame_rate is not None
                else None
            ),
            "real_frame_rate": (
                asset.video_stream.real_frame_rate.model_dump(mode="json")
                if asset.video_stream.real_frame_rate is not None
                else None
            ),
            "time_base": asset.video_stream.time_base.model_dump(mode="json"),
            "pixel_format": asset.video_stream.pixel_format,
            "rotation_degrees": asset.video_stream.rotation_degrees,
            "variable_frame_rate": asset.video_stream.variable_frame_rate,
        },
        "audio": [
            {
                "codec": stream.codec_name,
                "sample_rate_hz": stream.sample_rate_hz,
                "channels": stream.channels,
                "source_start_time_ms": stream.source_start_time_ms,
            }
            for stream in asset.audio_streams
        ],
    }


class InspectMediaTool:
    arguments_type = InspectMediaArgs
    spec = ToolSpec(
        name="inspect_media",
        version="m3a-inspect-media-v1",
        description="Return safe observable metadata for an approved media artifact.",
        capabilities=("media.inspect",),
        argument_schema=cast(dict[str, JsonValue], InspectMediaArgs.model_json_schema()),
        deterministic=True,
        produces_artifact=False,
    )

    def __init__(self, *, artifact_store: ArtifactStore, validator: MediaValidator) -> None:
        self.artifact_store = artifact_store
        self.validator = validator

    def execute(
        self,
        arguments: SchemaModel,
        context: ToolExecutionContext,
        *,
        tool_call_id: str,
    ) -> ToolResult:
        del tool_call_id
        args = cast(InspectMediaArgs, arguments)
        parent, path = self.artifact_store.resolve_allowed(args.input_artifact_id, context)
        validated = self.validator.probe(
            parent,
            path,
            context=context,
            decode_entire_video=False,
        )
        return ToolResult(
            public_summary="media metadata inspection completed",
            details=_safe_media_details(validated.asset),
            parent_artifacts=(parent,),
            validation_results=validated.checks,
        )


class ValidateMediaTool:
    arguments_type = ValidateMediaArgs
    spec = ToolSpec(
        name="validate_media",
        version="m3a-validate-media-v1",
        description="Probe and optionally fully decode an approved media artifact.",
        capabilities=("media.inspect", "media.decode"),
        argument_schema=cast(dict[str, JsonValue], ValidateMediaArgs.model_json_schema()),
        deterministic=True,
        produces_artifact=False,
    )

    def __init__(self, *, artifact_store: ArtifactStore, validator: MediaValidator) -> None:
        self.artifact_store = artifact_store
        self.validator = validator

    def execute(
        self,
        arguments: SchemaModel,
        context: ToolExecutionContext,
        *,
        tool_call_id: str,
    ) -> ToolResult:
        del tool_call_id
        args = cast(ValidateMediaArgs, arguments)
        parent, path = self.artifact_store.resolve_allowed(args.input_artifact_id, context)
        validated = self.validator.probe(
            parent,
            path,
            context=context,
            decode_entire_video=args.decode_entire_video,
            require_audio=args.require_audio,
        )
        return ToolResult(
            public_summary="media integrity validation completed",
            details={
                **_safe_media_details(validated.asset),
                "fully_decoded": args.decode_entire_video,
                "audio_required": args.require_audio,
            },
            parent_artifacts=(parent,),
            validation_results=validated.checks,
        )
