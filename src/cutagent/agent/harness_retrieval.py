"""Reuse one frozen retrieval chain over public ingestion/perception artifacts."""

from pathlib import Path
from urllib.parse import unquote, urlparse

from cutagent.retrieval.adaptive import AdaptiveHybridRetriever
from cutagent.retrieval.encoders import BGETextEncoder, Siglip2VisualEncoder
from cutagent.retrieval.fusion import WeightedRankFusion
from cutagent.retrieval.index import KeyframeAsset, RetrievalIndexBuilder
from cutagent.retrieval.protocols import MultimodalRetriever
from cutagent.retrieval.retrievers import DenseTextRetriever, SparseRetriever, VisualRetriever
from cutagent.schemas.media import IngestionResult
from cutagent.schemas.perception import PerceptionResult
from cutagent.schemas.retrieval import AdaptiveRetrievalConfig, RetrievalChannel, RetrievalConfig
from cutagent.tools.readonly import SearchVideoTool


def local_artifact_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError("prepared artifacts must be local files")
    value = unquote(parsed.path)
    if len(value) >= 3 and value[0] == "/" and value[2] == ":":
        value = value[1:]
    return Path(value).resolve(strict=True)


def prepared_search_tool(
    *, ingestion: IngestionResult, perception: PerceptionResult, index_root: Path, model_cache: Path
) -> SearchVideoTool:
    """No benchmark annotations are read; models must already exist in the local cache."""
    if perception.source_video_id != ingestion.video.video_id:
        raise ValueError("ingestion and perception refer to different videos")
    if perception.world_state.duration_ms != ingestion.video.duration_ms:
        raise ValueError("ingestion and perception durations differ")
    config = RetrievalConfig()
    text_encoder = BGETextEncoder(
        model_cache=model_cache,
        batch_size=config.text_batch_size,
        maximum_tokens=config.text_encoder_max_tokens,
    )
    visual_encoder = Siglip2VisualEncoder(
        model_cache=model_cache, batch_size=config.visual_batch_size
    )
    memory = RetrievalIndexBuilder(index_root).build(
        world_states=(perception.world_state,),
        keyframe_assets=tuple(
            KeyframeAsset(artifact=item.artifact, path=local_artifact_path(item.artifact.uri))
            for item in ingestion.keyframes
        ),
        config=config,
        text_encoder=text_encoder,
        visual_encoder=visual_encoder,
    )
    components: dict[RetrievalChannel, MultimodalRetriever] = {
        "bm25_transcript": SparseRetriever("transcript"),
        "bm25_ocr": SparseRetriever("ocr"),
        "bm25_structured": SparseRetriever("structured_semantic"),
        "bm25_combined": SparseRetriever("combined"),
        "dense_text": DenseTextRetriever(text_encoder),
        "visual": VisualRetriever(visual_encoder),
    }
    return SearchVideoTool(
        retriever=AdaptiveHybridRetriever(
            config=AdaptiveRetrievalConfig(native_video_enabled=False),
            fusion=WeightedRankFusion(components),
            strategy="query_aware",
            evidence_reranking=True,
            native_video=False,
        ),
        memory=memory,
    )
