"""M2 scene retrieval backends and deterministic local indices."""

from cutagent.retrieval.adaptive import AdaptiveHybridRetriever
from cutagent.retrieval.documents import SceneDocument, SceneDocumentBuilder
from cutagent.retrieval.fusion import WeightedRankFusion
from cutagent.retrieval.protocols import MultimodalRetriever, TextEncoder, VisualEncoder
from cutagent.retrieval.query import QueryAnalyzer, RetrievalRouter
from cutagent.retrieval.reranking import EvidenceAwareReranker
from cutagent.retrieval.retrievers import (
    DenseTextRetriever,
    FusionRetriever,
    SparseRetriever,
    VisualRetriever,
)

__all__ = [
    "AdaptiveHybridRetriever",
    "DenseTextRetriever",
    "EvidenceAwareReranker",
    "FusionRetriever",
    "MultimodalRetriever",
    "QueryAnalyzer",
    "RetrievalRouter",
    "SceneDocument",
    "SceneDocumentBuilder",
    "SparseRetriever",
    "TextEncoder",
    "VisualEncoder",
    "VisualRetriever",
    "WeightedRankFusion",
]
