"""Inspectable in-memory handle for a local persistent M2A index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cutagent.retrieval.documents import SceneDocument
from cutagent.retrieval.sparse import BM25Index
from cutagent.schemas.perception import EvidenceRef
from cutagent.schemas.retrieval import DocumentField, RetrievalIndexManifest


@dataclass(frozen=True, slots=True)
class VisualVectorRow:
    document_index: int
    evidence_ref: EvidenceRef


@dataclass(frozen=True, slots=True)
class RetrievalMemory:
    """Index data is local and explicit; no external vector service is hidden."""

    manifest: RetrievalIndexManifest
    documents: tuple[SceneDocument, ...]
    sparse_indices: dict[DocumentField, BM25Index]
    dense_document_field: DocumentField
    dense_embeddings: npt.NDArray[np.float32] | None
    visual_embeddings: npt.NDArray[np.float32] | None
    visual_rows: tuple[VisualVectorRow, ...]
    cache_hits: dict[str, bool]
