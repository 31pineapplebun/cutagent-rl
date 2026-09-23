"""Sparse, dense, visual, and rank-fusion M2A retrievers."""

from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np

from cutagent.retrieval.documents import SceneDocument
from cutagent.retrieval.protocols import MultimodalRetriever, TextEncoder, VisualEncoder
from cutagent.retrieval.store import RetrievalMemory
from cutagent.schemas.perception import EvidenceRef
from cutagent.schemas.retrieval import (
    AppliedRetrievalFilters,
    DocumentField,
    RetrievalCandidate,
    RetrievalEvidence,
    RetrievalEvidenceType,
    RetrievalQuery,
    RetrievalResponse,
)


def _overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def _document_evidence_types(document: SceneDocument) -> set[RetrievalEvidenceType]:
    output = {
        item.evidence_type for evidence in document.field_evidence.values() for item in evidence
    }
    if document.keyframe_evidence:
        output.add("keyframe")
    return output


def _eligible(query: RetrievalQuery, document: SceneDocument) -> bool:
    if query.video_id is not None and query.video_id != document.video_id:
        return False
    if query.approximate_time_range is not None and not _overlaps(
        query.approximate_time_range.start_ms,
        query.approximate_time_range.end_ms,
        document.time_range.start_ms,
        document.time_range.end_ms,
    ):
        return False
    available = _document_evidence_types(document)
    return all(required in available for required in query.required_evidence_types)


def _filters(query: RetrievalQuery) -> AppliedRetrievalFilters:
    return AppliedRetrievalFilters(
        video_id=query.video_id,
        approximate_time_range=query.approximate_time_range,
        required_evidence_types=query.required_evidence_types,
    )


def _unique_references(evidence: tuple[RetrievalEvidence, ...]) -> tuple[EvidenceRef, ...]:
    return tuple(dict.fromkeys(item.evidence_ref for item in evidence))


def _evidence_types(
    evidence: tuple[RetrievalEvidence, ...],
) -> tuple[RetrievalEvidenceType, ...]:
    order: tuple[RetrievalEvidenceType, ...] = (
        "transcript",
        "ocr",
        "structured_text",
        "keyframe",
    )
    found = {item.evidence_type for item in evidence}
    return tuple(item for item in order if item in found)


def _text_evidence(
    document: SceneDocument,
    field: DocumentField,
    score: float,
) -> tuple[RetrievalEvidence, ...]:
    output: list[RetrievalEvidence] = []
    seen: set[tuple[RetrievalEvidenceType, EvidenceRef]] = set()
    for item in document.field_evidence.get(field, ()):
        key = (item.evidence_type, item.reference)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            RetrievalEvidence(
                evidence_type=item.evidence_type,
                evidence_ref=item.reference,
                document_field=field,
                matched_text=item.text or None,
                score=score,
            )
        )
    return tuple(output)


def _candidate(
    *,
    document: SceneDocument,
    score: float,
    component_scores: dict[str, float],
    evidence: tuple[RetrievalEvidence, ...],
    rank: int,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        video_id=document.video_id,
        scene_id=document.scene_id,
        time_range=document.time_range,
        total_score=score,
        component_scores=component_scores,
        evidence_refs=_unique_references(evidence),
        evidence_types=_evidence_types(evidence),
        evidence=evidence,
        rank=rank,
    )


def _rank_indices(
    scores: Mapping[int, float], documents: tuple[SceneDocument, ...]
) -> tuple[tuple[int, float], ...]:
    return tuple(
        sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                documents[item[0]].video_id,
                documents[item[0]].scene_id,
            ),
        )
    )


class SparseRetriever:
    def __init__(self, field: DocumentField = "combined") -> None:
        self.field = field

    def search(self, query: RetrievalQuery, memory_ref: RetrievalMemory) -> RetrievalResponse:
        started = time.perf_counter()
        index = memory_ref.sparse_indices.get(self.field)
        if index is None:
            raise ValueError(f"sparse field is not indexed: {self.field}")
        raw_scores = index.scores(query.text)
        scores = {
            position: score
            for position, score in enumerate(raw_scores)
            if score > 0 and _eligible(query, memory_ref.documents[position])
        }
        candidates: list[RetrievalCandidate] = []
        for position, score in _rank_indices(scores, memory_ref.documents):
            document = memory_ref.documents[position]
            evidence = _text_evidence(document, self.field, score)
            if not evidence:
                continue
            candidates.append(
                _candidate(
                    document=document,
                    score=score,
                    component_scores={f"bm25_{self.field}": score},
                    evidence=evidence,
                    rank=len(candidates) + 1,
                )
            )
            if len(candidates) >= query.top_k:
                break
        return RetrievalResponse(
            query_id=query.query_id,
            method="bm25",
            candidates=tuple(candidates),
            applied_filters=_filters(query),
            latency_ms=(time.perf_counter() - started) * 1000,
            cache_hit=False,
        )


class DenseTextRetriever:
    def __init__(self, encoder: TextEncoder) -> None:
        self.encoder = encoder
        self._query_cache: dict[str, np.ndarray[tuple[int], np.dtype[np.float32]]] = {}

    def search(self, query: RetrievalQuery, memory_ref: RetrievalMemory) -> RetrievalResponse:
        started = time.perf_counter()
        if memory_ref.dense_embeddings is None:
            raise ValueError("dense text index is unavailable")
        query_vector = self._query_cache.get(query.text)
        cache_hit = query_vector is not None
        if query_vector is None:
            encoded = self.encoder.encode((query.text,))
            if encoded.shape[0] != 1:
                raise ValueError("text query encoder returned an unexpected row count")
            query_vector = encoded[0]
            self._query_cache[query.text] = query_vector
        similarities = memory_ref.dense_embeddings @ query_vector
        scores = {
            position: float(score)
            for position, score in enumerate(similarities)
            if _eligible(query, memory_ref.documents[position])
        }
        candidates: list[RetrievalCandidate] = []
        for position, score in _rank_indices(scores, memory_ref.documents):
            document = memory_ref.documents[position]
            evidence = _text_evidence(document, memory_ref.dense_document_field, score)
            if not evidence:
                continue
            candidates.append(
                _candidate(
                    document=document,
                    score=score,
                    component_scores={"dense_text": score},
                    evidence=evidence,
                    rank=len(candidates) + 1,
                )
            )
            if len(candidates) >= query.top_k:
                break
        return RetrievalResponse(
            query_id=query.query_id,
            method="dense_text",
            candidates=tuple(candidates),
            applied_filters=_filters(query),
            latency_ms=(time.perf_counter() - started) * 1000,
            cache_hit=cache_hit,
        )


class VisualRetriever:
    def __init__(self, encoder: VisualEncoder) -> None:
        self.encoder = encoder
        self._query_cache: dict[str, np.ndarray[tuple[int], np.dtype[np.float32]]] = {}

    def search(self, query: RetrievalQuery, memory_ref: RetrievalMemory) -> RetrievalResponse:
        started = time.perf_counter()
        if memory_ref.visual_embeddings is None:
            raise ValueError("visual index is unavailable")
        query_vector = self._query_cache.get(query.text)
        cache_hit = query_vector is not None
        if query_vector is None:
            encoded = self.encoder.encode_queries((query.text,))
            if encoded.shape[0] != 1:
                raise ValueError("visual query encoder returned an unexpected row count")
            query_vector = encoded[0]
            self._query_cache[query.text] = query_vector
        similarities = memory_ref.visual_embeddings @ query_vector
        best_by_document: dict[int, tuple[float, EvidenceRef]] = {}
        for row_index, row in enumerate(memory_ref.visual_rows):
            document = memory_ref.documents[row.document_index]
            if not _eligible(query, document):
                continue
            score = float(similarities[row_index])
            current = best_by_document.get(row.document_index)
            if current is None or score > current[0]:
                best_by_document[row.document_index] = (score, row.evidence_ref)
        candidates: list[RetrievalCandidate] = []
        ranked = _rank_indices(
            {position: value[0] for position, value in best_by_document.items()},
            memory_ref.documents,
        )
        for position, score in ranked[: query.top_k]:
            document = memory_ref.documents[position]
            evidence_ref = best_by_document[position][1]
            evidence = (
                RetrievalEvidence(
                    evidence_type="keyframe",
                    evidence_ref=evidence_ref,
                    score=score,
                ),
            )
            candidates.append(
                _candidate(
                    document=document,
                    score=score,
                    component_scores={"visual": score},
                    evidence=evidence,
                    rank=len(candidates) + 1,
                )
            )
        return RetrievalResponse(
            query_id=query.query_id,
            method="visual",
            candidates=tuple(candidates),
            applied_filters=_filters(query),
            latency_ms=(time.perf_counter() - started) * 1000,
            cache_hit=cache_hit,
        )


class FusionRetriever:
    """Reciprocal Rank Fusion; it never adds incompatible raw model scores."""

    def __init__(
        self,
        components: Mapping[str, MultimodalRetriever],
        *,
        rank_constant: int = 60,
    ) -> None:
        if len(components) < 2:
            raise ValueError("fusion requires at least two retrieval components")
        self.components = dict(components)
        self.rank_constant = rank_constant

    def search(self, query: RetrievalQuery, memory_ref: RetrievalMemory) -> RetrievalResponse:
        started = time.perf_counter()
        expanded = query.model_copy(update={"top_k": min(100, len(memory_ref.documents))})
        responses = {
            name: retriever.search(expanded, memory_ref)
            for name, retriever in self.components.items()
        }
        scores: dict[tuple[str, str], float] = {}
        candidates_by_identity: dict[tuple[str, str], list[RetrievalCandidate]] = {}
        for response in responses.values():
            for candidate in response.candidates:
                identity = (candidate.video_id, candidate.scene_id)
                scores[identity] = scores.get(identity, 0.0) + 1.0 / (
                    self.rank_constant + candidate.rank
                )
                candidates_by_identity.setdefault(identity, []).append(candidate)
        ranked_identities = sorted(scores, key=lambda item: (-scores[item], item[0], item[1]))
        output: list[RetrievalCandidate] = []
        for identity in ranked_identities[: query.top_k]:
            component_candidates = candidates_by_identity[identity]
            evidence = tuple(
                dict.fromkeys(
                    item for candidate in component_candidates for item in candidate.evidence
                )
            )
            document = next(
                item for item in memory_ref.documents if (item.video_id, item.scene_id) == identity
            )
            component_scores = {
                component_name: candidate.total_score
                for component_name, response in responses.items()
                for candidate in response.candidates
                if (candidate.video_id, candidate.scene_id) == identity
            }
            output.append(
                _candidate(
                    document=document,
                    score=scores[identity],
                    component_scores=component_scores,
                    evidence=evidence,
                    rank=len(output) + 1,
                )
            )
        return RetrievalResponse(
            query_id=query.query_id,
            method="fusion",
            candidates=tuple(output),
            applied_filters=_filters(query),
            latency_ms=(time.perf_counter() - started) * 1000,
            cache_hit=all(response.cache_hit for response in responses.values()),
        )
