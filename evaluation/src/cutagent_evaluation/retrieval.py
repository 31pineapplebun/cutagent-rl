"""Evaluator-private M2A retrieval labels, metrics, and failure attribution."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from cutagent.schemas.base import Identifier, SchemaModel
from cutagent.schemas.media import TimeRange
from cutagent.schemas.retrieval import (
    RetrievalCandidate,
    RetrievalEvidenceType,
    RetrievalMetricsSummary,
    RetrievalQuery,
    RetrievalResponse,
)
from cutagent_evaluation.m1b_metrics import temporal_iou
from cutagent_evaluation.schemas import DatasetSplit

RetrievalQueryType = Literal["speech", "ocr", "entity", "semantic", "action", "hard_negative"]
RetrievalFailureType = Literal[
    "lexical_mismatch",
    "semantic_confusion",
    "wrong_visual_entity",
    "action_confusion",
    "OCR_failure",
    "ASR_failure",
    "perception_failure",
    "hard_negative_confusion",
    "scene_boundary_error",
]


class RelevantSceneGold(SchemaModel):
    video_id: Identifier
    scene_id: Identifier
    acceptable_time_range: TimeRange
    expected_evidence_types: tuple[RetrievalEvidenceType, ...] = Field(min_length=1)


class RetrievalCaseGold(SchemaModel):
    """Never imported by runtime CutAgent modules."""

    query_id: Identifier
    source_group_id: Identifier
    split: Literal[DatasetSplit.DEV] = DatasetSplit.DEV
    query_type: RetrievalQueryType
    relevant_scenes: tuple[RelevantSceneGold, ...] = Field(min_length=1)
    hard_negative: bool = False
    required_evidence_available: bool
    upstream_failure_type: Literal["ASR_failure", "OCR_failure", "perception_failure"] | None = None


class FrozenRetrievalCase(SchemaModel):
    query: RetrievalQuery
    gold: RetrievalCaseGold

    def model_post_init(self, __context: object) -> None:
        if self.query.query_id != self.gold.query_id:
            raise ValueError("query and private gold identifiers differ")


class QueryRetrievalMetric(SchemaModel):
    query_id: Identifier
    query_type: RetrievalQueryType
    hard_negative: bool
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    ndcg_at_10: float = Field(ge=0, le=1)
    temporal_iou: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    failure_type: RetrievalFailureType | None = None


def _identity(candidate: RetrievalCandidate) -> tuple[str, str]:
    return candidate.video_id, candidate.scene_id


def _ndcg(relevance: list[int], relevant_count: int, k: int = 10) -> float:
    discounted_gain = sum(value / math.log2(rank + 2) for rank, value in enumerate(relevance[:k]))
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(relevant_count, k)))
    return discounted_gain / ideal if ideal else 0.0


def _failure_type(
    method: str, gold: RetrievalCaseGold, reciprocal_rank: float
) -> RetrievalFailureType | None:
    if reciprocal_rank > 0:
        return None
    if not gold.required_evidence_available:
        return gold.upstream_failure_type or "perception_failure"
    if gold.hard_negative:
        return "hard_negative_confusion"
    if gold.query_type == "action":
        return "action_confusion"
    if gold.query_type == "entity" and method == "visual":
        return "wrong_visual_entity"
    if method == "bm25":
        return "lexical_mismatch"
    return "semantic_confusion"


def evaluate_response(response: RetrievalResponse, gold: RetrievalCaseGold) -> QueryRetrievalMetric:
    relevant = {(item.video_id, item.scene_id) for item in gold.relevant_scenes}
    ranked = [_identity(candidate) for candidate in response.candidates]
    relevance = [int(identity in relevant) for identity in ranked]
    first_rank = next((rank for rank, value in enumerate(relevance, start=1) if value), None)
    reciprocal_rank = 1.0 / first_rank if first_rank is not None else 0.0
    if response.candidates:
        first = response.candidates[0]
        temporal = max(
            (
                temporal_iou(first.time_range, item.acceptable_time_range)
                for item in gold.relevant_scenes
                if item.video_id == first.video_id
            ),
            default=0.0,
        )
    else:
        temporal = 0.0
    return QueryRetrievalMetric(
        query_id=gold.query_id,
        query_type=gold.query_type,
        hard_negative=gold.hard_negative,
        recall_at_1=float(any(relevance[:1])),
        recall_at_5=float(any(relevance[:5])),
        recall_at_10=float(any(relevance[:10])),
        reciprocal_rank=reciprocal_rank,
        ndcg_at_10=_ndcg(relevance, len(relevant)),
        temporal_iou=temporal,
        latency_ms=response.latency_ms,
        failure_type=_failure_type(response.method, gold, reciprocal_rank),
    )


def summarize_metrics(
    method: str, metrics: tuple[QueryRetrievalMetric, ...]
) -> RetrievalMetricsSummary:
    if not metrics:
        return RetrievalMetricsSummary(
            method=method,
            query_count=0,
            recall_at_1=0,
            recall_at_5=0,
            recall_at_10=0,
            mrr=0,
            ndcg_at_10=0,
            mean_temporal_iou=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
        )
    latencies = sorted(item.latency_ms for item in metrics)
    p95_index = min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)
    count = len(metrics)
    return RetrievalMetricsSummary(
        method=method,
        query_count=count,
        recall_at_1=sum(item.recall_at_1 for item in metrics) / count,
        recall_at_5=sum(item.recall_at_5 for item in metrics) / count,
        recall_at_10=sum(item.recall_at_10 for item in metrics) / count,
        mrr=sum(item.reciprocal_rank for item in metrics) / count,
        ndcg_at_10=sum(item.ndcg_at_10 for item in metrics) / count,
        mean_temporal_iou=sum(item.temporal_iou for item in metrics) / count,
        latency_p50_ms=statistics.median(latencies),
        latency_p95_ms=latencies[p95_index],
    )


def evaluate_method(
    method: str,
    responses: Mapping[str, RetrievalResponse],
    gold_by_query: Mapping[str, RetrievalCaseGold],
) -> tuple[RetrievalMetricsSummary, tuple[QueryRetrievalMetric, ...]]:
    if set(responses) != set(gold_by_query):
        raise ValueError("every retrieval method must evaluate the same frozen query IDs")
    metrics = tuple(
        evaluate_response(responses[query_id], gold_by_query[query_id])
        for query_id in sorted(gold_by_query)
    )
    return summarize_metrics(method, metrics), metrics
