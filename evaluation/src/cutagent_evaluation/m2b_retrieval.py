"""Evaluator-private M2B validation labels, router analysis, and failure attribution."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from cutagent.schemas.base import Identifier, SchemaModel
from cutagent.schemas.retrieval import (
    AdaptiveRetrievalResult,
    QueryIntent,
    RetrievalMetricsSummary,
    RetrievalQuery,
    RetrievalResponse,
)
from cutagent_evaluation.m1b_metrics import temporal_iou
from cutagent_evaluation.retrieval import (
    QueryRetrievalMetric,
    RelevantSceneGold,
    RetrievalFailureType,
    RetrievalQueryType,
    summarize_metrics,
)
from cutagent_evaluation.schemas import DatasetSplit

HardNegativeCategory = Literal[
    "same_entity_different_action",
    "same_text_different_scene",
    "semantically_similar_transcript",
    "visually_similar_distractor",
]
M2BFailureAttribution = Literal[
    "upstream_evidence_absent",
    "candidate_generation_miss",
    "router_wrong_channel_mix",
    "reranker_reorder_error",
    "native_verifier_failure",
    "native_verifier_hallucination",
    "scene_boundary_failure",
    "hard_negative_ambiguity",
]


class M2BRetrievalCaseGold(SchemaModel):
    """Validation-only labels; runtime modules must never import this type."""

    query_id: Identifier
    source_group_id: Identifier
    split: Literal[DatasetSplit.VALIDATION] = DatasetSplit.VALIDATION
    query_type: RetrievalQueryType
    relevant_scenes: tuple[RelevantSceneGold, ...] = Field(min_length=1)
    hard_negative_category: HardNegativeCategory | None = None
    required_evidence_available: bool
    expected_route: QueryIntent


class FrozenM2BValidationCase(SchemaModel):
    query: RetrievalQuery
    gold: M2BRetrievalCaseGold

    def model_post_init(self, __context: object) -> None:
        if self.query.query_id != self.gold.query_id:
            raise ValueError("public query and private M2B Gold identifiers differ")


class M2BFailureRecord(SchemaModel):
    query_id: Identifier
    attribution: M2BFailureAttribution
    details: str


class NativeEffectSummary(SchemaModel):
    invoked_queries: int = Field(ge=0)
    qwen_calls: int = Field(ge=0)
    corrected: int = Field(ge=0)
    no_effect: int = Field(ge=0)
    harmed: int = Field(ge=0)


def _identities(response: RetrievalResponse, depth: int = 10) -> tuple[tuple[str, str], ...]:
    return tuple((item.video_id, item.scene_id) for item in response.candidates[:depth])


def _is_relevant(response: RetrievalResponse, gold: M2BRetrievalCaseGold, depth: int) -> bool:
    relevant = {(item.video_id, item.scene_id) for item in gold.relevant_scenes}
    return any(identity in relevant for identity in _identities(response, depth))


def evaluate_m2b_method(
    method: str,
    responses: Mapping[str, RetrievalResponse],
    gold_by_query: Mapping[str, M2BRetrievalCaseGold],
) -> tuple[RetrievalMetricsSummary, tuple[QueryRetrievalMetric, ...]]:
    if set(responses) != set(gold_by_query):
        raise ValueError("every M2B method must use the same held-out validation query IDs")
    metrics: list[QueryRetrievalMetric] = []
    for query_id in sorted(gold_by_query):
        private = gold_by_query[query_id]
        response = responses[query_id]
        relevant = {(item.video_id, item.scene_id) for item in private.relevant_scenes}
        ranked = _identities(response, 10)
        relevance = [int(identity in relevant) for identity in ranked]
        first_rank = next((rank for rank, value in enumerate(relevance, 1) if value), None)
        reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank
        discounted_gain = sum(value / math.log2(rank + 2) for rank, value in enumerate(relevance))
        ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), 10)))
        top = response.candidates[0] if response.candidates else None
        temporal = (
            0.0
            if top is None
            else max(
                (
                    temporal_iou(top.time_range, item.acceptable_time_range)
                    for item in private.relevant_scenes
                    if item.video_id == top.video_id
                ),
                default=0.0,
            )
        )
        failure: RetrievalFailureType | None
        if reciprocal_rank > 0:
            failure = None
        elif not private.required_evidence_available:
            failure = "perception_failure"
        elif private.hard_negative_category is not None:
            failure = "hard_negative_confusion"
        elif private.query_type == "action":
            failure = "action_confusion"
        elif response.method == "bm25":
            failure = "lexical_mismatch"
        else:
            failure = "semantic_confusion"
        metrics.append(
            QueryRetrievalMetric(
                query_id=query_id,
                query_type=private.query_type,
                hard_negative=private.hard_negative_category is not None,
                recall_at_1=float(any(relevance[:1])),
                recall_at_5=float(any(relevance[:5])),
                recall_at_10=float(any(relevance[:10])),
                reciprocal_rank=reciprocal_rank,
                ndcg_at_10=discounted_gain / ideal if ideal else 0.0,
                temporal_iou=temporal,
                latency_ms=response.latency_ms,
                failure_type=failure,
            )
        )
    metric_tuple = tuple(metrics)
    return summarize_metrics(method, metric_tuple), metric_tuple


def router_confusion(
    results: Mapping[str, AdaptiveRetrievalResult],
    gold: Mapping[str, M2BRetrievalCaseGold],
) -> dict[str, object]:
    if set(results) != set(gold):
        raise ValueError("router evaluation query sets differ")
    matrix: Counter[tuple[str, str]] = Counter()
    for query_id in sorted(gold):
        matrix[(gold[query_id].expected_route, results[query_id].plan.primary_intent)] += 1
    return {
        "distribution": dict(
            sorted(Counter(item.plan.primary_intent for item in results.values()).items())
        ),
        "confusion": {
            f"{expected}->{observed}": count
            for (expected, observed), count in sorted(matrix.items())
        },
        "accuracy": sum(
            gold[query_id].expected_route == results[query_id].plan.primary_intent
            for query_id in gold
        )
        / len(gold),
    }


def native_effect_summary(
    before: Mapping[str, RetrievalResponse],
    after: Mapping[str, AdaptiveRetrievalResult],
    gold: Mapping[str, M2BRetrievalCaseGold],
) -> NativeEffectSummary:
    corrected = harmed = no_effect = calls = invoked = 0
    for query_id, result in after.items():
        if not result.native_verifications:
            continue
        invoked += 1
        calls += len(result.native_verifications)
        was_correct = _is_relevant(before[query_id], gold[query_id], 1)
        now_correct = _is_relevant(result.response, gold[query_id], 1)
        if not was_correct and now_correct:
            corrected += 1
        elif was_correct and not now_correct:
            harmed += 1
        else:
            no_effect += 1
    return NativeEffectSummary(
        invoked_queries=invoked,
        qwen_calls=calls,
        corrected=corrected,
        no_effect=no_effect,
        harmed=harmed,
    )


def attribute_failures(
    *,
    uniform: Mapping[str, RetrievalResponse],
    routed: Mapping[str, RetrievalResponse],
    reranked: Mapping[str, RetrievalResponse],
    native: Mapping[str, AdaptiveRetrievalResult],
    gold: Mapping[str, M2BRetrievalCaseGold],
) -> tuple[M2BFailureRecord, ...]:
    records: list[M2BFailureRecord] = []
    for query_id, private in sorted(gold.items()):
        final = native[query_id]
        top_temporal = 0.0
        if final.response.candidates:
            top = final.response.candidates[0]
            top_temporal = max(
                (
                    temporal_iou(top.time_range, item.acceptable_time_range)
                    for item in private.relevant_scenes
                    if top.video_id == item.video_id
                ),
                default=0.0,
            )
        if _is_relevant(final.response, private, 1) and top_temporal >= 0.5:
            continue
        relevant_identities = {(item.video_id, item.scene_id) for item in private.relevant_scenes}
        native_harmed = (
            bool(final.native_verifications)
            and _is_relevant(reranked[query_id], private, 1)
            and not _is_relevant(final.response, private, 1)
        )
        native_supported_wrong_scene = any(
            item.status == "supports" and (item.video_id, item.scene_id) not in relevant_identities
            for item in final.native_verifications
        )
        attribution: M2BFailureAttribution
        if native_harmed:
            attribution = (
                "native_verifier_hallucination"
                if native_supported_wrong_scene
                else "native_verifier_failure"
            )
        elif not private.required_evidence_available:
            attribution = "upstream_evidence_absent"
        elif _is_relevant(uniform[query_id], private, 10) and not _is_relevant(
            routed[query_id], private, 10
        ):
            attribution = "router_wrong_channel_mix"
        elif not _is_relevant(routed[query_id], private, 10):
            attribution = "candidate_generation_miss"
        elif _is_relevant(routed[query_id], private, 1) and not _is_relevant(
            reranked[query_id], private, 1
        ):
            attribution = "reranker_reorder_error"
        elif _is_relevant(final.response, private, 1) and top_temporal < 0.5:
            attribution = "scene_boundary_failure"
        elif final.native_verifications and native_supported_wrong_scene:
            attribution = "native_verifier_hallucination"
        elif final.native_verifications:
            attribution = "native_verifier_failure"
        elif private.hard_negative_category is not None:
            attribution = "hard_negative_ambiguity"
        else:
            attribution = "reranker_reorder_error"
        records.append(
            M2BFailureRecord(
                query_id=query_id,
                attribution=attribution,
                details="final top-1 scene or temporal localization is incorrect",
            )
        )
    return tuple(records)
