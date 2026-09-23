"""M2A public schema contracts and private-field exclusions."""

import json

import pytest
from pydantic import ValidationError

from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import EvidenceRef
from cutagent.schemas.retrieval import (
    AdaptiveRetrievalConfig,
    QueryAnalysis,
    RetrievalCandidate,
    RetrievalEvidence,
    RetrievalPlan,
    RetrievalQuery,
)


def test_retrieval_query_forbids_evaluator_fields() -> None:
    for field in ("split", "source_group_id", "BenchmarkGold", "evaluator_metadata"):
        with pytest.raises(ValidationError):
            RetrievalQuery.model_validate(
                {"query_id": "query-1", "text": "red square", field: "private"}
            )


def test_candidate_evidence_summary_is_exact() -> None:
    reference = EvidenceRef(
        artifact_id="frame-1",
        evidence_kind="keyframe",
        segment_id="scene-1",
        observed_ms=100,
    )
    evidence = RetrievalEvidence(evidence_type="keyframe", evidence_ref=reference, score=0.8)
    candidate = RetrievalCandidate(
        video_id="video-1",
        scene_id="scene-1",
        time_range=TimeRange(start_ms=0, end_ms=200),
        total_score=0.8,
        evidence_refs=(reference,),
        evidence_types=("keyframe",),
        evidence=(evidence,),
        rank=1,
    )
    serialized = json.dumps(candidate.model_dump(mode="json"))
    assert "source_group_id" not in serialized
    assert "split" not in serialized
    with pytest.raises(ValidationError, match="exactly summarize"):
        RetrievalCandidate.model_validate(
            {**candidate.model_dump(mode="json"), "evidence_types": ["ocr"]}
        )


def test_query_filters_are_explicit_and_unique() -> None:
    with pytest.raises(ValidationError, match="unique"):
        RetrievalQuery(
            query_id="query-1",
            text="find exit",
            required_evidence_types=("ocr", "ocr"),
        )


def test_m2b_plan_and_analysis_reject_private_evaluator_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        QueryAnalysis.model_validate(
            {
                "analyzer_version": "v1",
                "primary_intent": "ambiguous",
                "intent_scores": {
                    "speech_or_quote": 0,
                    "visible_text": 0,
                    "static_visual_entity": 0,
                    "semantic_scene": 0,
                    "action_or_motion": 0,
                    "ambiguous": 1,
                },
                "reasons": ("fallback",),
                "query_type": "hard_negative",
            },
        )
    config = AdaptiveRetrievalConfig(native_video_enabled=False)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RetrievalPlan.model_validate(
            {
                "query_id": "query-plan",
                "analyzer_version": config.analyzer_version,
                "routing_policy_version": config.routing_policy_version,
                "primary_intent": "ambiguous",
                "enabled_channels": ("dense_text",),
                "channel_weights": {"dense_text": 1.0},
                "candidate_depth": 10,
                "rerank_depth": 5,
                "reranking_policy": "evidence_rules",
                "native_video_policy": "disabled",
                "native_video_depth": 0,
                "reasons": ("public query only",),
                "split": "validation",
            }
        )
