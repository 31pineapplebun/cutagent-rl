"""Private retrieval Gold and metric behavior."""

from typing import cast

from cutagent_evaluation.retrieval import (
    FrozenRetrievalCase,
    RelevantSceneGold,
    RetrievalCaseGold,
    evaluate_method,
)

from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import EvidenceRef
from cutagent.schemas.retrieval import (
    AppliedRetrievalFilters,
    RetrievalCandidate,
    RetrievalEvidence,
    RetrievalQuery,
    RetrievalResponse,
)


def _response(query_id: str, scene_ids: tuple[str, ...]) -> RetrievalResponse:
    candidates = []
    for rank, scene_id in enumerate(scene_ids, start=1):
        reference = EvidenceRef(
            artifact_id=f"frame-{scene_id}",
            evidence_kind="keyframe",
            segment_id=scene_id,
            observed_ms=500,
        )
        evidence = RetrievalEvidence(
            evidence_type="keyframe", evidence_ref=reference, score=1 / rank
        )
        candidates.append(
            RetrievalCandidate(
                video_id="video-1",
                scene_id=scene_id,
                time_range=TimeRange(start_ms=0, end_ms=1000),
                total_score=1 / rank,
                evidence_refs=(reference,),
                evidence_types=("keyframe",),
                evidence=(evidence,),
                rank=rank,
            )
        )
    return RetrievalResponse(
        query_id=query_id,
        method="visual",
        candidates=tuple(candidates),
        applied_filters=AppliedRetrievalFilters(),
        latency_ms=2,
        cache_hit=False,
    )


def test_private_gold_and_metrics_stay_offline() -> None:
    query = RetrievalQuery(query_id="query-1", text="red square")
    gold = RetrievalCaseGold(
        query_id="query-1",
        source_group_id="source-1",
        query_type="entity",
        relevant_scenes=(
            RelevantSceneGold(
                video_id="video-1",
                scene_id="scene-good",
                acceptable_time_range=TimeRange(start_ms=0, end_ms=1000),
                expected_evidence_types=("keyframe",),
            ),
        ),
        required_evidence_available=True,
    )
    case = FrozenRetrievalCase(query=query, gold=gold)
    assert "source_group_id" not in case.query.model_dump_json()
    summary, metrics = evaluate_method(
        "visual",
        {"query-1": _response("query-1", ("scene-bad", "scene-good"))},
        {"query-1": gold},
    )
    assert summary.recall_at_1 == 0
    assert summary.recall_at_5 == 1
    assert summary.mrr == 0.5
    assert metrics[0].failure_type is None


def test_evaluator_rejects_different_query_sets() -> None:
    try:
        evaluate_method("visual", {}, {"missing": cast(RetrievalCaseGold, object())})
    except ValueError as error:
        assert "same frozen query IDs" in str(error)
    else:
        raise AssertionError("mismatched query IDs must fail")
