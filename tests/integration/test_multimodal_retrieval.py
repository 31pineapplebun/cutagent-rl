"""M2A backends search the same memory with stable evidence-linked rankings."""

from pathlib import Path

from cutagent.core.artifacts import ArtifactRef
from cutagent.retrieval.index import KeyframeAsset, RetrievalIndexBuilder
from cutagent.retrieval.retrievers import (
    DenseTextRetriever,
    FusionRetriever,
    SparseRetriever,
    VisualRetriever,
)
from cutagent.retrieval.store import RetrievalMemory
from cutagent.schemas.media import TimeRange
from cutagent.schemas.retrieval import RetrievalConfig, RetrievalQuery
from tests.retrieval_fixtures import FakeTextEncoder, FakeVisualEncoder, sample_world_states


def _memory(
    tmp_path: Path,
) -> tuple[RetrievalMemory, FakeTextEncoder, FakeVisualEncoder]:
    assets = []
    for artifact_id in ("frame-red", "frame-blue"):
        path = tmp_path / f"{artifact_id}.jpg"
        path.write_text(artifact_id, encoding="utf-8")
        assets.append(
            KeyframeAsset(
                ArtifactRef.from_path(path, artifact_id=artifact_id, media_type="image/jpeg"),
                path,
            )
        )
    text = FakeTextEncoder()
    visual = FakeVisualEncoder()
    memory = RetrievalIndexBuilder(tmp_path / "index").build(
        world_states=sample_world_states(),
        keyframe_assets=tuple(assets),
        config=RetrievalConfig(),
        text_encoder=text,
        visual_encoder=visual,
    )
    return memory, text, visual


def test_independent_and_fused_retrievers(tmp_path: Path) -> None:
    memory, text, visual = _memory(tmp_path)
    query = RetrievalQuery(query_id="query-red", text="red square", top_k=2)
    sparse = SparseRetriever("combined")
    dense = DenseTextRetriever(text)
    visual_retriever = VisualRetriever(visual)
    assert sparse.search(query, memory).candidates[0].scene_id == "scene-red"
    assert dense.search(query, memory).candidates[0].scene_id == "scene-red"
    visual_response = visual_retriever.search(query, memory)
    assert visual_response.candidates[0].scene_id == "scene-red"
    assert visual_response.candidates[0].evidence_types == ("keyframe",)
    fusion = FusionRetriever({"bm25": sparse, "dense_text": dense, "visual": visual_retriever})
    first = fusion.search(query, memory)
    second = fusion.search(query, memory)
    assert first.candidates[0].scene_id == "scene-red"
    assert [item.scene_id for item in first.candidates] == [
        item.scene_id for item in second.candidates
    ]
    assert second.cache_hit is False  # BM25 has no query-embedding cache.


def test_observable_filters_apply_before_ranking(tmp_path: Path) -> None:
    memory, _, visual = _memory(tmp_path)
    query = RetrievalQuery(
        query_id="query-filter",
        text="red square",
        video_id="video-sample",
        approximate_time_range=TimeRange(start_ms=1000, end_ms=1800),
        required_evidence_types=("keyframe",),
    )
    response = VisualRetriever(visual).search(query, memory)
    assert [item.scene_id for item in response.candidates] == ["scene-blue"]
    assert response.applied_filters.approximate_time_range == query.approximate_time_range


def test_sparse_fields_expose_the_contributing_evidence(tmp_path: Path) -> None:
    memory, _, _ = _memory(tmp_path)
    ocr_response = SparseRetriever("ocr").search(
        RetrievalQuery(query_id="query-ocr", text="SALE 42"), memory
    )
    assert ocr_response.candidates[0].evidence_types == ("ocr",)
    transcript_response = SparseRetriever("transcript").search(
        RetrievalQuery(query_id="query-speech", text="moves right"), memory
    )
    assert transcript_response.candidates[0].evidence_types == ("transcript",)
