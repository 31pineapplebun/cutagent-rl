"""Weighted fusion, evidence reranking, and on-demand verification contracts."""

from pathlib import Path

from cutagent.core.artifacts import ArtifactRef
from cutagent.retrieval.adaptive import AdaptiveHybridRetriever
from cutagent.retrieval.fusion import WeightedRankFusion
from cutagent.retrieval.native_video import (
    SceneClipAsset,
    build_native_verification_cache_key,
)
from cutagent.retrieval.protocols import MultimodalRetriever
from cutagent.retrieval.query import QueryAnalyzer, RetrievalRouter
from cutagent.retrieval.reranking import EvidenceAwareReranker
from cutagent.retrieval.retrievers import DenseTextRetriever, SparseRetriever, VisualRetriever
from cutagent.retrieval.store import RetrievalMemory
from cutagent.schemas.perception import EvidenceRef, QwenModelSpec
from cutagent.schemas.retrieval import (
    AdaptiveRetrievalConfig,
    NativeVerificationStatus,
    NativeVideoVerification,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalQuery,
)
from tests.retrieval_fixtures import build_fake_memory


def _memory_and_fusion(tmp_path: Path) -> tuple[RetrievalMemory, WeightedRankFusion]:
    raw_memory, text, visual = build_fake_memory(tmp_path)
    assert isinstance(raw_memory, RetrievalMemory)
    components: dict[RetrievalChannel, MultimodalRetriever] = {
        "bm25_transcript": SparseRetriever("transcript"),
        "bm25_ocr": SparseRetriever("ocr"),
        "bm25_structured": SparseRetriever("structured_semantic"),
        "bm25_combined": SparseRetriever("combined"),
        "dense_text": DenseTextRetriever(text),
        "visual": VisualRetriever(visual),
    }
    return raw_memory, WeightedRankFusion(components)


def test_weighted_rrf_and_evidence_reranking_preserve_exact_ocr(tmp_path: Path) -> None:
    memory, fusion = _memory_and_fusion(tmp_path)
    query = RetrievalQuery(
        query_id="q-visible",
        text='Find the visible exact text "SALE 42"',
        top_k=2,
    )
    analysis = QueryAnalyzer().analyze(query)
    plan = RetrievalRouter(AdaptiveRetrievalConfig()).plan(query, analysis)
    fused = fusion.fuse(query, memory, plan).response
    reranked, traces = EvidenceAwareReranker().rerank(
        query=query,
        analysis=analysis,
        plan=plan,
        response=fused,
    )
    assert reranked.candidates[0].scene_id == "scene-red"
    assert "exact_ocr_phrase" in traces[0].matched_features
    assert any(key.startswith("rank.") for key in reranked.candidates[0].component_scores)


class _FakeVerifier:
    def __init__(self, raw_output: ArtifactRef) -> None:
        self.raw_output = raw_output
        self.calls = 0

    def verify(
        self,
        *,
        query: RetrievalQuery,
        candidate: RetrievalCandidate,
        scene_clip: SceneClipAsset,
    ) -> NativeVideoVerification:
        self.calls += 1
        status: NativeVerificationStatus = (
            "supports" if candidate.scene_id == "scene-red" else "contradicts"
        )
        return NativeVideoVerification(
            query_id=query.query_id,
            video_id=candidate.video_id,
            scene_id=candidate.scene_id,
            status=status,
            explanation=f"synthetic verifier {status}",
            evidence_ref=EvidenceRef(
                artifact_id=scene_clip.artifact.artifact_id,
                evidence_kind="scene_clip",
                segment_id=candidate.scene_id,
                time_range=candidate.time_range,
            ),
            raw_output_artifact=self.raw_output,
            prompt_version="m2b-native-verifier-v1",
            model_id="Qwen/Qwen3-VL-4B-Instruct",
            model_revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
            latency_ms=7,
            cache_hit=False,
        )


def _scene_clips(tmp_path: Path, memory: RetrievalMemory) -> dict[tuple[str, str], SceneClipAsset]:
    output = {}
    for document in memory.documents:
        path = tmp_path / f"{document.scene_id}.mp4"
        path.write_bytes(document.scene_id.encode())
        artifact = ArtifactRef.from_path(
            path,
            artifact_id=f"clip-{document.scene_id}",
            media_type="video/mp4",
        )
        output[(document.video_id, document.scene_id)] = SceneClipAsset(
            video_id=document.video_id,
            scene_id=document.scene_id,
            time_range=document.time_range,
            artifact=artifact,
            path=path,
        )
    return output


def test_adaptive_retriever_invokes_native_video_only_for_motion(tmp_path: Path) -> None:
    memory, fusion = _memory_and_fusion(tmp_path)
    raw = tmp_path / "raw.json"
    raw.write_text("{}", encoding="utf-8")
    verifier = _FakeVerifier(
        ArtifactRef.from_path(raw, artifact_id="raw-native", media_type="application/json")
    )
    retriever = AdaptiveHybridRetriever(
        config=AdaptiveRetrievalConfig(native_score_margin=1.0),
        fusion=fusion,
        verifier=verifier,
        scene_clips=_scene_clips(tmp_path, memory),
    )
    action = retriever.search_with_trace(
        RetrievalQuery(query_id="q-action", text="red square moving right", top_k=2),
        memory,
    )
    assert verifier.calls == 2
    assert action.native_verifications
    assert action.response.candidates[0].scene_id == "scene-red"
    assert "native_video_verification" in action.response.candidates[0].evidence_types

    entity = retriever.search_with_trace(
        RetrievalQuery(query_id="q-entity", text="Find a red square object", top_k=2),
        memory,
    )
    assert verifier.calls == 2
    assert entity.native_verifications == ()


def test_native_cache_key_tracks_prompt_routing_and_index_versions(tmp_path: Path) -> None:
    memory, fusion = _memory_and_fusion(tmp_path)
    query = RetrievalQuery(query_id="q-cache", text="red square moving right", top_k=2)
    plan = RetrievalRouter(AdaptiveRetrievalConfig()).plan(query, QueryAnalyzer().analyze(query))
    candidate = fusion.fuse(query, memory, plan).response.candidates[0]
    clip = _scene_clips(tmp_path, memory)[(candidate.video_id, candidate.scene_id)]
    base = AdaptiveRetrievalConfig()
    key = build_native_verification_cache_key(
        query=query,
        candidate=candidate,
        scene_clip=clip,
        config=base,
        model=QwenModelSpec(),
        index_id=memory.manifest.index_id,
        native_video_fps=2.0,
        backend_version="fake-v1",
    )
    changed = build_native_verification_cache_key(
        query=query,
        candidate=candidate,
        scene_clip=clip,
        config=base.model_copy(update={"native_verifier_prompt_version": "prompt-v2"}),
        model=QwenModelSpec(),
        index_id=memory.manifest.index_id,
        native_video_fps=2.0,
        backend_version="fake-v1",
    )
    assert key != changed
