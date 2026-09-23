"""Adaptive M2B retrieval orchestration without Agent or learned reranking."""

from __future__ import annotations

import time
from collections.abc import Mapping

from cutagent.core.run_context import config_sha256
from cutagent.retrieval.fusion import WeightedRankFusion
from cutagent.retrieval.native_video import SceneClipAsset
from cutagent.retrieval.protocols import MotionEvidenceVerifier
from cutagent.retrieval.query import QueryAnalyzer, RetrievalRouter, RoutingStrategy
from cutagent.retrieval.reranking import EvidenceAwareReranker
from cutagent.retrieval.store import RetrievalMemory
from cutagent.schemas.perception import EvidenceRef
from cutagent.schemas.retrieval import (
    AdaptiveRetrievalConfig,
    AdaptiveRetrievalResult,
    CandidateRerankTrace,
    NativeVideoVerification,
    RetrievalCandidate,
    RetrievalEvidence,
    RetrievalEvidenceType,
    RetrievalQuery,
    RetrievalResponse,
)


def _evidence_refs(evidence: tuple[RetrievalEvidence, ...]) -> tuple[EvidenceRef, ...]:
    return tuple(dict.fromkeys(item.evidence_ref for item in evidence))


def _evidence_types(
    evidence: tuple[RetrievalEvidence, ...],
) -> tuple[RetrievalEvidenceType, ...]:
    order: tuple[RetrievalEvidenceType, ...] = (
        "transcript",
        "ocr",
        "structured_text",
        "keyframe",
        "native_video_verification",
    )
    present = {item.evidence_type for item in evidence}
    return tuple(item for item in order if item in present)


class AdaptiveHybridRetriever:
    """Route, rank, optionally verify motion, and return an auditable trace."""

    def __init__(
        self,
        *,
        config: AdaptiveRetrievalConfig,
        fusion: WeightedRankFusion,
        strategy: RoutingStrategy = "query_aware",
        evidence_reranking: bool = True,
        native_video: bool = True,
        analyzer: QueryAnalyzer | None = None,
        reranker: EvidenceAwareReranker | None = None,
        verifier: MotionEvidenceVerifier | None = None,
        scene_clips: Mapping[tuple[str, str], SceneClipAsset] | None = None,
    ) -> None:
        self.config = config
        self.fusion = fusion
        self.strategy = strategy
        self.evidence_reranking = evidence_reranking
        self.native_video = native_video
        self.analyzer = analyzer or QueryAnalyzer()
        self.router = RetrievalRouter(config)
        self.reranker = reranker or EvidenceAwareReranker()
        self.verifier = verifier
        self.scene_clips = dict(scene_clips or {})

    def _should_verify(
        self,
        candidates: tuple[RetrievalCandidate, ...],
        traces: tuple[CandidateRerankTrace, ...],
    ) -> bool:
        if len(candidates) < 2:
            return False
        margin = abs(candidates[0].total_score - candidates[1].total_score)
        missing_action = any(
            "missing_requested_action_evidence" in item.penalties
            for item in traces[: self.config.native_video_depth]
        )
        return margin <= self.config.native_score_margin or missing_action

    @staticmethod
    def _apply_native_verification(
        response: RetrievalResponse,
        verifications: tuple[NativeVideoVerification, ...],
    ) -> RetrievalResponse:
        by_identity = {(item.video_id, item.scene_id): item for item in verifications}
        adjusted: list[RetrievalCandidate] = []
        for candidate in response.candidates:
            verification = by_identity.get((candidate.video_id, candidate.scene_id))
            if verification is None:
                adjusted.append(candidate)
                continue
            delta = {"supports": 0.020, "contradicts": -0.020, "uncertain": 0.0}[
                verification.status
            ]
            native_evidence = RetrievalEvidence(
                evidence_type="native_video_verification",
                evidence_ref=verification.evidence_ref,
                matched_text=verification.explanation,
                score=delta,
            )
            evidence = (*candidate.evidence, native_evidence)
            adjusted.append(
                candidate.model_copy(
                    update={
                        "total_score": candidate.total_score + delta,
                        "component_scores": {
                            **candidate.component_scores,
                            "native_video_verification": delta,
                        },
                        "evidence": evidence,
                        "evidence_refs": _evidence_refs(evidence),
                        "evidence_types": _evidence_types(evidence),
                    }
                )
            )
        adjusted.sort(key=lambda item: (-item.total_score, item.video_id, item.scene_id))
        ranked = tuple(
            item.model_copy(update={"rank": rank}) for rank, item in enumerate(adjusted, 1)
        )
        return response.model_copy(update={"method": "adaptive_hybrid", "candidates": ranked})

    def search_with_trace(
        self,
        query: RetrievalQuery,
        memory_ref: RetrievalMemory,
    ) -> AdaptiveRetrievalResult:
        started = time.perf_counter()
        analysis = self.analyzer.analyze(query)
        plan = self.router.plan(
            query,
            analysis,
            strategy=self.strategy,
            evidence_reranking=self.evidence_reranking,
            native_video=self.native_video,
        )
        fusion_result = self.fusion.fuse(query, memory_ref, plan)
        response, rerank_trace = self.reranker.rerank(
            query=query,
            analysis=analysis,
            plan=plan,
            response=fusion_result.response,
        )
        verifications: list[NativeVideoVerification] = []
        if plan.native_video_policy != "disabled" and self._should_verify(
            response.candidates, rerank_trace
        ):
            if self.verifier is None:
                raise ValueError("native-video policy triggered without a verifier")
            for candidate in response.candidates[: plan.native_video_depth]:
                clip = self.scene_clips.get((candidate.video_id, candidate.scene_id))
                if clip is None:
                    raise ValueError(
                        "native-video policy triggered without a candidate scene clip: "
                        f"{candidate.video_id}/{candidate.scene_id}"
                    )
                verifications.append(
                    self.verifier.verify(query=query, candidate=candidate, scene_clip=clip)
                )
        response = self._apply_native_verification(response, tuple(verifications))
        if not verifications:
            response = response.model_copy(update={"method": "adaptive_hybrid"})
        final_rank = {
            (candidate.video_id, candidate.scene_id): candidate.rank
            for candidate in response.candidates
        }
        final_trace = tuple(
            item.model_copy(update={"final_rank": final_rank[(item.video_id, item.scene_id)]})
            for item in rerank_trace
        )
        component_latency: dict[str, float] = {
            channel: item.latency_ms for channel, item in fusion_result.component_responses.items()
        }
        if verifications:
            component_latency["native_video"] = float(
                sum(item.latency_ms for item in verifications)
            )
        total_ms = (time.perf_counter() - started) * 1000
        response = response.model_copy(
            update={
                "candidates": response.candidates[: query.top_k],
                "latency_ms": total_ms,
                "cache_hit": (response.cache_hit and all(item.cache_hit for item in verifications)),
            }
        )
        return AdaptiveRetrievalResult(
            query_analysis=analysis,
            plan=plan,
            response=response,
            rerank_trace=final_trace,
            native_verifications=tuple(verifications),
            component_latency_ms=component_latency,
            total_latency_ms=total_ms,
            config_sha256=config_sha256(
                {
                    "adaptive": self.config.model_dump(mode="json"),
                    "strategy": self.strategy,
                    "evidence_reranking": self.evidence_reranking,
                    "native_video": self.native_video,
                    "index_id": memory_ref.manifest.index_id,
                }
            ),
        )

    def search(self, query: RetrievalQuery, memory_ref: RetrievalMemory) -> RetrievalResponse:
        return self.search_with_trace(query, memory_ref).response
