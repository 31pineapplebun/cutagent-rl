"""Query-aware weighted reciprocal-rank fusion for M2B."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

from cutagent.retrieval.protocols import MultimodalRetriever
from cutagent.retrieval.store import RetrievalMemory
from cutagent.schemas.perception import EvidenceRef
from cutagent.schemas.retrieval import (
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalEvidence,
    RetrievalEvidenceType,
    RetrievalPlan,
    RetrievalQuery,
    RetrievalResponse,
)


@dataclass(frozen=True, slots=True)
class WeightedFusionResult:
    response: RetrievalResponse
    component_responses: Mapping[RetrievalChannel, RetrievalResponse]


def _unique_evidence(
    candidates: list[RetrievalCandidate],
) -> tuple[RetrievalEvidence, ...]:
    output: list[RetrievalEvidence] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in candidates:
        for item in candidate.evidence:
            key = (
                item.evidence_type,
                item.evidence_ref,
                item.document_field,
                item.matched_text,
            )
            if key not in seen:
                seen.add(key)
                output.append(item)
    return tuple(output)


def _references(evidence: tuple[RetrievalEvidence, ...]) -> tuple[EvidenceRef, ...]:
    return tuple(dict.fromkeys(item.evidence_ref for item in evidence))


def _types(evidence: tuple[RetrievalEvidence, ...]) -> tuple[RetrievalEvidenceType, ...]:
    order: tuple[RetrievalEvidenceType, ...] = (
        "transcript",
        "ocr",
        "structured_text",
        "keyframe",
        "native_video_verification",
    )
    present = {item.evidence_type for item in evidence}
    return tuple(item for item in order if item in present)


class WeightedRankFusion:
    """Fuse ranks only; incompatible BM25/BGE/SigLIP raw scores remain separate."""

    def __init__(
        self,
        components: Mapping[RetrievalChannel, MultimodalRetriever],
        *,
        rank_constant: int = 60,
    ) -> None:
        if rank_constant < 1:
            raise ValueError("rank constant must be positive")
        self.components = dict(components)
        self.rank_constant = rank_constant

    def fuse(
        self,
        query: RetrievalQuery,
        memory_ref: RetrievalMemory,
        plan: RetrievalPlan,
    ) -> WeightedFusionResult:
        started = time.perf_counter()
        missing = set(plan.enabled_channels) - set(self.components)
        if missing:
            raise ValueError(f"retrieval plan requests unavailable channels: {sorted(missing)}")
        expanded = query.model_copy(update={"top_k": plan.candidate_depth})
        responses: dict[RetrievalChannel, RetrievalResponse] = {
            channel: self.components[channel].search(expanded, memory_ref)
            for channel in plan.enabled_channels
        }
        scores: dict[tuple[str, str], float] = {}
        by_identity: dict[tuple[str, str], list[RetrievalCandidate]] = {}
        ranks: dict[tuple[str, str], dict[RetrievalChannel, int]] = {}
        raw_scores: dict[tuple[str, str], dict[RetrievalChannel, float]] = {}
        for channel, response in responses.items():
            weight = plan.channel_weights[channel]
            for candidate in response.candidates:
                identity = (candidate.video_id, candidate.scene_id)
                scores[identity] = scores.get(identity, 0.0) + weight / (
                    self.rank_constant + candidate.rank
                )
                by_identity.setdefault(identity, []).append(candidate)
                ranks.setdefault(identity, {})[channel] = candidate.rank
                raw_scores.setdefault(identity, {})[channel] = candidate.total_score
        ranked = sorted(scores, key=lambda item: (-scores[item], item[0], item[1]))
        output: list[RetrievalCandidate] = []
        for identity in ranked[: plan.candidate_depth]:
            members = by_identity[identity]
            evidence = _unique_evidence(members)
            components: dict[str, float] = {}
            for channel in plan.enabled_channels:
                if channel not in ranks[identity]:
                    continue
                components[f"raw.{channel}"] = raw_scores[identity][channel]
                components[f"rank.{channel}"] = float(ranks[identity][channel])
                components[f"weighted_rrf.{channel}"] = plan.channel_weights[channel] / (
                    self.rank_constant + ranks[identity][channel]
                )
            first = members[0]
            output.append(
                RetrievalCandidate(
                    video_id=first.video_id,
                    scene_id=first.scene_id,
                    time_range=first.time_range,
                    total_score=scores[identity],
                    component_scores=components,
                    evidence_refs=_references(evidence),
                    evidence_types=_types(evidence),
                    evidence=evidence,
                    rank=len(output) + 1,
                )
            )
        response = RetrievalResponse(
            query_id=query.query_id,
            method="fusion",
            candidates=tuple(output),
            applied_filters=next(iter(responses.values())).applied_filters,
            latency_ms=(time.perf_counter() - started) * 1000,
            cache_hit=all(item.cache_hit for item in responses.values()),
        )
        return WeightedFusionResult(response=response, component_responses=responses)
