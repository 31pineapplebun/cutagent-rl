"""Deterministic evidence-aware reranking without learned parameters."""

from __future__ import annotations

from collections.abc import Iterable

from cutagent.schemas.retrieval import (
    CandidateRerankTrace,
    QueryAnalysis,
    RetrievalCandidate,
    RetrievalPlan,
    RetrievalQuery,
    RetrievalResponse,
)

from .query import _ACTION_GROUPS


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _evidence_text(candidate: RetrievalCandidate, evidence_type: str | None = None) -> str:
    return "\n".join(
        item.matched_text or ""
        for item in candidate.evidence
        if evidence_type is None or item.evidence_type == evidence_type
    ).casefold()


def _contains_action(text: str, action_code: str) -> bool:
    return any(term in text for term in _ACTION_GROUPS[action_code])


def _quoted_match(phrases: Iterable[str], evidence_text: str) -> bool:
    return any(_normalize(phrase) in evidence_text for phrase in phrases)


class EvidenceAwareReranker:
    """General rules over observable evidence; never query IDs or private types."""

    version = "m2b-evidence-reranker-v1"

    def rerank(
        self,
        *,
        query: RetrievalQuery,
        analysis: QueryAnalysis,
        plan: RetrievalPlan,
        response: RetrievalResponse,
    ) -> tuple[RetrievalResponse, tuple[CandidateRerankTrace, ...]]:
        if plan.reranking_policy == "none" or not response.candidates:
            zero_traces = tuple(
                CandidateRerankTrace(
                    video_id=item.video_id,
                    scene_id=item.scene_id,
                    original_rank=item.rank,
                    final_rank=item.rank,
                    adjustment=0,
                )
                for item in response.candidates
            )
            return response, zero_traces

        scored: list[tuple[RetrievalCandidate, float, tuple[str, ...], tuple[str, ...]]] = []
        signature_counts: dict[str, int] = {}
        for candidate in response.candidates[: plan.rerank_depth]:
            adjustment = 0.0
            matched: list[str] = []
            penalties: list[str] = []
            all_text = _evidence_text(candidate)
            structured = _evidence_text(candidate, "structured_text")
            transcript = _evidence_text(candidate, "transcript")
            ocr = _evidence_text(candidate, "ocr")

            if analysis.primary_intent == "visible_text" and analysis.quoted_phrases:
                if _quoted_match(analysis.quoted_phrases, ocr):
                    adjustment += 0.025
                    matched.append("exact_ocr_phrase")
                else:
                    adjustment -= 0.008
                    penalties.append("missing_exact_ocr_phrase")
            if analysis.primary_intent == "speech_or_quote" and analysis.quoted_phrases:
                if _quoted_match(analysis.quoted_phrases, transcript):
                    adjustment += 0.020
                    matched.append("exact_transcript_phrase")
                else:
                    adjustment -= 0.006
                    penalties.append("missing_exact_transcript_phrase")

            if analysis.primary_intent == "action_or_motion":
                action_hits = tuple(
                    code for code in analysis.action_terms if _contains_action(structured, code)
                )
                other_actions = tuple(
                    code
                    for code in _ACTION_GROUPS
                    if code not in analysis.action_terms and _contains_action(structured, code)
                )
                if action_hits:
                    adjustment += 0.014 + 0.002 * (len(action_hits) - 1)
                    matched.append("requested_action=" + ",".join(action_hits))
                else:
                    adjustment -= 0.010
                    penalties.append("missing_requested_action_evidence")
                    if other_actions:
                        adjustment -= 0.004
                        penalties.append("conflicting_action=" + ",".join(other_actions))
                if candidate.evidence_types == ("keyframe",):
                    adjustment -= 0.004
                    penalties.append("visual_only_for_action_query")

            entity_hits = tuple(term for term in analysis.entity_terms if term in all_text)
            if entity_hits:
                adjustment += min(0.006, 0.002 * len(entity_hits))
                matched.append("entity=" + ",".join(entity_hits))
            query_terms = tuple(
                token
                for token in _normalize(query.text).replace('"', " ").split()
                if len(token) > 2
            )
            if query_terms and not any(term in all_text for term in query_terms):
                adjustment -= 0.003
                penalties.append("no_textual_query_support")

            signature = _normalize(all_text)
            duplicate_index = signature_counts.get(signature, 0) if signature else 0
            if signature:
                signature_counts[signature] = duplicate_index + 1
            if duplicate_index:
                adjustment -= min(0.004, duplicate_index * 0.001)
                penalties.append("duplicate_evidence_signature")
            scored.append((candidate, adjustment, tuple(matched), tuple(penalties)))

        scored.sort(
            key=lambda item: (
                -(item[0].total_score + item[1]),
                item[0].video_id,
                item[0].scene_id,
            )
        )
        tail = list(response.candidates[plan.rerank_depth :])
        ordered = [item[0] for item in scored] + tail
        detail = {
            (item.video_id, item.scene_id): (adjustment, matched, penalties)
            for item, adjustment, matched, penalties in scored
        }
        output: list[RetrievalCandidate] = []
        trace_output: list[CandidateRerankTrace] = []
        for rank, candidate in enumerate(ordered, start=1):
            adjustment, final_features, final_penalties = detail.get(
                (candidate.video_id, candidate.scene_id), (0.0, (), ())
            )
            updated = candidate.model_copy(
                update={
                    "rank": rank,
                    "total_score": candidate.total_score + adjustment,
                    "component_scores": {
                        **candidate.component_scores,
                        "evidence_rerank_adjustment": adjustment,
                    },
                }
            )
            output.append(updated)
            trace_output.append(
                CandidateRerankTrace(
                    video_id=candidate.video_id,
                    scene_id=candidate.scene_id,
                    original_rank=candidate.rank,
                    final_rank=rank,
                    adjustment=adjustment,
                    matched_features=final_features,
                    penalties=final_penalties,
                )
            )
        return response.model_copy(update={"candidates": tuple(output)}), tuple(trace_output)
