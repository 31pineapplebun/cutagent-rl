"""Offline/private M1B gold contracts and transparent perception metrics."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import Field

from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import PerceptionResult


class PerceptionCaseGold(SchemaModel):
    """Evaluator-only generated-case annotation; never imported by runtime modules."""

    case_id: Identifier
    source_group_id: Identifier
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: NonEmptyStr
    expected_entities: tuple[NonEmptyStr, ...] = Field(min_length=1)
    expected_action: NonEmptyStr | None = None
    expected_visible_text: NonEmptyStr
    reference_transcript: NonEmptyStr
    reference_speech_range: TimeRange
    expected_event_range: TimeRange | None = None


def normalize_text(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = re.findall(r"[^\W_]+", reference.casefold(), flags=re.UNICODE)
    hypothesis_words = re.findall(r"[^\W_]+", hypothesis.casefold(), flags=re.UNICODE)
    return edit_distance(reference_words, hypothesis_words) / max(1, len(reference_words))


def character_error_rate(reference: str, hypothesis: str) -> float:
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    return edit_distance(normalized_reference, normalized_hypothesis) / max(
        1, len(normalized_reference)
    )


def temporal_iou(left: TimeRange, right: TimeRange) -> float:
    intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    return intersection / union if union else 0.0


def _entity_matches(expected: str, predicted: str) -> bool:
    expected_tokens = set(re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", expected.casefold()))
    predicted_tokens = set(re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", predicted.casefold()))
    return bool(expected_tokens) and expected_tokens.issubset(predicted_tokens)


def _entity_is_supported(expected_entities: tuple[str, ...], predicted: str) -> bool:
    return any(
        _entity_matches(expected, predicted) or _entity_matches(predicted, expected)
        for expected in expected_entities
    )


def _action_matches(expected: str, subject: str, action: str, object_value: str | None) -> bool:
    combined = " ".join(value for value in (subject, action, object_value) if value).casefold()
    if expected == "moves_right":
        return ("move" in combined and "right" in combined) or "向右" in combined
    return normalize_text(expected) in normalize_text(combined)


def _ocr_evidence_is_temporally_consistent(result: PerceptionResult, expected: str) -> bool:
    matching = [span for span in result.ocr_spans if normalize_text(span.exact_text) == expected]
    for span in matching:
        for reference in span.evidence_refs:
            if reference.segment_id is not None and reference.segment_id != span.segment_id:
                continue
            if span.observed_ms is not None and reference.observed_ms == span.observed_ms:
                return True
            if (
                span.time_range is not None
                and reference.observed_ms is not None
                and span.time_range.start_ms <= reference.observed_ms < span.time_range.end_ms
            ):
                return True
            if (
                span.time_range is not None
                and reference.time_range is not None
                and reference.time_range.start_ms <= span.time_range.start_ms
                and reference.time_range.end_ms >= span.time_range.end_ms
            ):
                return True
    return False


def evaluate_visual_result(result: PerceptionResult, gold: PerceptionCaseGold) -> dict[str, object]:
    entities = [
        entity.label
        for observation in result.visual_observations
        for entity in observation.entities
    ]
    actions = [
        action for observation in result.visual_observations for action in observation.actions
    ]
    visible_text = [span.exact_text for span in result.ocr_spans]
    events = list(result.temporal_events)
    primary_entity = gold.expected_entities[0]
    entity_hits = [label for label in entities if _entity_matches(primary_entity, label)]
    hallucinated_entities = [
        label for label in entities if not _entity_is_supported(gold.expected_entities, label)
    ]
    if gold.expected_action is None:
        action_correct = not actions
    else:
        action_correct = any(
            _action_matches(gold.expected_action, action.subject, action.action, action.object)
            for action in actions
        )
    expected_text = normalize_text(gold.expected_visible_text)
    ocr_exact = any(text == gold.expected_visible_text for text in visible_text)
    ocr_normalized = any(normalize_text(text) == expected_text for text in visible_text)
    ocr_cer = min(
        (character_error_rate(gold.expected_visible_text, text) for text in visible_text),
        default=1.0,
    )
    if gold.expected_event_range is None:
        event_correct = not events
    else:
        event_correct = any(
            temporal_iou(event.time_range, gold.expected_event_range) >= 0.5 for event in events
        )
    catalog_ids = {item.artifact_id for item in result.world_state.evidence_catalog}
    all_refs = [
        reference
        for observation in result.visual_observations
        for group in (
            observation.summary_evidence_refs,
            *(entity.evidence_refs for entity in observation.entities),
            *(action.evidence_refs for action in observation.actions),
            *(text.evidence_refs for text in observation.directly_visible_text),
            *(event.evidence_refs for event in observation.temporal_events),
        )
        for reference in group
    ]
    visual_performance = [item for item in result.performance if item.operation.startswith("qwen_")]
    return {
        "entity_correct": bool(entity_hits),
        "predicted_entities": entities,
        "hallucinated_entities": hallucinated_entities,
        "action_correct": action_correct,
        "predicted_actions": [action.model_dump(mode="json") for action in actions],
        "visible_text_exact": ocr_exact,
        "visible_text_normalized": ocr_normalized,
        "visible_text_cer": ocr_cer,
        "predicted_visible_text": visible_text,
        "ocr_evidence_correct": _ocr_evidence_is_temporally_consistent(result, expected_text),
        "temporal_event_correct": event_correct,
        "predicted_events": [event.model_dump(mode="json") for event in events],
        "evidence_integrity": all(ref.artifact_id in catalog_ids for ref in all_refs),
        "repair_count": sum(item.repair_count for item in visual_performance),
        "latency_ms": sum(item.latency_ms for item in visual_performance),
        "peak_allocated_bytes": max(
            (item.peak_allocated_bytes or 0 for item in visual_performance), default=0
        ),
        "peak_reserved_bytes": max(
            (item.peak_reserved_bytes or 0 for item in visual_performance), default=0
        ),
        "frames": sum(item.frames for item in visual_performance),
        "input_tokens": sum(item.input_tokens or 0 for item in visual_performance),
    }


def evaluate_asr_result(result: PerceptionResult, gold: PerceptionCaseGold) -> dict[str, object]:
    hypothesis = " ".join(span.text for span in result.transcript_spans)
    if result.transcript_spans:
        observed = TimeRange(
            start_ms=result.transcript_spans[0].time_range.start_ms,
            end_ms=result.transcript_spans[-1].time_range.end_ms,
        )
        alignment_error_ms: float | None = (
            abs(observed.start_ms - gold.reference_speech_range.start_ms)
            + abs(observed.end_ms - gold.reference_speech_range.end_ms)
        ) / 2
    else:
        alignment_error_ms = None
    return {
        "reference": gold.reference_transcript,
        "hypothesis": hypothesis,
        "wer": word_error_rate(gold.reference_transcript, hypothesis),
        "cer": character_error_rate(gold.reference_transcript, hypothesis),
        "timestamp_alignment_error_ms": alignment_error_ms,
    }
