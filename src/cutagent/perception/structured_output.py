"""Strict Qwen structured-output contract and bounded repair controller."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import Field, ValidationError, model_validator

from cutagent.core.errors import StructuredOutputError
from cutagent.schemas.base import NonEmptyStr, SchemaModel
from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import (
    DirectlyVisibleText,
    EvidenceRef,
    PerceptionMode,
    TemporalEvent,
    VisualAction,
    VisualEntity,
    VisualObservation,
)


class EvidenceClaim(SchemaModel):
    evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)


class SummaryClaim(EvidenceClaim):
    text: NonEmptyStr


class EntityClaim(EvidenceClaim):
    label: NonEmptyStr
    attributes: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)


class ActionClaim(EvidenceClaim):
    subject: NonEmptyStr
    action: NonEmptyStr
    object: NonEmptyStr | None = None


class VisibleTextClaim(EvidenceClaim):
    exact_text: NonEmptyStr
    normalized_text: NonEmptyStr | None = None


class EventClaim(EvidenceClaim):
    description: NonEmptyStr
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    uncertainty: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_order(self) -> EventClaim:
        if self.start_ms >= self.end_ms:
            raise ValueError("event must satisfy start_ms < end_ms")
        return self


class StructuredVisualOutput(SchemaModel):
    scene_summary: SummaryClaim | None
    entities: tuple[EntityClaim, ...]
    actions: tuple[ActionClaim, ...]
    directly_visible_text: tuple[VisibleTextClaim, ...]
    inferred_semantic_text: tuple[NonEmptyStr, ...]
    temporal_events: tuple[EventClaim, ...]
    uncertainties: tuple[NonEmptyStr, ...]


def parse_json_object(raw: str) -> dict[str, Any]:
    stripped = raw.strip()
    if stripped.startswith("```"):
        raise StructuredOutputError("markdown fences are not valid structured output")
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise StructuredOutputError(f"invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise StructuredOutputError("structured output must be a JSON object")
    return value


def parse_with_bounded_repair(
    initial_output: str,
    *,
    maximum_repairs: int,
    repair: Callable[[str, str], str],
) -> tuple[StructuredVisualOutput, tuple[str, ...]]:
    attempts = [initial_output]
    current = initial_output
    for repair_index in range(maximum_repairs + 1):
        try:
            return StructuredVisualOutput.model_validate(parse_json_object(current)), tuple(
                attempts
            )
        except (StructuredOutputError, ValidationError) as error:
            if repair_index >= maximum_repairs:
                raise StructuredOutputError(
                    f"structured output invalid after {len(attempts)} attempt(s): {error}",
                    attempts=tuple(attempts),
                ) from error
            current = repair(current, str(error))
            attempts.append(current)
    raise AssertionError("bounded repair loop must return or raise")


def _resolve(
    aliases: tuple[str, ...], available: Mapping[str, EvidenceRef]
) -> tuple[EvidenceRef, ...]:
    resolved: list[EvidenceRef] = []
    for alias in aliases:
        evidence = available.get(alias)
        if evidence is None:
            raise StructuredOutputError(f"unknown evidence alias: {alias}")
        if evidence not in resolved:
            resolved.append(evidence)
    if not resolved:
        raise StructuredOutputError("claim has no supported evidence")
    return tuple(resolved)


def to_visual_observation(
    output: StructuredVisualOutput,
    *,
    observation_id: str,
    segment_id: str,
    mode: PerceptionMode,
    scene_range: TimeRange,
    available_evidence: Mapping[str, EvidenceRef],
    repair_count: int,
) -> VisualObservation:
    summary = output.scene_summary
    events: list[TemporalEvent] = []
    for index, claim in enumerate(output.temporal_events):
        event_range = TimeRange(start_ms=claim.start_ms, end_ms=claim.end_ms)
        if event_range.start_ms < scene_range.start_ms or event_range.end_ms > scene_range.end_ms:
            raise StructuredOutputError("model temporal event lies outside the scene")
        events.append(
            TemporalEvent(
                event_id=f"event-{observation_id}-{index:03d}",
                description=claim.description,
                time_range=event_range,
                evidence_refs=_resolve(claim.evidence_ids, available_evidence),
                confidence=None,
                uncertainty=claim.uncertainty,
            )
        )
    return VisualObservation(
        observation_id=observation_id,
        segment_id=segment_id,
        mode=mode,
        time_range=scene_range,
        scene_summary=None if summary is None else summary.text,
        summary_evidence_refs=(
            () if summary is None else _resolve(summary.evidence_ids, available_evidence)
        ),
        entities=tuple(
            VisualEntity(
                label=claim.label,
                attributes=claim.attributes,
                evidence_refs=_resolve(claim.evidence_ids, available_evidence),
            )
            for claim in output.entities
        ),
        actions=tuple(
            VisualAction(
                subject=claim.subject,
                action=claim.action,
                object=claim.object,
                evidence_refs=_resolve(claim.evidence_ids, available_evidence),
            )
            for claim in output.actions
        ),
        directly_visible_text=tuple(
            DirectlyVisibleText(
                exact_text=claim.exact_text,
                normalized_text=claim.normalized_text,
                evidence_refs=_resolve(claim.evidence_ids, available_evidence),
            )
            for claim in output.directly_visible_text
        ),
        inferred_semantic_text=output.inferred_semantic_text,
        temporal_events=tuple(events),
        uncertainties=output.uncertainties,
        repair_count=repair_count,
    )
