"""Evaluator-only M1B.5 motion gold contracts and diagnostic metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import Field, JsonValue

from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import PerceptionResult

MotionTaskType = Literal[
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "stationary",
    "appear",
    "disappear",
    "enter_frame",
    "exit_frame",
    "approach",
    "move_away",
    "a_before_b",
    "b_before_a",
    "stop_then_start",
    "start_then_stop",
    "short_motion",
    "long_motion",
]
MotionActionCode = Literal[
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "appear",
    "disappear",
    "enter_frame",
    "exit_frame",
    "approach",
    "move_away",
    "start_moving",
    "stop_moving",
]
MotionDirection = Literal["left", "right", "up", "down", "toward", "away"]


class MotionEntityGold(SchemaModel):
    label: NonEmptyStr
    aliases: tuple[NonEmptyStr, ...] = ()


class MotionEventGold(SchemaModel):
    actor: NonEmptyStr
    actor_aliases: tuple[NonEmptyStr, ...] = ()
    action: MotionActionCode
    time_range: TimeRange


class MotionCaseGold(SchemaModel):
    """Private deterministic annotation for one generated diagnostic video."""

    case_id: Identifier
    source_group_id: Identifier
    task_type: MotionTaskType
    seed: int = Field(ge=0)
    duration_ms: int = Field(gt=0)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_entities: tuple[MotionEntityGold, ...] = Field(min_length=1)
    supported_entity_terms: tuple[NonEmptyStr, ...] = ()
    expected_events: tuple[MotionEventGold, ...] = ()
    expected_direction: MotionDirection | None = None
    expected_order: tuple[NonEmptyStr, NonEmptyStr] | None = None
    generation_config: dict[NonEmptyStr, JsonValue]


_ACTION_PATTERNS: dict[str, tuple[tuple[str, ...], ...]] = {
    "move_left": (("move", "left"), ("moving", "left"), ("leftward",), ("向左",)),
    "move_right": (("move", "right"), ("moving", "right"), ("rightward",), ("向右",)),
    "move_up": (("move", "up"), ("moving", "up"), ("upward",), ("向上",)),
    "move_down": (("move", "down"), ("moving", "down"), ("downward",), ("向下",)),
    "appear": (("appear",), ("becomes visible",), ("出现",)),
    "disappear": (("disappear",), ("vanish",), ("no longer visible",), ("消失",)),
    "enter_frame": (("enter", "frame"), ("comes into", "view"), ("进入", "画面")),
    "exit_frame": (("exit", "frame"), ("leave", "frame"), ("leaves", "view"), ("离开", "画面")),
    "approach": (("approach",), ("closer",), ("toward", "camera"), ("接近",)),
    "move_away": (("move", "away"), ("farther",), ("recede",), ("远离",)),
    "start_moving": (("start", "mov"), ("begin", "mov"), ("resume", "mov"), ("开始", "移动")),
    "stop_moving": (("stop", "mov"), ("becomes stationary",), ("halts",), ("停止", "移动")),
}


def _normalized(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


def _matches_term(value: str, term: str) -> bool:
    normalized_value = _normalized(value)
    normalized_term = _normalized(term)
    return normalized_term in normalized_value or normalized_value in normalized_term


def _matches_any_term(value: str, terms: Iterable[str]) -> bool:
    return any(_matches_term(value, term) for term in terms)


def _matches_action(value: str, action: str) -> bool:
    normalized = _normalized(value)
    return any(all(term in normalized for term in pattern) for pattern in _ACTION_PATTERNS[action])


def _event_boundary_error_ms(expected: TimeRange, predicted: TimeRange) -> float:
    return (
        abs(expected.start_ms - predicted.start_ms) + abs(expected.end_ms - predicted.end_ms)
    ) / 2


def _temporal_iou(expected: TimeRange, predicted: TimeRange) -> float:
    intersection = max(
        0,
        min(expected.end_ms, predicted.end_ms) - max(expected.start_ms, predicted.start_ms),
    )
    union = max(expected.end_ms, predicted.end_ms) - min(expected.start_ms, predicted.start_ms)
    return intersection / union if union else 0.0


def evaluate_motion_result(result: PerceptionResult, gold: MotionCaseGold) -> dict[str, object]:
    """Score only deterministic facts; return raw predictions beside every score."""

    entities = [
        entity.label
        for observation in result.visual_observations
        for entity in observation.entities
    ]
    actions = [
        action for observation in result.visual_observations for action in observation.actions
    ]
    events = list(result.temporal_events)
    action_claims = [
        " ".join(value for value in (action.subject, action.action, action.object) if value)
        for action in actions
    ]
    event_claims = [event.description for event in events]
    all_claims = action_claims + event_claims

    entity_scores = []
    for expected_entity in gold.expected_entities:
        aliases = (expected_entity.label, *expected_entity.aliases)
        entity_scores.append(any(_matches_any_term(predicted, aliases) for predicted in entities))
    supported_terms = (
        *gold.supported_entity_terms,
        *(
            term
            for expected_entity in gold.expected_entities
            for term in (expected_entity.label, *expected_entity.aliases)
        ),
    )
    hallucinated_entities = [
        predicted for predicted in entities if not _matches_any_term(predicted, supported_terms)
    ]

    def claim_matches_expected(claim: str, expected: MotionEventGold) -> bool:
        actors = (expected.actor, *expected.actor_aliases)
        return _matches_any_term(claim, actors) and _matches_action(claim, expected.action)

    expected_claim_hits = [
        any(claim_matches_expected(claim, expected) for claim in all_claims)
        for expected in gold.expected_events
    ]
    if gold.expected_events:
        action_correct = all(expected_claim_hits)
    else:
        action_correct = not any(
            _matches_any_term(claim, (entity.label, *entity.aliases))
            and any(_matches_action(claim, code) for code in _ACTION_PATTERNS)
            for claim in all_claims
            for entity in gold.expected_entities
        )

    localization_errors: list[float] = []
    temporal_hits: list[bool] = []
    for expected_event in gold.expected_events:
        matching_events = [
            event for event in events if claim_matches_expected(event.description, expected_event)
        ]
        if matching_events:
            localization_errors.append(
                min(
                    _event_boundary_error_ms(expected_event.time_range, event.time_range)
                    for event in matching_events
                )
            )
        temporal_hits.append(
            any(
                _temporal_iou(expected_event.time_range, event.time_range) >= 0.3
                for event in matching_events
            )
        )
    temporal_event_correct = all(temporal_hits) if gold.expected_events else not events

    if gold.expected_direction is None:
        direction_correct: bool | None = None
    else:
        direction_code = {
            "left": "move_left",
            "right": "move_right",
            "up": "move_up",
            "down": "move_down",
            "toward": "approach",
            "away": "move_away",
        }[gold.expected_direction]
        direction_correct = any(_matches_action(claim, direction_code) for claim in all_claims)

    if gold.expected_order is None:
        event_order_correct: bool | None = None
    else:
        first_actor, second_actor = gold.expected_order
        first_starts = [
            event.time_range.start_ms
            for event in events
            if _matches_term(event.description, first_actor)
        ]
        second_starts = [
            event.time_range.start_ms
            for event in events
            if _matches_term(event.description, second_actor)
        ]
        event_order_correct = bool(
            first_starts and second_starts and min(first_starts) < min(second_starts)
        )

    unsupported_claims = [
        claim
        for claim in all_claims
        if not any(claim_matches_expected(claim, expected) for expected in gold.expected_events)
    ]
    predicted_claim_count = len(entities) + len(all_claims)
    unsupported_count = len(hallucinated_entities) + len(unsupported_claims)
    visual_performance = [item for item in result.performance if item.operation.startswith("qwen_")]
    return {
        "task_type": gold.task_type,
        "entity_correctness": sum(entity_scores) / len(entity_scores),
        "action_correct": action_correct,
        "temporal_event_correct": temporal_event_correct,
        "direction_correct": direction_correct,
        "event_order_correct": event_order_correct,
        "temporal_localization_error_ms": (
            sum(localization_errors) / len(localization_errors) if localization_errors else None
        ),
        "unsupported_claim_rate": unsupported_count / max(1, predicted_claim_count),
        "hallucinated_entities": hallucinated_entities,
        "unsupported_claims": unsupported_claims,
        "predicted_entities": entities,
        "predicted_actions": [action.model_dump(mode="json") for action in actions],
        "predicted_events": [event.model_dump(mode="json") for event in events],
        "evidence_integrity": True,
        "latency_ms": sum(item.latency_ms for item in visual_performance),
        "peak_allocated_bytes": max(
            (item.peak_allocated_bytes or 0 for item in visual_performance), default=0
        ),
        "peak_reserved_bytes": max(
            (item.peak_reserved_bytes or 0 for item in visual_performance), default=0
        ),
        "frames": sum(item.frames for item in visual_performance),
        "input_tokens": sum(item.input_tokens or 0 for item in visual_performance),
        "repair_count": sum(item.repair_count for item in visual_performance),
    }
