"""Frozen 150-query M2A development diagnostic assembled from M1B evidence."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from cutagent.schemas.perception import PerceptionResult
from cutagent.schemas.retrieval import RetrievalEvidenceType, RetrievalQuery
from cutagent_evaluation.m1b5_motion import MotionCaseGold
from cutagent_evaluation.m1b_metrics import PerceptionCaseGold, normalize_text
from cutagent_evaluation.retrieval import (
    FrozenRetrievalCase,
    RelevantSceneGold,
    RetrievalCaseGold,
    RetrievalQueryType,
)

_ACTION_EN = {
    "a_before_b": "object A moves before object B",
    "appear": "appears",
    "approach": "approaches the camera",
    "b_before_a": "object B moves before object A",
    "disappear": "disappears",
    "enter_frame": "enters the frame",
    "exit_frame": "exits the frame",
    "long_motion": "moves for a long interval",
    "move_away": "moves away from the camera",
    "move_down": "moves downward",
    "move_left": "moves left",
    "move_right": "moves right",
    "move_up": "moves upward",
    "short_motion": "moves for a short interval",
    "start_then_stop": "starts moving and then stops",
    "stationary": "remains stationary",
    "stop_then_start": "stops and then starts moving",
}
_ACTION_ZH = {
    "a_before_b": "物体A先于物体B移动",
    "appear": "出现",
    "approach": "靠近镜头",
    "b_before_a": "物体B先于物体A移动",
    "disappear": "消失",
    "enter_frame": "进入画面",
    "exit_frame": "离开画面",
    "long_motion": "长时间移动",
    "move_away": "远离镜头",
    "move_down": "向下移动",
    "move_left": "向左移动",
    "move_right": "向右移动",
    "move_up": "向上移动",
    "short_motion": "短时间移动",
    "start_then_stop": "开始移动后停止",
    "stationary": "保持静止",
    "stop_then_start": "停止后重新移动",
}
_ENTITY_ZH = {
    "red square": "红色方块",
    "green circle": "绿色圆形",
    "blue square": "蓝色方块",
    "yellow circle": "黄色圆形",
    "purple square": "紫色方块",
    "orange circle": "橙色圆形",
    "cyan square": "青色方块",
    "magenta circle": "品红色圆形",
}


def load_perception_results(paths: Iterable[Path]) -> dict[str, PerceptionResult]:
    results: dict[str, PerceptionResult] = {}
    for path in sorted(paths):
        result = PerceptionResult.model_validate_json(path.read_text(encoding="utf-8"))
        case_id = path.stem
        if case_id in results:
            raise ValueError(f"duplicate perception case ID: {case_id}")
        results[case_id] = result
    return results


def load_m1b_gold(path: Path) -> tuple[PerceptionCaseGold, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("M1B private gold must be a JSON list")
    return tuple(PerceptionCaseGold.model_validate(item) for item in payload)


def load_motion_gold(path: Path) -> tuple[MotionCaseGold, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("motion private gold must be a JSON list")
    return tuple(MotionCaseGold.model_validate(item) for item in payload)


def _relevant(
    case_ids: Iterable[str],
    results: dict[str, PerceptionResult],
    evidence_types: tuple[RetrievalEvidenceType, ...],
) -> tuple[RelevantSceneGold, ...]:
    output: list[RelevantSceneGold] = []
    for case_id in sorted(set(case_ids)):
        result = results.get(case_id)
        if result is None:
            continue
        for scene in result.world_state.scenes:
            output.append(
                RelevantSceneGold(
                    video_id=result.world_state.video_id,
                    scene_id=scene.segment_id,
                    acceptable_time_range=scene.time_range,
                    expected_evidence_types=evidence_types,
                )
            )
    if not output:
        raise ValueError("query has no relevant scene in the indexed corpus")
    return tuple(output)


def _has_evidence(
    relevant: tuple[RelevantSceneGold, ...],
    results: dict[str, PerceptionResult],
    query_type: RetrievalQueryType,
) -> bool:
    relevant_ids = {(item.video_id, item.scene_id) for item in relevant}
    for result in results.values():
        for scene in result.world_state.scenes:
            if (result.world_state.video_id, scene.segment_id) not in relevant_ids:
                continue
            if query_type == "speech" and scene.transcript_spans:
                return True
            if query_type == "ocr" and scene.ocr_spans:
                return True
            if query_type in {"entity", "semantic"} and (scene.entities or scene.scene_summary):
                return True
            if query_type in {"action", "hard_negative"} and (
                scene.actions or scene.temporal_events
            ):
                return True
    return False


def _case(
    *,
    query_id: str,
    text: str,
    query_type: RetrievalQueryType,
    relevant_ids: Iterable[str],
    results: dict[str, PerceptionResult],
    evidence_types: tuple[RetrievalEvidenceType, ...],
    hard_negative: bool = False,
) -> FrozenRetrievalCase:
    relevant = _relevant(relevant_ids, results, evidence_types)
    available = _has_evidence(relevant, results, query_type)
    upstream: Literal["ASR_failure", "OCR_failure", "perception_failure"]
    if query_type == "speech":
        upstream = "ASR_failure"
    elif query_type == "ocr":
        upstream = "OCR_failure"
    else:
        upstream = "perception_failure"
    return FrozenRetrievalCase(
        query=RetrievalQuery(query_id=query_id, text=text, top_k=10),
        gold=RetrievalCaseGold(
            query_id=query_id,
            source_group_id="m2a-generated-development-v1",
            query_type=query_type,
            relevant_scenes=relevant,
            hard_negative=hard_negative,
            required_evidence_available=available,
            upstream_failure_type=None if available else upstream,
        ),
    )


def build_frozen_development_cases(
    *,
    m1b_gold: tuple[PerceptionCaseGold, ...],
    motion_gold: tuple[MotionCaseGold, ...],
    results: dict[str, PerceptionResult],
) -> tuple[FrozenRetrievalCase, ...]:
    """Create exactly 150 deterministic, bilingual queries over fixed real outputs."""

    cases: list[FrozenRetrievalCase] = []
    m1b_by_id = {item.case_id: item for item in m1b_gold if item.case_id in results}
    motion_by_id = {item.case_id: item for item in motion_gold if item.case_id in results}

    transcript_groups: dict[str, list[str]] = {}
    for gold in m1b_by_id.values():
        transcript_groups.setdefault(normalize_text(gold.reference_transcript), []).append(
            gold.case_id
        )
    for index, (_normalized, relevant_ids) in enumerate(sorted(transcript_groups.items())):
        reference = m1b_by_id[relevant_ids[0]].reference_transcript
        cases.append(
            _case(
                query_id=f"speech-en-{index:02d}",
                text=f'Find the scene where the narration says "{reference}"',
                query_type="speech",
                relevant_ids=relevant_ids,
                results=results,
                evidence_types=("transcript",),
            )
        )
        cases.append(
            _case(
                query_id=f"speech-zh-{index:02d}",
                text=f"找到旁白内容为“{reference}”的场景",
                query_type="speech",
                relevant_ids=relevant_ids,
                results=results,
                evidence_types=("transcript",),
            )
        )

    for index, gold in enumerate(sorted(m1b_by_id.values(), key=lambda item: item.case_id)):
        cases.append(
            _case(
                query_id=f"ocr-{index:02d}",
                text=f'Find the scene showing the exact text "{gold.expected_visible_text}"',
                query_type="ocr",
                relevant_ids=(gold.case_id,),
                results=results,
                evidence_types=("ocr",),
            )
        )

    entity_groups: dict[str, list[str]] = {}
    for gold in m1b_by_id.values():
        entity_groups.setdefault(gold.expected_entities[0], []).append(gold.case_id)
    for motion_item in motion_by_id.values():
        for entity in motion_item.expected_entities:
            entity_groups.setdefault(entity.label, []).append(motion_item.case_id)
    for index, (label, relevant_ids) in enumerate(sorted(entity_groups.items())[:8]):
        cases.append(
            _case(
                query_id=f"entity-en-{index:02d}",
                text=f"Find a scene containing a {label}",
                query_type="entity",
                relevant_ids=relevant_ids,
                results=results,
                evidence_types=("structured_text", "keyframe"),
            )
        )
        cases.append(
            _case(
                query_id=f"entity-zh-{index:02d}",
                text=f"找到包含{_ENTITY_ZH.get(label, label)}的场景",
                query_type="entity",
                relevant_ids=relevant_ids,
                results=results,
                evidence_types=("structured_text", "keyframe"),
            )
        )

    semantic_seeds = sorted(motion_by_id.values(), key=lambda item: item.case_id)[:15]
    for index, motion_item in enumerate(semantic_seeds):
        label = motion_item.expected_entities[0].label
        matching = [
            item.case_id
            for item in motion_by_id.values()
            if item.task_type == motion_item.task_type and item.expected_entities[0].label == label
        ]
        cases.append(
            _case(
                query_id=f"semantic-en-{index:02d}",
                text=f"A scene where the {label} {_ACTION_EN[motion_item.task_type]}",
                query_type="semantic",
                relevant_ids=matching,
                results=results,
                evidence_types=("structured_text", "keyframe"),
            )
        )
        cases.append(
            _case(
                query_id=f"semantic-zh-{index:02d}",
                text=(f"{_ENTITY_ZH.get(label, label)}{_ACTION_ZH[motion_item.task_type]}的场景"),
                query_type="semantic",
                relevant_ids=matching,
                results=results,
                evidence_types=("structured_text", "keyframe"),
            )
        )

    for index, task_type in enumerate(sorted(_ACTION_EN)):
        relevant_ids = [
            gold.case_id for gold in motion_by_id.values() if gold.task_type == task_type
        ]
        for language, text in (
            ("en", f"Find the scene where an object {_ACTION_EN[task_type]}"),
            ("zh", f"找到物体{_ACTION_ZH[task_type]}的场景"),
        ):
            cases.append(
                _case(
                    query_id=f"action-{language}-{index:02d}",
                    text=text,
                    query_type="action",
                    relevant_ids=relevant_ids,
                    results=results,
                    evidence_types=("structured_text", "keyframe"),
                )
            )

        candidates = sorted(
            (gold for gold in motion_by_id.values() if gold.task_type == task_type),
            key=lambda item: item.case_id,
        )
        chosen_label = candidates[0].expected_entities[0].label
        exact_ids = [
            gold.case_id for gold in candidates if gold.expected_entities[0].label == chosen_label
        ]
        for language, text in (
            (
                "en",
                f"Find the {chosen_label} that {_ACTION_EN[task_type]}; "
                "exclude the same object doing another action",
            ),
            (
                "zh",
                f"找到{_ENTITY_ZH.get(chosen_label, chosen_label)}"
                f"{_ACTION_ZH[task_type]}的场景;排除同一物体的其他动作",
            ),
        ):
            cases.append(
                _case(
                    query_id=f"hard-negative-{language}-{index:02d}",
                    text=text,
                    query_type="hard_negative",
                    relevant_ids=exact_ids,
                    results=results,
                    evidence_types=("structured_text", "keyframe"),
                    hard_negative=True,
                )
            )

    if len(cases) != 150:
        raise ValueError(f"frozen M2A query set must have exactly 150 cases, got {len(cases)}")
    if len({case.query.query_id for case in cases}) != len(cases):
        raise ValueError("frozen M2A query identifiers must be unique")
    return tuple(cases)
