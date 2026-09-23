"""Deterministic held-out multi-scene media and 96-query M2B validation set."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import PerceptionResult, ScenePerception
from cutagent.schemas.retrieval import QueryIntent, RetrievalEvidenceType, RetrievalQuery
from cutagent_evaluation.m2b_retrieval import (
    FrozenM2BValidationCase,
    HardNegativeCategory,
    M2BRetrievalCaseGold,
)
from cutagent_evaluation.retrieval import RelevantSceneGold, RetrievalQueryType

SCENE_DURATION_MS = 3000
SCENES_PER_VIDEO = 4
VIDEO_DURATION_MS = SCENE_DURATION_MS * SCENES_PER_VIDEO
FRAME_RATE = 8
FRAME_COUNT = VIDEO_DURATION_MS * FRAME_RATE // 1000
GENERATOR_VERSION = "m2b-heldout-multiscene-v1"

HeldOutAction = Literal["stationary", "appear", "move_left", "move_right", "disappear"]


class HeldOutSceneGold(SchemaModel):
    scene_index: int = Field(ge=0, lt=SCENES_PER_VIDEO)
    nominal_time_range: TimeRange
    primary_entity: NonEmptyStr
    companion_entity: NonEmptyStr
    action: HeldOutAction
    ocr_text: NonEmptyStr
    transcript: NonEmptyStr


class HeldOutVideoGold(SchemaModel):
    source_group_id: Identifier
    seed: int = Field(ge=0)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: Literal["CC0-1.0"] = "CC0-1.0"
    generator_version: Literal["m2b-heldout-multiscene-v1"] = "m2b-heldout-multiscene-v1"
    duration_ms: Literal[12000] = 12000
    scenes: tuple[HeldOutSceneGold, ...] = Field(min_length=4, max_length=4)


def _font(size: int) -> Any:
    image_font: Any = __import__("PIL.ImageFont", fromlist=["ImageFont"])
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ):
        if candidate.is_file():
            return image_font.truetype(str(candidate), size=size)
    return image_font.load_default()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _draw_shape(draw: Any, label: str, x: float, y: float, size: float) -> None:
    color, shape = label.split()
    half = size / 2
    bounds = (round(x - half), round(y - half), round(x + half), round(y + half))
    if shape == "circle":
        draw.ellipse(bounds, fill=color, outline="black", width=3)
    else:
        draw.rectangle(bounds, fill=color, outline="black", width=3)


def _primary_state(action: HeldOutAction, local_ms: int) -> tuple[bool, float]:
    if action == "appear":
        return local_ms >= 1000, 105.0
    if action == "disappear":
        return local_ms < 1600, 105.0
    if action == "move_right":
        return True, 55.0 + 245.0 * min(1.0, local_ms / 2875)
    if action == "move_left":
        return True, 300.0 - 245.0 * min(1.0, local_ms / 2875)
    return True, 105.0


def _run(command: list[str], *, label: str) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed: {completed.stderr}")


def generate_heldout_video(
    root: Path,
    *,
    video_index: int,
    seed: int,
    ffmpeg: str = "ffmpeg",
) -> tuple[Path, HeldOutVideoGold]:
    """Generate one CC0-dedicated four-scene video with exact private annotations."""

    if not 0 <= video_index < 8:
        raise ValueError("M2B held-out video index must be in [0, 8)")
    case_root = root / f"heldout-{video_index + 1:02d}"
    frames_root = case_root / "frames"
    audio_root = case_root / "audio"
    frames_root.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)
    primaries = (
        "red square",
        "blue circle",
        "green square",
        "yellow circle",
        "purple square",
        "orange circle",
        "cyan square",
        "magenta circle",
    )
    companions = (
        "green circle",
        "yellow square",
        "blue square",
        "red circle",
        "orange square",
        "purple circle",
        "magenta square",
        "cyan circle",
    )
    backgrounds = ("#f8f8f8", "#dcecff", "#fff0cc", "#e8dcff")
    action_three: HeldOutAction = "move_right" if video_index % 2 == 0 else "move_left"
    actions: tuple[HeldOutAction, ...] = (
        "stationary",
        "appear",
        action_three,
        "disappear",
    )
    primary = primaries[video_index]
    scene_gold: list[HeldOutSceneGold] = []
    font = _font(21)
    image_module: Any = __import__("PIL.Image", fromlist=["Image"])
    image_draw: Any = __import__("PIL.ImageDraw", fromlist=["ImageDraw"])
    for scene_index, action in enumerate(actions):
        companion = companions[(video_index + scene_index) % len(companions)]
        ocr_text = f"M2B{video_index + 1:02d} S{scene_index + 1}"
        transcript = f"Video {video_index + 1} scene {scene_index + 1} marker."
        start_ms = scene_index * SCENE_DURATION_MS
        scene_gold.append(
            HeldOutSceneGold(
                scene_index=scene_index,
                nominal_time_range=TimeRange(
                    start_ms=start_ms,
                    end_ms=start_ms + SCENE_DURATION_MS,
                ),
                primary_entity=primary,
                companion_entity=companion,
                action=action,
                ocr_text=ocr_text,
                transcript=transcript,
            )
        )
        audio_path = audio_root / f"scene_{scene_index}.wav"
        _run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"flite=text='{transcript}'",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(audio_path),
            ],
            label="M2B flite speech generation",
        )
    for frame_index in range(FRAME_COUNT):
        timestamp_ms = frame_index * 1000 // FRAME_RATE
        scene_index = min(SCENES_PER_VIDEO - 1, timestamp_ms // SCENE_DURATION_MS)
        local_ms = timestamp_ms - scene_index * SCENE_DURATION_MS
        specification = scene_gold[scene_index]
        image = image_module.new("RGB", (384, 256), backgrounds[scene_index])
        draw = image_draw.Draw(image)
        draw.text((12, 10), specification.ocr_text, font=font, fill="#202020")
        visible, x = _primary_state(specification.action, local_ms)
        if visible:
            _draw_shape(draw, primary, x, 145.0, 62.0)
        _draw_shape(draw, specification.companion_entity, 325.0, 190.0, 38.0)
        image.save(frames_root / f"frame_{frame_index:04d}.png")
    source = case_root / "source.mp4"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(FRAME_RATE),
        "-i",
        str(frames_root / "frame_%04d.png"),
    ]
    for scene_index in range(SCENES_PER_VIDEO):
        command.extend(("-i", str(audio_root / f"scene_{scene_index}.wav")))
    filters = ";".join(
        f"[{index + 1}:a]apad,atrim=0:3[a{index}]" for index in range(SCENES_PER_VIDEO)
    )
    filters += ";[a0][a1][a2][a3]concat=n=4:v=0:a=1[aout]"
    command.extend(
        (
            "-filter_complex",
            filters,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-t",
            "12",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "16000",
            str(source),
        )
    )
    _run(command, label="M2B multi-scene video generation")
    return source, HeldOutVideoGold(
        source_group_id=f"m2b-heldout-source-{video_index + 1:02d}",
        seed=seed,
        source_sha256=_sha256(source),
        scenes=tuple(scene_gold),
    )


def generate_heldout_dataset(
    root: Path, *, seed: int = 20260823, ffmpeg: str = "ffmpeg"
) -> tuple[tuple[Path, HeldOutVideoGold], ...]:
    return tuple(
        generate_heldout_video(
            root,
            video_index=index,
            seed=seed + index,
            ffmpeg=ffmpeg,
        )
        for index in range(8)
    )


def _actual_scene(result: PerceptionResult, nominal: TimeRange) -> ScenePerception:
    midpoint = (nominal.start_ms + nominal.end_ms) // 2
    matches = [
        scene
        for scene in result.world_state.scenes
        if scene.time_range.start_ms <= midpoint < scene.time_range.end_ms
    ]
    if len(matches) != 1:
        raise ValueError("nominal held-out scene does not map to exactly one M1A scene")
    return matches[0]


def _evidence_available(
    scene: ScenePerception,
    query_type: RetrievalQueryType,
    *,
    require_action: bool,
) -> bool:
    if require_action:
        return bool(scene.actions or scene.temporal_events)
    if query_type == "speech":
        return bool(scene.transcript_spans)
    if query_type == "ocr":
        return bool(scene.ocr_spans)
    if query_type == "entity":
        return bool(scene.entities or scene.keyframe_evidence)
    if query_type in {"semantic", "action", "hard_negative"}:
        return bool(scene.scene_summary or scene.entities)
    return False


def _make_case(
    *,
    query_id: str,
    text: str,
    query_type: RetrievalQueryType,
    source_group_id: str,
    result: PerceptionResult,
    gold_scene: HeldOutSceneGold,
    expected_evidence: tuple[RetrievalEvidenceType, ...],
    expected_route: QueryIntent,
    video_filter: bool,
    hard_category: HardNegativeCategory | None = None,
    require_action: bool = False,
) -> FrozenM2BValidationCase:
    scene = _actual_scene(result, gold_scene.nominal_time_range)
    return FrozenM2BValidationCase(
        query=RetrievalQuery(
            query_id=query_id,
            text=text,
            top_k=10,
            video_id=result.world_state.video_id if video_filter else None,
        ),
        gold=M2BRetrievalCaseGold(
            query_id=query_id,
            source_group_id=source_group_id,
            query_type=query_type,
            relevant_scenes=(
                RelevantSceneGold(
                    video_id=result.world_state.video_id,
                    scene_id=scene.segment_id,
                    acceptable_time_range=gold_scene.nominal_time_range,
                    expected_evidence_types=expected_evidence,
                ),
            ),
            hard_negative_category=hard_category,
            required_evidence_available=_evidence_available(
                scene,
                query_type,
                require_action=require_action,
            ),
            expected_route=expected_route,
        ),
    )


def build_heldout_validation_cases(
    records: tuple[HeldOutVideoGold, ...],
    results_by_source_group: dict[str, PerceptionResult],
) -> tuple[FrozenM2BValidationCase, ...]:
    """Build exactly 96 cases; no result text is used to write the private answers."""

    cases: list[FrozenM2BValidationCase] = []
    secondary_hard: tuple[HardNegativeCategory, ...] = (
        "same_text_different_scene",
        "semantically_similar_transcript",
        "visually_similar_distractor",
        "same_entity_different_action",
    )
    for video_index, record in enumerate(records):
        result = results_by_source_group[record.source_group_id]
        scenes = record.scenes
        prefix = f"m2b-v{video_index + 1:02d}"
        for local_index in (0, 1):
            gold = scenes[local_index]
            cases.append(
                _make_case(
                    query_id=f"{prefix}-speech-{local_index}",
                    text=f'Find where the narration says "{gold.transcript}"',
                    query_type="speech",
                    source_group_id=record.source_group_id,
                    result=result,
                    gold_scene=gold,
                    expected_evidence=("transcript",),
                    expected_route="speech_or_quote",
                    video_filter=False,
                )
            )
        for local_index in (2, 3):
            gold = scenes[local_index]
            cases.append(
                _make_case(
                    query_id=f"{prefix}-ocr-{local_index}",
                    text=f'Find the scene showing the exact text "{gold.ocr_text}"',
                    query_type="ocr",
                    source_group_id=record.source_group_id,
                    result=result,
                    gold_scene=gold,
                    expected_evidence=("ocr",),
                    expected_route="visible_text",
                    video_filter=False,
                )
            )
        for local_index in (0, 1):
            gold = scenes[local_index]
            cases.append(
                _make_case(
                    query_id=f"{prefix}-entity-{local_index}",
                    text=f"Find the scene containing a {gold.companion_entity}",
                    query_type="entity",
                    source_group_id=record.source_group_id,
                    result=result,
                    gold_scene=gold,
                    expected_evidence=("structured_text", "keyframe"),
                    expected_route="static_visual_entity",
                    video_filter=True,
                )
            )
        for local_index in (1, 2):
            gold = scenes[local_index]
            cases.append(
                _make_case(
                    query_id=f"{prefix}-semantic-{local_index}",
                    text=(
                        f"A scene where the {gold.primary_entity} is beside a "
                        f"{gold.companion_entity}"
                    ),
                    query_type="semantic",
                    source_group_id=record.source_group_id,
                    result=result,
                    gold_scene=gold,
                    expected_evidence=("structured_text", "keyframe"),
                    expected_route="semantic_scene",
                    video_filter=True,
                )
            )
        for local_index in (2, 3):
            gold = scenes[local_index]
            action_text = gold.action.replace("_", " ")
            cases.append(
                _make_case(
                    query_id=f"{prefix}-action-{local_index}",
                    text=f"Find the {gold.primary_entity} that {action_text}",
                    query_type="action",
                    source_group_id=record.source_group_id,
                    result=result,
                    gold_scene=gold,
                    expected_evidence=("structured_text", "keyframe"),
                    expected_route="action_or_motion",
                    video_filter=True,
                    require_action=True,
                )
            )
        motion = scenes[2]
        cases.append(
            _make_case(
                query_id=f"{prefix}-hard-action",
                text=(
                    f"Find the {motion.primary_entity} that {motion.action.replace('_', ' ')}; "
                    "exclude the same object doing another action"
                ),
                query_type="hard_negative",
                source_group_id=record.source_group_id,
                result=result,
                gold_scene=motion,
                expected_evidence=("structured_text", "keyframe"),
                expected_route="action_or_motion",
                video_filter=True,
                hard_category="same_entity_different_action",
                require_action=True,
            )
        )
        selected = scenes[(video_index + 1) % SCENES_PER_VIDEO]
        category = secondary_hard[video_index % len(secondary_hard)]
        if category == "same_text_different_scene":
            text = f'Find only the scene with exact visible text "{selected.ocr_text}"'
            route: QueryIntent = "visible_text"
            expected_evidence: tuple[RetrievalEvidenceType, ...] = ("ocr",)
            requires_action = False
        elif category == "semantically_similar_transcript":
            text = f'Find the exact narration "{selected.transcript}"'
            route = "speech_or_quote"
            expected_evidence = ("transcript",)
            requires_action = False
        elif category == "visually_similar_distractor":
            text = f"Find the {selected.primary_entity} beside the {selected.companion_entity}"
            route = "static_visual_entity"
            expected_evidence = ("structured_text", "keyframe")
            requires_action = False
        else:
            text = f"Find the {selected.primary_entity} that {selected.action.replace('_', ' ')}"
            route = "action_or_motion"
            expected_evidence = ("structured_text", "keyframe")
            requires_action = True
        cases.append(
            _make_case(
                query_id=f"{prefix}-hard-secondary",
                text=text,
                query_type="hard_negative",
                source_group_id=record.source_group_id,
                result=result,
                gold_scene=selected,
                expected_evidence=expected_evidence,
                expected_route=route,
                video_filter=True,
                hard_category=category,
                require_action=requires_action,
            )
        )
    if len(cases) != 96:
        raise ValueError(f"M2B validation must contain exactly 96 queries, got {len(cases)}")
    if len({case.query.query_id for case in cases}) != len(cases):
        raise ValueError("M2B validation query identifiers must be unique")
    return tuple(cases)


def assert_source_group_disjoint(
    cases: tuple[FrozenM2BValidationCase, ...],
    prohibited_groups: set[str],
) -> None:
    observed = {case.gold.source_group_id for case in cases}
    overlap = observed & prohibited_groups
    if overlap:
        raise ValueError(f"M2B validation source groups overlap development: {sorted(overlap)}")
