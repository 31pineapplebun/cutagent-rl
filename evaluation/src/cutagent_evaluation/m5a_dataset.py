"""Deterministic CutAgentBench v0.1 source and task construction."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.base import NonEmptyStr, SchemaModel
from cutagent.schemas.media import TimeRange
from cutagent.schemas.task_input import (
    AspectRatioConstraint,
    DurationConstraint,
    OutputRequest,
    RequiredContentConstraint,
    TaskInput,
)
from cutagent.schemas.tools import ToolName
from cutagent_evaluation.m5a_schemas import (
    AbsenceGoldConstraint,
    BenchmarkFailureCause,
    BenchmarkSourceRecord,
    CutAgentBenchCase,
    CutAgentBenchGold,
    DerivedArtifactGoldConstraint,
    DifficultyLevel,
    DurationGoldConstraint,
    ExpectedTerminalBehavior,
    FailureInjectionGold,
    GeneratedSceneAnnotation,
    GroundingConstraint,
    MediaFamily,
    ObjectiveConstraint,
    ResolutionGoldConstraint,
    SceneOrderGoldConstraint,
    SpeedGoldConstraint,
    StreamGoldConstraint,
    SubtitleGoldConstraint,
    TaskFamily,
)
from cutagent_evaluation.schemas import DatasetSplit

M5A_GENERATOR_VERSION = "cutagentbench-generated-media-v1"
SCENE_DURATION_MS = 3000
SCENES_PER_VIDEO = 5
VIDEO_DURATION_MS = SCENE_DURATION_MS * SCENES_PER_VIDEO
FRAME_RATE = 8
WIDTH = 384
HEIGHT = 256

SPLIT_SOURCE_COUNTS: dict[DatasetSplit, int] = {
    DatasetSplit.TRAIN: 8,
    DatasetSplit.DEV: 8,
    DatasetSplit.VALIDATION: 10,
    DatasetSplit.LOCKED_TEST: 12,
    DatasetSplit.ADVERSARIAL_TEST: 6,
}

Action = Literal[
    "stationary",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "appear",
    "disappear",
    "approach",
    "move_away",
]


class SourceBlueprint(SchemaModel):
    source_group_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    split: DatasetSplit
    global_index: int = Field(ge=0)
    seed: int = Field(ge=0)
    primary_entity: NonEmptyStr
    companion_entities: tuple[NonEmptyStr, ...] = Field(min_length=5, max_length=5)
    actions: tuple[Action, ...] = Field(min_length=5, max_length=5)
    visible_texts: tuple[NonEmptyStr, ...] = Field(min_length=5, max_length=5)
    transcripts: tuple[NonEmptyStr, ...] = Field(min_length=5, max_length=5)
    pattern_variant: int = Field(ge=0)


def _canonical_bytes(value: object) -> bytes:
    if isinstance(value, SchemaModel):
        payload: Any = value.model_dump(mode="json", exclude={"schema_version"})
    else:
        payload = value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def build_source_blueprints() -> tuple[SourceBlueprint, ...]:
    """Return the frozen 44-source split-disjoint construction plan."""

    entities = (
        "red square",
        "blue circle",
        "green triangle",
        "yellow diamond",
        "purple hexagon",
        "orange circle",
        "cyan square",
        "magenta triangle",
        "navy diamond",
        "lime hexagon",
        "maroon square",
    )
    action_cycles: tuple[tuple[Action, ...], ...] = (
        ("stationary", "appear", "move_right", "move_left", "disappear"),
        ("move_up", "stationary", "move_down", "approach", "move_away"),
        ("appear", "move_left", "stationary", "move_right", "disappear"),
        ("approach", "move_up", "move_away", "move_down", "stationary"),
    )
    records: list[SourceBlueprint] = []
    global_index = 0
    for split, count in SPLIT_SOURCE_COUNTS.items():
        for split_index in range(count):
            opaque_group = hashlib.sha256(f"cab-v01-source-{global_index}".encode()).hexdigest()[
                :16
            ]
            primary = entities[global_index % len(entities)]
            companions = tuple(
                entities[(global_index * 3 + scene_index + 2) % len(entities)]
                for scene_index in range(SCENES_PER_VIDEO)
            )
            actions = action_cycles[global_index % len(action_cycles)]
            marker = hashlib.sha256(f"cab-marker-{global_index}".encode()).hexdigest()[:6].upper()
            visible = tuple(f"CAB {marker} PANEL {index + 1}" for index in range(5))
            transcripts = tuple(
                (
                    f"Marker {marker} scene {index + 1} the {primary} is "
                    f"{actions[index].replace('_', ' ')}"
                )
                for index in range(5)
            )
            records.append(
                SourceBlueprint(
                    source_group_id=f"cab-source-{opaque_group}",
                    split=split,
                    global_index=global_index,
                    seed=50_000 + global_index,
                    primary_entity=primary,
                    companion_entities=companions,
                    actions=actions,
                    visible_texts=visible,
                    transcripts=transcripts,
                    pattern_variant=(global_index * 7 + split_index) % 13,
                )
            )
            global_index += 1
    return tuple(records)


def _font(size: int) -> Any:
    font_module: Any = __import__("PIL.ImageFont", fromlist=["ImageFont"])
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ):
        if candidate.is_file():
            return font_module.truetype(str(candidate), size=size)
    return font_module.load_default()


def _draw_shape(draw: Any, label: str, x: float, y: float, size: float) -> None:
    color, shape = label.split()
    half = size / 2
    box = (round(x - half), round(y - half), round(x + half), round(y + half))
    if shape == "circle":
        draw.ellipse(box, fill=color, outline="black", width=3)
    elif shape == "triangle":
        draw.polygon(
            (
                (round(x), round(y - half)),
                (round(x - half), round(y + half)),
                (round(x + half), round(y + half)),
            ),
            fill=color,
            outline="black",
        )
    elif shape == "diamond":
        draw.polygon(
            (
                (round(x), round(y - half)),
                (round(x - half), round(y)),
                (round(x), round(y + half)),
                (round(x + half), round(y)),
            ),
            fill=color,
            outline="black",
        )
    elif shape == "hexagon":
        quarter = half / 2
        draw.polygon(
            (
                (round(x - half), round(y)),
                (round(x - quarter), round(y - half)),
                (round(x + quarter), round(y - half)),
                (round(x + half), round(y)),
                (round(x + quarter), round(y + half)),
                (round(x - quarter), round(y + half)),
            ),
            fill=color,
            outline="black",
        )
    else:
        draw.rectangle(box, fill=color, outline="black", width=3)


def _motion_state(action: Action, local_ms: int) -> tuple[bool, float, float, float]:
    progress = min(1.0, local_ms / (SCENE_DURATION_MS - 125))
    if action == "appear":
        return local_ms >= 1000, 95.0, 150.0, 62.0
    if action == "disappear":
        return local_ms < 1800, 95.0, 150.0, 62.0
    if action == "move_right":
        return True, 55.0 + 245.0 * progress, 150.0, 62.0
    if action == "move_left":
        return True, 300.0 - 245.0 * progress, 150.0, 62.0
    if action == "move_up":
        return True, 110.0, 205.0 - 125.0 * progress, 62.0
    if action == "move_down":
        return True, 110.0, 80.0 + 125.0 * progress, 62.0
    if action == "approach":
        return True, 110.0, 150.0, 32.0 + 58.0 * progress
    if action == "move_away":
        return True, 110.0, 150.0, 90.0 - 58.0 * progress
    return True, 105.0, 150.0, 62.0


def _run(command: list[str], *, label: str) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = (
            completed.stderr.strip().splitlines()[-1]
            if completed.stderr.strip()
            else "unknown error"
        )
        raise RuntimeError(f"{label} failed: {message}")


def materialize_source(
    root: Path,
    blueprint: SourceBlueprint,
    *,
    ffmpeg: str = "ffmpeg",
) -> BenchmarkSourceRecord:
    """Generate and hash one immutable CC0 benchmark source video."""

    case_root = root / blueprint.source_group_id
    frames_root = case_root / "frames"
    audio_root = case_root / "audio"
    frames_root.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)
    font = _font(19)
    image_module: Any = __import__("PIL.Image", fromlist=["Image"])
    draw_module: Any = __import__("PIL.ImageDraw", fromlist=["ImageDraw"])
    background_palette = ("#f8f8f8", "#dcecff", "#fff0cc", "#e8dcff", "#dff5e2")
    for scene_index in range(SCENES_PER_VIDEO):
        audio = audio_root / f"scene_{scene_index}.wav"
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
                f"flite=text='{blueprint.transcripts[scene_index]}'",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(audio),
            ],
            label="CutAgentBench speech generation",
        )
    frame_count = VIDEO_DURATION_MS * FRAME_RATE // 1000
    for frame_index in range(frame_count):
        timestamp_ms = frame_index * 1000 // FRAME_RATE
        scene_index = min(SCENES_PER_VIDEO - 1, timestamp_ms // SCENE_DURATION_MS)
        local_ms = timestamp_ms - scene_index * SCENE_DURATION_MS
        palette_index = (blueprint.pattern_variant + scene_index) % len(background_palette)
        image = image_module.new("RGB", (WIDTH, HEIGHT), background_palette[palette_index])
        draw = draw_module.Draw(image)
        stripe = 14 + (blueprint.pattern_variant % 5) * 3
        for stripe_x in range(0, WIDTH, stripe * 2):
            draw.rectangle((stripe_x, HEIGHT - 18, stripe_x + stripe, HEIGHT), fill="#b7b7b7")
        draw.text((12, 9), blueprint.visible_texts[scene_index], font=font, fill="#202020")
        visible, position_x, position_y, size = _motion_state(
            blueprint.actions[scene_index], local_ms
        )
        if visible:
            _draw_shape(draw, blueprint.primary_entity, position_x, position_y, size)
        _draw_shape(draw, blueprint.companion_entities[scene_index], 325.0, 196.0, 38.0)
        image.save(frames_root / f"frame_{frame_index:04d}.png")
    source_path = case_root / "source.mp4"
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
    filters += ";[a0][a1][a2][a3][a4]concat=n=5:v=0:a=1[aout]"
    command.extend(
        (
            "-filter_complex",
            filters,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FRAME_RATE),
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(source_path),
        )
    )
    _run(command, label="CutAgentBench source encoding")
    temporary = ArtifactRef.from_path(
        source_path,
        artifact_id=f"cab-video-{blueprint.global_index:03d}",
        media_type="video/mp4",
    )
    video_id = f"cab-video-{temporary.sha256[:20]}"
    artifact = temporary.model_copy(update={"artifact_id": video_id})
    scenes = tuple(
        GeneratedSceneAnnotation(
            scene_id=f"{blueprint.source_group_id}-scene-{index + 1}",
            time_range=TimeRange(
                start_ms=index * SCENE_DURATION_MS,
                end_ms=(index + 1) * SCENE_DURATION_MS,
            ),
            entities=(blueprint.primary_entity, blueprint.companion_entities[index]),
            action=blueprint.actions[index],
            visible_text=blueprint.visible_texts[index],
            transcript=blueprint.transcripts[index],
        )
        for index in range(SCENES_PER_VIDEO)
    )
    structural = _digest(
        {
            "primary": blueprint.primary_entity,
            "companions": blueprint.companion_entities,
            "actions": blueprint.actions,
            "texts": blueprint.visible_texts,
            "pattern": blueprint.pattern_variant,
        }
    )
    return BenchmarkSourceRecord(
        source_group_id=blueprint.source_group_id,
        split=blueprint.split,
        media_family=MediaFamily.GENERATED,
        video_id=video_id,
        source_artifact=artifact,
        source_sha256=artifact.sha256,
        structural_fingerprint=structural,
        generator_version=M5A_GENERATOR_VERSION,
        provenance_reference="Generated by CutAgentBench v0.1 and dedicated to CC0-1.0",
        license="CC0-1.0",
        scenes=scenes,
    )


def materialize_source_registry(
    root: Path,
    *,
    ffmpeg: str = "ffmpeg",
) -> tuple[BenchmarkSourceRecord, ...]:
    return tuple(
        materialize_source(root, item, ffmpeg=ffmpeg) for item in build_source_blueprints()
    )


def _opaque_task_id(source_group_id: str, slot: int) -> str:
    suffix = hashlib.sha256(f"cab-v01-task|{source_group_id}|{slot}".encode()).hexdigest()[:20]
    return f"cab-task-{suffix}"


def _family_for(source: BenchmarkSourceRecord, slot: int, split_index: int) -> TaskFamily:
    if source.split == DatasetSplit.ADVERSARIAL_TEST:
        return (
            TaskFamily.HARD_NEGATIVE,
            TaskFamily.OBSERVABLE_RECOVERY,
            TaskFamily.IMPOSSIBLE,
            TaskFamily.HARD_NEGATIVE,
            TaskFamily.OBSERVABLE_RECOVERY,
        )[slot]
    if slot < 4:
        return (
            TaskFamily.SEARCH_GROUNDING,
            TaskFamily.SINGLE_EDIT,
            TaskFamily.MULTI_SCENE_COMPOSITION,
            TaskFamily.MULTI_CONSTRAINT,
        )[slot]
    return (
        TaskFamily.HARD_NEGATIVE,
        TaskFamily.IMPOSSIBLE,
        TaskFamily.OBSERVABLE_RECOVERY,
    )[split_index % 3]


def _difficulty(
    *,
    tool_sequence: tuple[ToolName, ...],
    scene_count: int,
    constraint_count: int,
    hard_negative: bool,
    recovery: bool,
) -> DifficultyLevel:
    score = len(tool_sequence) + max(0, scene_count - 1) + max(0, constraint_count - 2)
    score += 2 if hard_negative else 0
    score += 3 if recovery else 0
    if score <= 3:
        return DifficultyLevel.L1
    if score <= 6:
        return DifficultyLevel.L2
    if score <= 9:
        return DifficultyLevel.L3
    return DifficultyLevel.L4


def _task_case(
    source: BenchmarkSourceRecord,
    *,
    slot: int,
    split_index: int,
) -> CutAgentBenchCase:
    family = _family_for(source, slot, split_index)
    stable_target = hashlib.sha256(f"{source.source_group_id}|{slot}".encode()).hexdigest()
    target_index = (int(stable_target[:8], 16) + split_index) % 5
    target = source.scenes[target_index]
    task_id = _opaque_task_id(source.source_group_id, slot)
    query_phrases = (
        f'the spoken quote "{target.transcript}"',
        f'the visible text "{target.visible_text}"',
        f"the {source.scenes[target_index].entities[0]}",
        f"the scene where the main object is {target.action.replace('_', ' ')}",
        f"panel {target_index + 1} with the companion {target.entities[1]}",
    )
    query = query_phrases[(split_index + slot) % len(query_phrases)]
    scene_ids: tuple[str, ...] = (target.scene_id,)
    ranges: tuple[TimeRange, ...] = (target.time_range,)
    required: tuple[ToolName, ...] = ("search_video", "trim_video", "validate_media")
    sequence: tuple[ToolName, ...] = required
    public_constraints: list[Any] = [
        DurationConstraint(min_ms=2850, max_ms=3150),
        RequiredContentConstraint(description=query),
    ]
    constraints: list[ObjectiveConstraint] = [
        GroundingConstraint(
            relevant_video_ids=(source.video_id,),
            relevant_scene_ids=scene_ids,
            acceptable_time_ranges=ranges,
        ),
        DurationGoldConstraint(target_ms=SCENE_DURATION_MS),
        DerivedArtifactGoldConstraint(),
        StreamGoldConstraint(require_video=True, require_audio=True),
    ]
    subtype = "locate_scene"
    instruction = f"Find {query} and export exactly that complete scene. Validate the result."
    failure: FailureInjectionGold | None = None
    primary_failure = BenchmarkFailureCause.RETRIEVAL_ERROR

    if family == TaskFamily.SINGLE_EDIT:
        edit_kind = ("trim", "speed", "subtitle", "reframe")[split_index % 4]
        subtype = edit_kind
        if edit_kind == "speed":
            required = ("search_video", "trim_video", "change_speed", "validate_media")
            sequence = required
            instruction = f"Find {query}, export that scene at 2x speed, and validate it."
            constraints[1] = DurationGoldConstraint(target_ms=1500)
            constraints.append(SpeedGoldConstraint(speed_factor=2.0))
            public_constraints[0] = DurationConstraint(min_ms=1380, max_ms=1620)
        elif edit_kind == "subtitle":
            subtitle_text = "CutAgentBench verified"
            required = ("search_video", "trim_video", "add_subtitles", "validate_media")
            sequence = required
            instruction = (
                f'Find {query}, export it, add subtitle "{subtitle_text}" from 0 to 1500 ms, '
                "and validate it."
            )
            constraints.append(
                SubtitleGoldConstraint(
                    text=subtitle_text,
                    time_range=TimeRange(start_ms=0, end_ms=1500),
                )
            )
        elif edit_kind == "reframe":
            required = ("search_video", "trim_video", "reframe_video", "validate_media")
            sequence = required
            instruction = f"Find {query}, export it as 216 by 384 portrait video, and validate it."
            constraints.append(ResolutionGoldConstraint(width=216, height=384))
            public_constraints.append(AspectRatioConstraint(width=9, height=16))

    elif family == TaskFamily.MULTI_SCENE_COMPOSITION:
        other_index = (target_index + 2) % 5
        other = source.scenes[other_index]
        scene_ids = (target.scene_id, other.scene_id)
        ranges = (target.time_range, other.time_range)
        required = ("search_video", "trim_video", "concat_videos", "validate_media")
        sequence = (
            "search_video",
            "trim_video",
            "search_video",
            "trim_video",
            "concat_videos",
            "validate_media",
        )
        subtype = "ordered_two_scene_concat"
        instruction = (
            f"Export the scene containing {query}, followed by the scene with visible text "
            f'"{other.visible_text}". Concatenate in that order and validate.'
        )
        public_constraints[0] = DurationConstraint(min_ms=5750, max_ms=6250)
        constraints = [
            GroundingConstraint(
                relevant_video_ids=(source.video_id,),
                relevant_scene_ids=scene_ids,
                acceptable_time_ranges=ranges,
            ),
            DurationGoldConstraint(target_ms=6000, tolerance_ms=180),
            SceneOrderGoldConstraint(ordered_scene_ids=scene_ids),
            DerivedArtifactGoldConstraint(),
            StreamGoldConstraint(require_video=True, require_audio=True),
        ]

    elif family == TaskFamily.MULTI_CONSTRAINT:
        other_index = (target_index + 3) % 5
        other = source.scenes[other_index]
        scene_ids = (target.scene_id, other.scene_id)
        ranges = (target.time_range, other.time_range)
        subtitle_text = "M5 multi constraint"
        required = (
            "search_video",
            "trim_video",
            "concat_videos",
            "change_speed",
            "reframe_video",
            "add_subtitles",
            "validate_media",
        )
        sequence = (
            "search_video",
            "trim_video",
            "search_video",
            "trim_video",
            "concat_videos",
            "change_speed",
            "reframe_video",
            "add_subtitles",
            "validate_media",
        )
        subtype = "ordered_speed_portrait_subtitle"
        instruction = (
            f"Put the scene containing {query} before panel {other_index + 1}; make the result "
            f'2x speed and 216 by 384, add subtitle "{subtitle_text}" from 0 to 1500 ms, '
            "then validate."
        )
        public_constraints = [
            DurationConstraint(min_ms=2850, max_ms=3150),
            AspectRatioConstraint(width=9, height=16),
            RequiredContentConstraint(description=query),
        ]
        constraints = [
            GroundingConstraint(
                relevant_video_ids=(source.video_id,),
                relevant_scene_ids=scene_ids,
                acceptable_time_ranges=ranges,
            ),
            DurationGoldConstraint(target_ms=3000, tolerance_ms=180),
            ResolutionGoldConstraint(width=216, height=384),
            SceneOrderGoldConstraint(ordered_scene_ids=scene_ids),
            SpeedGoldConstraint(speed_factor=2.0),
            SubtitleGoldConstraint(
                text=subtitle_text, time_range=TimeRange(start_ms=0, end_ms=1500)
            ),
            DerivedArtifactGoldConstraint(),
            StreamGoldConstraint(require_video=True, require_audio=True),
        ]

    elif family == TaskFamily.HARD_NEGATIVE:
        subtype = (
            "same_entity_different_action",
            "same_text_adjacent_scene",
            "similar_transcript",
            "same_video_wrong_temporal_scene",
            "visually_similar_distractor",
        )[split_index % 5]
        instruction = (
            f"Several scenes are similar. Find only {query}; reject adjacent scenes with the "
            "wrong action or marker, export the full matching scene, and validate it."
        )

    elif family == TaskFamily.IMPOSSIBLE:
        absent_marker = (
            hashlib.sha256(f"absent-request|{source.source_group_id}|{slot}".encode())
            .hexdigest()[:5]
            .upper()
        )
        nonexistent = f"a silver star spelling ABSENT-{absent_marker} while rotating"
        subtype = ("absent_event", "unavailable_evidence", "incompatible_constraints")[
            split_index % 3
        ]
        instruction = (
            f"Find {nonexistent} and export it. If it is absent, say the task cannot be completed."
        )
        public_constraints = [RequiredContentConstraint(description=nonexistent)]
        constraints = [AbsenceGoldConstraint(requested_description=nonexistent)]
        required = ("search_video",)
        sequence = required
        scene_ids = ()
        ranges = ()
        primary_failure = BenchmarkFailureCause.PREMATURE_FINISH

    elif family == TaskFamily.OBSERVABLE_RECOVERY:
        categories = (
            ("search_no_results", "search_video", "error"),
            ("timeout", "trim_video", "timeout"),
            ("invalid_arguments", "trim_video", "invalid"),
            ("artifact_not_allowed", "trim_video", "error"),
            ("post_validation_failure", "validate_media", "error"),
            ("output_too_large", "trim_video", "error"),
        )
        category, injection_tool, status = categories[split_index % len(categories)]
        failure = FailureInjectionGold(
            category=cast(Any, category),
            inject_on_tool=cast(ToolName, injection_tool),
            expected_online_status=cast(Any, status),
        )
        subtype = category
        instruction = (
            f"Find {query}, export that complete scene, validate it, and recover safely from any "
            "observable tool failure."
        )
        primary_failure = BenchmarkFailureCause.RECOVERY_ERROR

    constraints_tuple = tuple(constraints)
    difficulty = _difficulty(
        tool_sequence=sequence,
        scene_count=len(scene_ids),
        constraint_count=len(constraints_tuple),
        hard_negative=family == TaskFamily.HARD_NEGATIVE,
        recovery=family == TaskFamily.OBSERVABLE_RECOVERY,
    )
    expected = (
        ExpectedTerminalBehavior.CANNOT_COMPLETE
        if family == TaskFamily.IMPOSSIBLE
        else ExpectedTerminalBehavior.SUCCESS
    )
    public = TaskInput(
        task_id=task_id,
        video_ref=source.source_artifact,
        instruction=instruction,
        user_constraints=tuple(public_constraints),
        requested_output=OutputRequest(container="mp4"),
    )
    gold = CutAgentBenchGold(
        gold_id=f"gold-{task_id}",
        task_id=task_id,
        source_group_id=source.source_group_id,
        split=source.split,
        task_family=family,
        task_subtype=subtype,
        difficulty=difficulty,
        relevant_video_ids=(source.video_id,) if scene_ids else (),
        relevant_scene_ids=scene_ids,
        acceptable_time_ranges=ranges,
        required_tool_capabilities=tuple(
            sorted(
                {
                    "retrieval.read" if tool == "search_video" else "media.read_or_write"
                    for tool in required
                }
            )
        ),
        required_tools=required,
        acceptable_tool_sequences=(sequence,),
        objective_constraints=constraints_tuple,
        expected_terminal_behavior=expected,
        failure_injection=failure,
        primary_failure_if_unsolved=primary_failure,
    )
    return CutAgentBenchCase(public_task=public, private_gold=gold)


def build_benchmark_cases(
    sources: tuple[BenchmarkSourceRecord, ...],
) -> tuple[CutAgentBenchCase, ...]:
    """Build exactly five deterministic tasks per materialized source."""

    by_split_index: dict[DatasetSplit, int] = {split: 0 for split in DatasetSplit}
    cases: list[CutAgentBenchCase] = []
    for source in sources:
        split_index = by_split_index[source.split]
        for slot in range(5):
            cases.append(_task_case(source, slot=slot, split_index=split_index))
        by_split_index[source.split] += 1
    return tuple(cases)
