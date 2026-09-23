"""Deterministic 51-case motion-centric synthetic set for M1B.5."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from cutagent.schemas.media import TimeRange

from .m1b5_motion import (
    MotionActionCode,
    MotionCaseGold,
    MotionDirection,
    MotionEntityGold,
    MotionEventGold,
    MotionTaskType,
)

FRAME_RATE = 8
DURATION_MS = 4000
FRAME_COUNT = FRAME_RATE * DURATION_MS // 1000
MOTION_TASK_TYPES: tuple[MotionTaskType, ...] = (
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
)


def _font(size: int) -> Any:
    image_font: Any = __import__("PIL.ImageFont", fromlist=["ImageFont"])
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return image_font.truetype(str(candidate), size=size)
    return image_font.load_default()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(timestamp_ms: int, start_ms: int, end_ms: int) -> float:
    return min(1.0, max(0.0, (timestamp_ms - start_ms) / (end_ms - start_ms)))


def _interpolate(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def _actor_state(
    task_type: MotionTaskType, timestamp_ms: int
) -> tuple[bool, float, float, float, bool, float, float, float]:
    """Return primary and secondary visibility/x/y/size at a normalized timestamp."""

    visible = True
    x, y, size = 192.0, 128.0, 58.0
    secondary_visible = task_type in {"a_before_b", "b_before_a"}
    secondary_x, secondary_y, secondary_size = 310.0, 190.0, 48.0
    full = _progress(timestamp_ms, 0, DURATION_MS - 125)
    if task_type == "move_left":
        x = _interpolate(320, 64, full)
    elif task_type == "move_right":
        x = _interpolate(64, 320, full)
    elif task_type == "move_up":
        y = _interpolate(205, 52, full)
    elif task_type == "move_down":
        y = _interpolate(52, 205, full)
    elif task_type == "appear":
        visible = timestamp_ms >= 1500
    elif task_type == "disappear":
        visible = timestamp_ms < 2500
    elif task_type == "enter_frame":
        x = _interpolate(-45, 120, _progress(timestamp_ms, 0, 1500))
    elif task_type == "exit_frame":
        x = _interpolate(190, 430, _progress(timestamp_ms, 2200, 3875))
    elif task_type == "approach":
        size = _interpolate(22, 108, _progress(timestamp_ms, 500, 3500))
    elif task_type == "move_away":
        size = _interpolate(108, 22, _progress(timestamp_ms, 500, 3500))
    elif task_type == "a_before_b":
        x = _interpolate(60, 170, _progress(timestamp_ms, 500, 1500))
        secondary_x = _interpolate(320, 210, _progress(timestamp_ms, 2200, 3200))
    elif task_type == "b_before_a":
        secondary_x = _interpolate(320, 210, _progress(timestamp_ms, 500, 1500))
        x = _interpolate(60, 170, _progress(timestamp_ms, 2200, 3200))
    elif task_type == "stop_then_start":
        if timestamp_ms < 1000:
            x = _interpolate(60, 140, _progress(timestamp_ms, 0, 1000))
        elif timestamp_ms < 2500:
            x = 140
        else:
            x = _interpolate(140, 320, _progress(timestamp_ms, 2500, 3875))
    elif task_type == "start_then_stop":
        if timestamp_ms < 1000:
            x = 60
        elif timestamp_ms < 2500:
            x = _interpolate(60, 300, _progress(timestamp_ms, 1000, 2500))
        else:
            x = 300
    elif task_type == "short_motion":
        x = _interpolate(110, 270, _progress(timestamp_ms, 1750, 2250))
    elif task_type == "long_motion":
        x = _interpolate(60, 320, _progress(timestamp_ms, 500, 3500))
    return visible, x, y, size, secondary_visible, secondary_x, secondary_y, secondary_size


def _draw_actor(draw: Any, shape: str, color: str, x: float, y: float, size: float) -> None:
    half = size / 2
    bounds = (round(x - half), round(y - half), round(x + half), round(y + half))
    if shape == "circle":
        draw.ellipse(bounds, fill=color, outline="black", width=3)
    else:
        draw.rectangle(bounds, fill=color, outline="black", width=3)


def _events(
    task_type: MotionTaskType,
    primary: MotionEntityGold,
    secondary: MotionEntityGold,
) -> tuple[MotionEventGold, ...]:
    actor_aliases = primary.aliases
    single: dict[MotionTaskType, tuple[MotionActionCode, int, int]] = {
        "move_left": ("move_left", 0, 4000),
        "move_right": ("move_right", 0, 4000),
        "move_up": ("move_up", 0, 4000),
        "move_down": ("move_down", 0, 4000),
        "appear": ("appear", 1375, 1625),
        "disappear": ("disappear", 2375, 2625),
        "enter_frame": ("enter_frame", 0, 1500),
        "exit_frame": ("exit_frame", 2200, 4000),
        "approach": ("approach", 500, 3500),
        "move_away": ("move_away", 500, 3500),
        "short_motion": ("move_right", 1750, 2250),
        "long_motion": ("move_right", 500, 3500),
    }
    if task_type in single:
        action, start_ms, end_ms = single[task_type]
        return (
            MotionEventGold(
                actor=primary.label,
                actor_aliases=actor_aliases,
                action=action,
                time_range=TimeRange(start_ms=start_ms, end_ms=end_ms),
            ),
        )
    if task_type in {"a_before_b", "b_before_a"}:
        first_primary = task_type == "a_before_b"
        return (
            MotionEventGold(
                actor=primary.label,
                actor_aliases=primary.aliases,
                action="move_right",
                time_range=TimeRange(
                    start_ms=500 if first_primary else 2200,
                    end_ms=1500 if first_primary else 3200,
                ),
            ),
            MotionEventGold(
                actor=secondary.label,
                actor_aliases=secondary.aliases,
                action="move_left",
                time_range=TimeRange(
                    start_ms=2200 if first_primary else 500,
                    end_ms=3200 if first_primary else 1500,
                ),
            ),
        )
    if task_type == "stop_then_start":
        return (
            MotionEventGold(
                actor=primary.label,
                actor_aliases=actor_aliases,
                action="stop_moving",
                time_range=TimeRange(start_ms=875, end_ms=1250),
            ),
            MotionEventGold(
                actor=primary.label,
                actor_aliases=actor_aliases,
                action="start_moving",
                time_range=TimeRange(start_ms=2375, end_ms=2750),
            ),
        )
    if task_type == "start_then_stop":
        return (
            MotionEventGold(
                actor=primary.label,
                actor_aliases=actor_aliases,
                action="start_moving",
                time_range=TimeRange(start_ms=875, end_ms=1250),
            ),
            MotionEventGold(
                actor=primary.label,
                actor_aliases=actor_aliases,
                action="stop_moving",
                time_range=TimeRange(start_ms=2375, end_ms=2750),
            ),
        )
    return ()


def generate_motion_case(
    root: Path,
    *,
    task_type: MotionTaskType,
    variant_index: int,
    seed: int,
    ffmpeg: str = "ffmpeg",
) -> tuple[Path, MotionCaseGold]:
    if not 0 <= variant_index <= 2:
        raise ValueError("motion variant_index must be 0, 1, or 2")
    case_index = MOTION_TASK_TYPES.index(task_type) * 3 + variant_index + 1
    case_id = f"motion-{case_index:03d}"
    case_root = root / case_id
    frames_root = case_root / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    image_module: Any = __import__("PIL.Image", fromlist=["Image"])
    image_draw: Any = __import__("PIL.ImageDraw", fromlist=["ImageDraw"])
    palettes = (
        ("red", "square", "blue", "circle", "white"),
        ("green", "circle", "yellow", "square", "#eeeeee"),
        ("blue", "square", "red", "circle", "#fff6df"),
    )
    primary_color, primary_shape, secondary_color, secondary_shape, background = palettes[
        variant_index
    ]
    primary = MotionEntityGold(
        label=f"{primary_color} {primary_shape}",
        aliases=(
            f"{primary_color} rectangle",
            f"{primary_color} box",
        )
        if primary_shape == "square"
        else (f"{primary_color} ball",),
    )
    secondary = MotionEntityGold(
        label=f"{secondary_color} {secondary_shape}",
        aliases=(
            f"{secondary_color} rectangle",
            f"{secondary_color} box",
        )
        if secondary_shape == "square"
        else (f"{secondary_color} ball",),
    )
    font = _font(18)
    for frame_index in range(FRAME_COUNT):
        timestamp_ms = frame_index * 1000 // FRAME_RATE
        image = image_module.new("RGB", (384, 256), background)
        draw = image_draw.Draw(image)
        draw.text((10, 8), f"CASE {case_index:03d}", font=font, fill="#333333")
        visible, x, y, size, second_visible, second_x, second_y, second_size = _actor_state(
            task_type, timestamp_ms
        )
        if visible:
            _draw_actor(draw, primary_shape, primary_color, x, y, size)
        if second_visible:
            _draw_actor(
                draw,
                secondary_shape,
                secondary_color,
                second_x,
                second_y,
                second_size,
            )
        image.save(frames_root / f"frame_{frame_index:03d}.png")
    source = case_root / "source.mp4"
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(FRAME_RATE),
            "-i",
            str(frames_root / "frame_%03d.png"),
            "-t",
            f"{DURATION_MS / 1000:.3f}",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"motion fixture generation failed: {completed.stderr}")
    expected_entities = (
        (primary, secondary) if task_type in {"a_before_b", "b_before_a"} else (primary,)
    )
    direction_by_task: dict[MotionTaskType, MotionDirection] = {
        "move_left": "left",
        "move_right": "right",
        "move_up": "up",
        "move_down": "down",
        "approach": "toward",
        "move_away": "away",
        "short_motion": "right",
        "long_motion": "right",
    }
    direction = direction_by_task.get(task_type)
    expected_order = None
    if task_type == "a_before_b":
        expected_order = (primary.label, secondary.label)
    elif task_type == "b_before_a":
        expected_order = (secondary.label, primary.label)
    gold = MotionCaseGold(
        case_id=case_id,
        source_group_id="generated-motion-m1b5-v1",
        task_type=task_type,
        seed=seed,
        duration_ms=DURATION_MS,
        source_sha256=_sha256(source),
        expected_entities=expected_entities,
        supported_entity_terms=(
            "background",
            "text",
            "label",
            "case number",
            "video frame",
        ),
        expected_events=_events(task_type, primary, secondary),
        expected_direction=direction,
        expected_order=expected_order,
        generation_config={
            "generator_version": "m1b5-motion-v1",
            "width": 384,
            "height": 256,
            "fps": FRAME_RATE,
            "frame_count": FRAME_COUNT,
            "duration_ms": DURATION_MS,
            "variant_index": variant_index,
            "primary": primary.label,
            "secondary": secondary.label,
            "background": background,
            "overlay": f"CASE {case_index:03d}",
        },
    )
    return source, gold


def generate_motion_dataset(
    root: Path, *, seed: int = 20260822, ffmpeg: str = "ffmpeg"
) -> tuple[tuple[Path, MotionCaseGold], ...]:
    records = []
    for task_index, task_type in enumerate(MOTION_TASK_TYPES):
        for variant_index in range(3):
            records.append(
                generate_motion_case(
                    root,
                    task_type=task_type,
                    variant_index=variant_index,
                    seed=seed + task_index * 3 + variant_index,
                    ffmpeg=ffmpeg,
                )
            )
    return tuple(records)
