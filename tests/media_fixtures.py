"""Deterministic FFmpeg media generators used only by automated tests."""

import shutil
import subprocess
from pathlib import Path

import pytest


def require_ffmpeg() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("FFmpeg and ffprobe are required for media integration tests")
    return ffmpeg, ffprobe


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fixture generation failed: {completed.stderr}")


def generate_multiscene_with_audio(path: Path) -> Path:
    ffmpeg, _ = require_ffmpeg()
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x90:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=160x90:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=3",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "3:a",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            "-y",
            str(path),
        ]
    )
    return path


def generate_nonzero_start(path: Path) -> Path:
    ffmpeg, _ = require_ffmpeg()
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x90:r=10:d=1",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-output_ts_offset",
            "2",
            "-y",
            str(path),
        ]
    )
    return path


def generate_vfr_like(path: Path) -> Path:
    ffmpeg, _ = require_ffmpeg()
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x90:r=5:d=1",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x90:r=12:d=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0,settb=1/1000[v]",
            "-map",
            "[v]",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-vsync",
            "vfr",
            "-y",
            str(path),
        ]
    )
    return path


def generate_low_frame_rate(path: Path) -> Path:
    """Create four frames whose final display interval ends at two seconds."""

    ffmpeg, _ = require_ffmpeg()
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x90:r=2:d=2",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ]
    )
    return path
