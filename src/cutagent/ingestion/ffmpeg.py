"""Narrow subprocess wrapper for deterministic M1A FFmpeg operations."""

import subprocess
from collections.abc import Sequence

from cutagent.core.errors import MediaProcessingError


class FFmpegRunner:
    def __init__(self, executable: str = "ffmpeg", *, timeout_seconds: int = 120) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.version = self._detect_version()

    def _detect_version(self) -> str:
        completed = self.run(("-version",), operation="FFmpeg version probe")
        if not completed.stdout.strip():
            raise MediaProcessingError("FFmpeg version probe returned no version")
        return completed.stdout.splitlines()[0].strip()

    def run(
        self,
        arguments: Sequence[str],
        *,
        operation: str,
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                [self.executable, *arguments],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise MediaProcessingError(f"{operation} could not execute: {error}") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise MediaProcessingError(
                f"{operation} failed with exit code {completed.returncode}: {detail[-2000:]}"
            )
        return completed
