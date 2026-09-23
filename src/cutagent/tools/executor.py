"""Private FFmpeg-only subprocess boundary; not exposed as a public tool."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cutagent.schemas.tools import ToolExecutionContext
from cutagent.tools.errors import ToolFailure, ToolTimeout


@dataclass(frozen=True, slots=True)
class ProcessResult:
    return_code: int
    stdout: str
    stderr: str
    latency_ms: int


class FFmpegExecutor:
    """Runs only an injected FFmpeg binary with internally constructed argv."""

    contract_version = "m3a-ffmpeg-argv-v1"

    def __init__(self, executable: str = "ffmpeg") -> None:
        if not executable or "\x00" in executable:
            raise ValueError("FFmpeg executable must be a non-empty program name")
        self.executable = executable
        probe_context = ToolExecutionContext(
            execution_id="ffmpeg-version-probe",
            allowed_output_root_id="runtime-probe",
            allowed_artifact_ids=(),
            allowed_capabilities=("media.inspect",),
            timeout_ms=30_000,
        )
        completed = self.run(("-version",), context=probe_context, operation="version probe")
        if completed.return_code != 0 or not completed.stdout.strip():
            raise ToolFailure("ffmpeg_error", "FFmpeg version probe failed")
        self.version = completed.stdout.splitlines()[0].strip()

    def run(
        self,
        arguments: Sequence[str],
        *,
        context: ToolExecutionContext,
        operation: str,
        cwd: Path | None = None,
    ) -> ProcessResult:
        if any(not isinstance(item, str) or "\x00" in item for item in arguments):
            raise ToolFailure("ffmpeg_error", "FFmpeg argument vector is invalid")
        started = time.perf_counter_ns()
        try:
            completed = subprocess.run(
                [self.executable, *arguments],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=context.timeout_ms / 1000,
                cwd=cwd,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolTimeout(f"{operation} exceeded the configured timeout") from exc
        except OSError as exc:
            raise ToolFailure("ffmpeg_error", f"{operation} could not start") from exc
        latency_ms = (time.perf_counter_ns() - started) // 1_000_000
        return ProcessResult(
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            latency_ms=latency_ms,
        )
