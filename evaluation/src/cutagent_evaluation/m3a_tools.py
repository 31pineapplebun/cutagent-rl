"""Evaluator-private M3A engineering cases and metric aggregation."""

from __future__ import annotations

import math
import re
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, JsonValue

from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.event import ToolStatus
from cutagent.schemas.tools import (
    ToolExecutionContext,
    ToolExecutionRecord,
    ToolSpec,
    ValidateMediaArgs,
)
from cutagent.tools.errors import ToolFailure
from cutagent.tools.factory import create_media_tool_registry
from cutagent.tools.protocols import ToolResult
from cutagent.tools.registry import ToolRegistry

EvaluationSuite = Literal["synthetic", "licensed_real", "sequence", "failure_injection"]
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


class ToolCaseResult(SchemaModel):
    case_id: Identifier
    suite: EvaluationSuite
    tool_name: Identifier
    expected_status: ToolStatus
    expected_error: Identifier | None = None
    observed_status: ToolStatus
    observed_error: Identifier | None = None
    expectation_met: bool
    post_validation_passed: bool
    latency_ms: int = Field(ge=0)
    cache_hit: bool
    duration_error_ms: int | None = Field(default=None, ge=0)
    resolution_correct: bool | None = None
    input_artifact_ids: tuple[Identifier, ...] = ()
    output_artifact_id: Identifier | None = None
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ToolLatencySummary(SchemaModel):
    tool_name: Identifier
    count: int = Field(ge=0)
    median_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)


class M3AEvaluationSummary(SchemaModel):
    suite: NonEmptyStr
    total_cases: int = Field(ge=0)
    execution_success_rate: float = Field(ge=0, le=1)
    expected_outcome_rate: float = Field(ge=0, le=1)
    post_validation_success_rate: float = Field(ge=0, le=1)
    structured_observation_validity_rate: float = Field(ge=0, le=1)
    failure_injection_handling_rate: float | None = Field(default=None, ge=0, le=1)
    cache_hit_count: int = Field(ge=0)
    duration_error_median_ms: float | None = Field(default=None, ge=0)
    duration_error_p95_ms: float | None = Field(default=None, ge=0)
    resolution_correctness_rate: float | None = Field(default=None, ge=0, le=1)
    latency_by_tool: tuple[ToolLatencySummary, ...]
    sequence_success_rate: float | None = Field(default=None, ge=0, le=1)


def percentile(values: Sequence[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[position])


def summarize_cases(
    cases: Sequence[ToolCaseResult],
    *,
    suite: str,
    sequence_success_rate: float | None = None,
) -> M3AEvaluationSummary:
    by_tool: dict[str, list[int]] = defaultdict(list)
    duration_errors: list[int] = []
    resolutions: list[bool] = []
    failures: list[ToolCaseResult] = []
    for case in cases:
        by_tool[case.tool_name].append(case.latency_ms)
        if case.duration_error_ms is not None:
            duration_errors.append(case.duration_error_ms)
        if case.resolution_correct is not None:
            resolutions.append(case.resolution_correct)
        if case.suite == "failure_injection":
            failures.append(case)
    total = len(cases)
    denominator = max(total, 1)
    executable = [item for item in cases if item.expected_status == "success"]
    successful = sum(item.observed_status == "success" for item in executable)
    expected = sum(item.expectation_met for item in cases)
    post_valid = sum(item.post_validation_passed for item in executable)
    latency = tuple(
        ToolLatencySummary(
            tool_name=tool,
            count=len(values),
            median_ms=float(statistics.median(values)),
            p95_ms=percentile(values, 0.95),
        )
        for tool, values in sorted(by_tool.items())
    )
    return M3AEvaluationSummary(
        suite=suite,
        total_cases=total,
        execution_success_rate=successful / max(len(executable), 1),
        expected_outcome_rate=expected / denominator,
        post_validation_success_rate=post_valid / max(len(executable), 1),
        structured_observation_validity_rate=1.0,
        failure_injection_handling_rate=(
            sum(item.expectation_met for item in failures) / len(failures) if failures else None
        ),
        cache_hit_count=sum(item.cache_hit for item in cases),
        duration_error_median_ms=(
            float(statistics.median(duration_errors)) if duration_errors else None
        ),
        duration_error_p95_ms=(percentile(duration_errors, 0.95) if duration_errors else None),
        resolution_correctness_rate=(sum(resolutions) / len(resolutions) if resolutions else None),
        latency_by_tool=latency,
        sequence_success_rate=sequence_success_rate,
    )


def generate_media_fixture(
    path: Path,
    *,
    ffmpeg: str,
    color: str,
    width: int = 160,
    height: int = 90,
    duration_seconds: float = 2.0,
    with_audio: bool = True,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={width}x{height}:r=10:d={duration_seconds}",
    ]
    if with_audio:
        frequency = {"red": 440, "blue": 550, "green": 660, "yellow": 770}.get(color, 440)
        command.extend(
            (
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:sample_rate=16000:duration={duration_seconds}",
            )
        )
    command.extend(("-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"))
    if with_audio:
        command.extend(("-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"))
    command.extend(("-movflags", "+faststart", "-y", str(path)))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"M3A fixture generation failed: {completed.stderr}")
    return path


def _artifact_arguments(call: Mapping[str, Any]) -> tuple[str, ...]:
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping):
        return ()
    single = arguments.get("input_artifact_id")
    multiple = arguments.get("input_artifact_ids")
    if isinstance(single, str) and _IDENTIFIER.fullmatch(single):
        return (single,)
    if (
        isinstance(multiple, list)
        and all(isinstance(item, str) for item in multiple)
        and all(_IDENTIFIER.fullmatch(item) for item in cast(list[str], multiple))
    ):
        return tuple(cast(list[str], multiple))
    return ()


def run_case(
    registry: ToolRegistry,
    context: ToolExecutionContext,
    *,
    case_id: str,
    suite: EvaluationSuite,
    call: Mapping[str, Any],
    expected_status: ToolStatus = "success",
    expected_error: str | None = None,
    expected_dimensions: tuple[int, int] | None = None,
) -> tuple[ToolCaseResult, ToolExecutionRecord]:
    record = registry.execute(call, context)
    observation = ToolExecutionRecord.model_validate(record.model_dump()).observation
    observed_error = observation.error_code
    expectation_met = observation.status == expected_status and (
        expected_error is None or observed_error == expected_error
    )
    validations = record.trace.validation_results
    post_valid = (
        all(item.passed for item in validations)
        if observation.status == "success"
        else expectation_met
    )
    duration_error = observation.details.get("duration_error_ms")
    normalized_duration_error = duration_error if isinstance(duration_error, int) else None
    resolution_correct = None
    if expected_dimensions is not None:
        resolution_correct = (
            observation.details.get("width"),
            observation.details.get("height"),
        ) == expected_dimensions
    output = record.trace.output_artifact
    result = ToolCaseResult(
        case_id=case_id,
        suite=suite,
        tool_name=str(call.get("tool_name", "invalid_tool")),
        expected_status=expected_status,
        expected_error=expected_error,
        observed_status=observation.status,
        observed_error=observed_error,
        expectation_met=expectation_met,
        post_validation_passed=post_valid,
        latency_ms=record.trace.latency_ms,
        cache_hit=record.trace.cache_hit,
        duration_error_ms=normalized_duration_error,
        resolution_correct=resolution_correct,
        input_artifact_ids=_artifact_arguments(call),
        output_artifact_id=output.artifact_id if output is not None else None,
        output_sha256=output.sha256 if output is not None else None,
    )
    return result, record


def context_for(
    *,
    execution_id: str,
    artifact_ids: Sequence[str],
    timeout_ms: int = 120_000,
    maximum_output_bytes: int = 1_000_000_000,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        execution_id=execution_id,
        allowed_output_root_id="m3a-evaluation",
        allowed_artifact_ids=tuple(sorted(set(artifact_ids))),
        allowed_capabilities=(
            "retrieval.read",
            "media.inspect",
            "media.decode",
            "media.write",
            "audio.write",
        ),
        timeout_ms=timeout_ms,
        maximum_output_bytes=maximum_output_bytes,
    )


def build_media_registry(root: Path, *, ffmpeg: str, ffprobe: str) -> ToolRegistry:
    return create_media_tool_registry(
        root=root,
        ffmpeg_executable=ffmpeg,
        ffprobe_executable=ffprobe,
    )


class InjectedOutputValidationFailureTool:
    """Evaluator-only fault used to exercise the registry's error boundary."""

    arguments_type = ValidateMediaArgs
    spec = ToolSpec(
        name="validate_media",
        version="m3a-injected-validation-failure-v1",
        description="Evaluator-only injected post-validation failure.",
        capabilities=("media.inspect", "media.decode"),
        argument_schema=cast(dict[str, JsonValue], ValidateMediaArgs.model_json_schema()),
        deterministic=True,
        produces_artifact=False,
    )

    def execute(
        self,
        arguments: SchemaModel,
        context: ToolExecutionContext,
        *,
        tool_call_id: str,
    ) -> ToolResult:
        del arguments, context, tool_call_id
        raise ToolFailure(
            "output_validation_failed",
            "controlled output validation failure",
        )
