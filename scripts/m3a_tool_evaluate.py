"""Run the real FFmpeg M3A synthetic, sequence, failure, or licensed evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, cast

from cutagent_evaluation.m3a_tools import (
    InjectedOutputValidationFailureTool,
    ToolCaseResult,
    build_media_registry,
    context_for,
    generate_media_fixture,
    run_case,
    summarize_cases,
)

from cutagent.schemas.tools import ToolExecutionContext, ToolExecutionRecord
from cutagent.tools.artifacts import ArtifactStore
from cutagent.tools.registry import ToolRegistry
from cutagent.tools.trace import ToolTraceRecorder


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _output_id(record: ToolExecutionRecord) -> str:
    if not record.observation.artifacts:
        raise RuntimeError("successful editing case did not produce an artifact")
    return record.observation.artifacts[0].artifact_id


def _normal_calls(
    source_ids: list[str],
) -> list[tuple[str, dict[str, Any], tuple[int, int] | None]]:
    calls: list[tuple[str, dict[str, Any], tuple[int, int] | None]] = []
    for index, artifact_id in enumerate(source_ids[:4], 1):
        calls.append(
            (
                f"inspect-{index:02d}",
                {
                    "tool_name": "inspect_media",
                    "tool_call_id": f"inspect-{index:02d}",
                    "arguments": {"input_artifact_id": artifact_id},
                },
                None,
            )
        )
        calls.append(
            (
                f"validate-{index:02d}",
                {
                    "tool_name": "validate_media",
                    "tool_call_id": f"validate-{index:02d}",
                    "arguments": {
                        "input_artifact_id": artifact_id,
                        "require_audio": True,
                    },
                },
                None,
            )
        )
    trim_ranges = (
        (0, 500),
        (0, 1000),
        (100, 800),
        (250, 1250),
        (500, 1500),
        (750, 1750),
        (1000, 2000),
        (125, 625),
        (400, 1400),
        (1500, 2000),
    )
    for index, (start_ms, end_ms) in enumerate(trim_ranges, 1):
        calls.append(
            (
                f"trim-{index:02d}",
                {
                    "tool_name": "trim_video",
                    "tool_call_id": f"trim-{index:02d}",
                    "arguments": {
                        "input_artifact_id": source_ids[(index - 1) % 4],
                        "time_range": {"start_ms": start_ms, "end_ms": end_ms},
                    },
                },
                None,
            )
        )
    concat_pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    for index, (left, right) in enumerate(concat_pairs, 1):
        calls.append(
            (
                f"concat-{index:02d}",
                {
                    "tool_name": "concat_videos",
                    "tool_call_id": f"concat-{index:02d}",
                    "arguments": {"input_artifact_ids": [source_ids[left], source_ids[right]]},
                },
                None,
            )
        )
    for index, factor in enumerate((0.5, 0.75, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0), 1):
        calls.append(
            (
                f"speed-{index:02d}",
                {
                    "tool_name": "change_speed",
                    "tool_call_id": f"speed-{index:02d}",
                    "arguments": {
                        "input_artifact_id": source_ids[(index - 1) % 4],
                        "speed_factor": factor,
                    },
                },
                None,
            )
        )
    dimensions = ((90, 160), (90, 160), (128, 128), (320, 180), (160, 90), (192, 108))
    for index, dimensions_item in enumerate(dimensions, 1):
        width, height = dimensions_item
        calls.append(
            (
                f"reframe-{index:02d}",
                {
                    "tool_name": "reframe_video",
                    "tool_call_id": f"reframe-{index:02d}",
                    "arguments": {
                        "input_artifact_id": source_ids[(index - 1) % 4],
                        "width": width,
                        "height": height,
                        "fit": "crop" if index % 2 else "pad",
                    },
                },
                dimensions_item,
            )
        )
    for index, text in enumerate(("M3A", "安全工具", "HALF OPEN", "CACHE"), 1):
        calls.append(
            (
                f"subtitle-{index:02d}",
                {
                    "tool_name": "add_subtitles",
                    "tool_call_id": f"subtitle-{index:02d}",
                    "arguments": {
                        "input_artifact_id": source_ids[index - 1],
                        "cues": [
                            {
                                "cue_id": f"cue-{index:02d}",
                                "time_range": {"start_ms": 250, "end_ms": 1500},
                                "text": text,
                            }
                        ],
                    },
                },
                None,
            )
        )
    for index, target in enumerate((-14.0, -16.0, -18.0, -20.0), 1):
        calls.append(
            (
                f"audio-{index:02d}",
                {
                    "tool_name": "normalize_audio",
                    "tool_call_id": f"audio-{index:02d}",
                    "arguments": {
                        "input_artifact_id": source_ids[index - 1],
                        "target_lufs": target,
                    },
                },
                None,
            )
        )
    return calls


def _cache_repeat_calls(
    source_ids: list[str],
) -> list[tuple[str, dict[str, Any], tuple[int, int] | None]]:
    return [
        (
            "cache-trim",
            {
                "tool_name": "trim_video",
                "tool_call_id": "cache-trim",
                "arguments": {
                    "input_artifact_id": source_ids[0],
                    "time_range": {"start_ms": 0, "end_ms": 500},
                },
            },
            None,
        ),
        (
            "cache-speed",
            {
                "tool_name": "change_speed",
                "tool_call_id": "cache-speed",
                "arguments": {
                    "input_artifact_id": source_ids[0],
                    "speed_factor": 0.5,
                },
            },
            None,
        ),
        (
            "cache-reframe",
            {
                "tool_name": "reframe_video",
                "tool_call_id": "cache-reframe",
                "arguments": {
                    "input_artifact_id": source_ids[0],
                    "width": 90,
                    "height": 160,
                    "fit": "crop",
                },
            },
            (90, 160),
        ),
        (
            "cache-subtitle",
            {
                "tool_name": "add_subtitles",
                "tool_call_id": "cache-subtitle",
                "arguments": {
                    "input_artifact_id": source_ids[0],
                    "cues": [
                        {
                            "cue_id": "cue-01",
                            "time_range": {"start_ms": 250, "end_ms": 1500},
                            "text": "M3A",
                        }
                    ],
                },
            },
            None,
        ),
    ]


def _failure_calls(
    source_ids: list[str],
    incompatible_id: str,
    corrupt_id: str,
) -> list[tuple[str, dict[str, Any], str, str]]:
    return [
        (
            "failure-unknown",
            {"tool_name": "unknown", "tool_call_id": "failure-unknown", "arguments": {}},
            "invalid",
            "unknown_tool",
        ),
        (
            "failure-malformed",
            {"tool_name": "trim_video", "tool_call_id": "failure-malformed", "arguments": {}},
            "invalid",
            "invalid_call",
        ),
        (
            "failure-negative",
            {
                "tool_name": "trim_video",
                "tool_call_id": "failure-negative",
                "arguments": {
                    "input_artifact_id": source_ids[0],
                    "time_range": {"start_ms": -1, "end_ms": 500},
                },
            },
            "invalid",
            "invalid_call",
        ),
        (
            "failure-order",
            {
                "tool_name": "trim_video",
                "tool_call_id": "failure-order",
                "arguments": {
                    "input_artifact_id": source_ids[0],
                    "time_range": {"start_ms": 500, "end_ms": 500},
                },
            },
            "invalid",
            "invalid_call",
        ),
        (
            "failure-outside",
            {
                "tool_name": "trim_video",
                "tool_call_id": "failure-outside",
                "arguments": {
                    "input_artifact_id": source_ids[0],
                    "time_range": {"start_ms": 1000, "end_ms": 3000},
                },
            },
            "invalid",
            "invalid_interval",
        ),
        (
            "failure-concat",
            {
                "tool_name": "concat_videos",
                "tool_call_id": "failure-concat",
                "arguments": {"input_artifact_ids": [source_ids[0], incompatible_id]},
            },
            "invalid",
            "incompatible_media",
        ),
        (
            "failure-missing",
            {
                "tool_name": "inspect_media",
                "tool_call_id": "failure-missing",
                "arguments": {"input_artifact_id": "missing-artifact"},
            },
            "invalid",
            "artifact_not_found",
        ),
        (
            "failure-corrupt",
            {
                "tool_name": "inspect_media",
                "tool_call_id": "failure-corrupt",
                "arguments": {"input_artifact_id": corrupt_id},
            },
            "error",
            "corrupt_media",
        ),
        (
            "failure-subtitle",
            {
                "tool_name": "add_subtitles",
                "tool_call_id": "failure-subtitle",
                "arguments": {
                    "input_artifact_id": source_ids[0],
                    "cues": [
                        {
                            "cue_id": "outside",
                            "time_range": {"start_ms": 1500, "end_ms": 2500},
                            "text": "outside",
                        }
                    ],
                },
            },
            "invalid",
            "invalid_subtitle",
        ),
        (
            "failure-traversal",
            {
                "tool_name": "trim_video",
                "tool_call_id": "failure-traversal",
                "arguments": {
                    "input_artifact_id": "../../etc/passwd",
                    "time_range": {"start_ms": 0, "end_ms": 500},
                },
            },
            "invalid",
            "invalid_call",
        ),
    ]


def _run_sequences(
    registry: ToolRegistry,
    context: ToolExecutionContext,
    source_ids: list[str],
) -> tuple[list[dict[str, object]], float]:
    sequences: list[dict[str, object]] = []
    sequence_b: list[dict[str, Any]] = [
        {
            "tool_name": "trim_video",
            "tool_call_id": "seq-b-trim-a",
            "arguments": {
                "input_artifact_id": source_ids[0],
                "time_range": {"start_ms": 0, "end_ms": 1000},
            },
        },
        {
            "tool_name": "trim_video",
            "tool_call_id": "seq-b-trim-b",
            "arguments": {
                "input_artifact_id": source_ids[1],
                "time_range": {"start_ms": 500, "end_ms": 1500},
            },
        },
    ]
    outputs: list[str] = []
    records: list[ToolExecutionRecord] = []
    for call in sequence_b:
        record = registry.execute(call, context)
        records.append(record)
        outputs.append(_output_id(record))
    concatenated = registry.execute(
        {
            "tool_name": "concat_videos",
            "tool_call_id": "seq-b-concat",
            "arguments": {"input_artifact_ids": outputs},
        },
        context,
    )
    records.append(concatenated)
    concatenated_id = _output_id(concatenated)
    subtitle = registry.execute(
        {
            "tool_name": "add_subtitles",
            "tool_call_id": "seq-b-subtitle",
            "arguments": {
                "input_artifact_id": concatenated_id,
                "cues": [
                    {
                        "cue_id": "seq-b-cue",
                        "time_range": {"start_ms": 250, "end_ms": 1500},
                        "text": "SEQUENCE B",
                    }
                ],
            },
        },
        context,
    )
    records.append(subtitle)
    validation = registry.execute(
        {
            "tool_name": "validate_media",
            "tool_call_id": "seq-b-validate",
            "arguments": {"input_artifact_id": _output_id(subtitle), "require_audio": True},
        },
        context,
    )
    records.append(validation)
    sequences.append(
        {
            "sequence_id": "sequence-b",
            "tools": [item.observation.tool_name for item in records],
            "success": all(item.observation.status == "success" for item in records),
            "output_artifact_id": _output_id(subtitle),
        }
    )

    records = []
    trim = registry.execute(
        {
            "tool_name": "trim_video",
            "tool_call_id": "seq-c-trim",
            "arguments": {
                "input_artifact_id": source_ids[2],
                "time_range": {"start_ms": 0, "end_ms": 1500},
            },
        },
        context,
    )
    records.append(trim)
    speed = registry.execute(
        {
            "tool_name": "change_speed",
            "tool_call_id": "seq-c-speed",
            "arguments": {"input_artifact_id": _output_id(trim), "speed_factor": 1.5},
        },
        context,
    )
    records.append(speed)
    reframe = registry.execute(
        {
            "tool_name": "reframe_video",
            "tool_call_id": "seq-c-reframe",
            "arguments": {
                "input_artifact_id": _output_id(speed),
                "width": 90,
                "height": 160,
                "fit": "crop",
            },
        },
        context,
    )
    records.append(reframe)
    validation = registry.execute(
        {
            "tool_name": "validate_media",
            "tool_call_id": "seq-c-validate",
            "arguments": {"input_artifact_id": _output_id(reframe), "require_audio": True},
        },
        context,
    )
    records.append(validation)
    sequences.append(
        {
            "sequence_id": "sequence-c",
            "tools": [item.observation.tool_name for item in records],
            "success": all(item.observation.status == "success" for item in records),
            "output_artifact_id": _output_id(reframe),
        }
    )
    return sequences, sum(bool(item["success"]) for item in sequences) / len(sequences)


def run_synthetic(root: Path, *, ffmpeg: str, ffprobe: str) -> int:
    started = time.perf_counter()
    fixtures_root = root / "fixtures"
    paths = [
        generate_media_fixture(fixtures_root / f"{color}.mp4", ffmpeg=ffmpeg, color=color)
        for color in ("red", "blue", "green", "yellow")
    ]
    incompatible = generate_media_fixture(
        fixtures_root / "incompatible.mp4",
        ffmpeg=ffmpeg,
        color="white",
        width=320,
        height=180,
    )
    corrupt = fixtures_root / "corrupt.mp4"
    corrupt.write_bytes(b"controlled corrupt input")
    registry = build_media_registry(root / "runtime", ffmpeg=ffmpeg, ffprobe=ffprobe)
    source_refs = [
        registry.artifact_store.import_file(path, media_type="video/mp4") for path in paths
    ]
    incompatible_ref = registry.artifact_store.import_file(incompatible, media_type="video/mp4")
    corrupt_ref = registry.artifact_store.import_file(corrupt, media_type="video/mp4")
    source_ids = [item.artifact_id for item in source_refs]
    allowed = [
        *source_ids,
        incompatible_ref.artifact_id,
        corrupt_ref.artifact_id,
        "missing-artifact",
    ]
    context = context_for(execution_id="m3a-synthetic", artifact_ids=allowed)
    cases: list[ToolCaseResult] = []
    for case_id, call, dimensions in _normal_calls(source_ids):
        result, _ = run_case(
            registry,
            context,
            case_id=case_id,
            suite="synthetic",
            call=call,
            expected_dimensions=dimensions,
        )
        cases.append(result)
    for case_id, call, dimensions in _cache_repeat_calls(source_ids):
        result, _ = run_case(
            registry,
            context,
            case_id=case_id,
            suite="synthetic",
            call=call,
            expected_dimensions=dimensions,
        )
        cases.append(result)
    for case_id, call, expected_status, error in _failure_calls(
        source_ids, incompatible_ref.artifact_id, corrupt_ref.artifact_id
    ):
        result, _ = run_case(
            registry,
            context,
            case_id=case_id,
            suite="failure_injection",
            call=call,
            expected_status=cast(Any, expected_status),
            expected_error=error,
        )
        cases.append(result)

    timeout_registry = build_media_registry(
        root / "timeout_runtime", ffmpeg=ffmpeg, ffprobe=ffprobe
    )
    timeout_ref = timeout_registry.artifact_store.import_file(paths[0], media_type="video/mp4")
    timeout_case, _ = run_case(
        timeout_registry,
        context_for(
            execution_id="m3a-timeout",
            artifact_ids=(timeout_ref.artifact_id,),
            timeout_ms=1,
        ),
        case_id="failure-timeout",
        suite="failure_injection",
        call={
            "tool_name": "trim_video",
            "tool_call_id": "failure-timeout",
            "arguments": {
                "input_artifact_id": timeout_ref.artifact_id,
                "time_range": {"start_ms": 0, "end_ms": 500},
            },
        },
        expected_status="timeout",
        expected_error="timeout",
    )
    cases.append(timeout_case)

    size_registry = build_media_registry(root / "size_runtime", ffmpeg=ffmpeg, ffprobe=ffprobe)
    size_ref = size_registry.artifact_store.import_file(paths[0], media_type="video/mp4")
    size_case, _ = run_case(
        size_registry,
        context_for(
            execution_id="m3a-size",
            artifact_ids=(size_ref.artifact_id,),
            maximum_output_bytes=32,
        ),
        case_id="failure-output-size",
        suite="failure_injection",
        call={
            "tool_name": "trim_video",
            "tool_call_id": "failure-output-size",
            "arguments": {
                "input_artifact_id": size_ref.artifact_id,
                "time_range": {"start_ms": 0, "end_ms": 500},
            },
        },
        expected_status="error",
        expected_error="output_too_large",
    )
    cases.append(size_case)

    injected_registry = ToolRegistry(
        artifact_store=ArtifactStore(root / "injected_runtime" / "store"),
        trace_recorder=ToolTraceRecorder(root / "injected_runtime" / "traces"),
    )
    injected_registry.register(InjectedOutputValidationFailureTool())
    injected_source = root / "injected_runtime" / "controlled.bin"
    injected_source.write_bytes(b"controlled validation-failure artifact")
    injected_ref = injected_registry.artifact_store.import_file(
        injected_source,
        media_type="video/mp4",
        artifact_id="injected-artifact",
    )
    injected_case, _ = run_case(
        injected_registry,
        context_for(
            execution_id="m3a-injected",
            artifact_ids=(injected_ref.artifact_id,),
        ),
        case_id="failure-output-validation",
        suite="failure_injection",
        call={
            "tool_name": "validate_media",
            "tool_call_id": "failure-output-validation",
            "arguments": {"input_artifact_id": "injected-artifact"},
        },
        expected_status="error",
        expected_error="output_validation_failed",
    )
    cases.append(injected_case)

    sequences, sequence_rate = _run_sequences(registry, context, source_ids)
    summary = summarize_cases(
        cases,
        suite="synthetic_and_failure",
        sequence_success_rate=sequence_rate,
    )
    output = {
        "summary": summary.model_dump(mode="json"),
        "cases": [item.model_dump(mode="json") for item in cases],
        "sequences": sequences,
        "tool_manifest": registry.manifest().model_dump(mode="json"),
        "wall_time_ms": round((time.perf_counter() - started) * 1000),
    }
    _write_json(root / "synthetic_evaluation.json", output)
    if len(cases) != 61:
        raise RuntimeError(f"expected 61 M3A cases, observed {len(cases)}")
    if not all(item.expectation_met and item.post_validation_passed for item in cases):
        raise RuntimeError("one or more M3A synthetic/failure cases failed")
    if summary.cache_hit_count < 4:
        raise RuntimeError("expected cache-repeat cases did not all hit")
    if sequence_rate != 1.0:
        raise RuntimeError("one or more deterministic tool sequences failed")
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def run_licensed(root: Path, *, ffmpeg: str, ffprobe: str, provenance: Path) -> int:
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    media_root = provenance.parent
    selected: list[dict[str, object]] = []
    for source in payload["sources"]:
        clips = source.get("clips", [])
        if clips:
            selected.append({"source": source, "clip": clips[0]})
        if len(selected) == 5:
            break
    if len(selected) < 3:
        raise RuntimeError("licensed provenance contains fewer than three usable source groups")
    registry = build_media_registry(root / "licensed_runtime", ffmpeg=ffmpeg, ffprobe=ffprobe)
    refs = []
    provenance_rows = []
    for item in selected:
        source = cast(dict[str, Any], item["source"])
        clip = cast(dict[str, Any], item["clip"])
        clip_path = media_root / cast(str, clip["relative_path"])
        if not clip_path.is_file():
            raise RuntimeError(f"licensed clip is unavailable: {clip_path}")
        reference = registry.artifact_store.import_file(clip_path, media_type="video/mp4")
        if reference.sha256 != clip["sha256"]:
            raise RuntimeError(f"licensed clip hash mismatch: {clip['clip_id']}")
        refs.append(reference)
        provenance_rows.append(
            {
                "source_id": source["source_id"],
                "clip_id": clip["clip_id"],
                "source_page": source["source_page"],
                "license": source["license"],
                "license_url": source["license_url"],
                "source_sha256": source["sha256"],
                "clip_sha256": clip["sha256"],
                "download_date_utc": source["download_date_utc"],
                "clip_extraction_command": clip["extraction_command"],
            }
        )
    context = context_for(
        execution_id="m3a-licensed-real",
        artifact_ids=[item.artifact_id for item in refs],
        maximum_output_bytes=2_000_000_000,
    )
    cases: list[ToolCaseResult] = []
    reframed_outputs: list[tuple[str, bool]] = []
    for index, reference in enumerate(refs, 1):
        operations: list[tuple[str, dict[str, Any], tuple[int, int] | None]] = [
            (
                f"real-trim-{index}",
                {
                    "tool_name": "trim_video",
                    "tool_call_id": f"real-trim-{index}",
                    "arguments": {
                        "input_artifact_id": reference.artifact_id,
                        "time_range": {"start_ms": 0, "end_ms": 1000},
                    },
                },
                None,
            ),
            (
                f"real-speed-{index}",
                {
                    "tool_name": "change_speed",
                    "tool_call_id": f"real-speed-{index}",
                    "arguments": {"input_artifact_id": reference.artifact_id, "speed_factor": 1.25},
                },
                None,
            ),
            (
                f"real-reframe-{index}",
                {
                    "tool_name": "reframe_video",
                    "tool_call_id": f"real-reframe-{index}",
                    "arguments": {
                        "input_artifact_id": reference.artifact_id,
                        "width": 160,
                        "height": 90,
                        "fit": "crop",
                    },
                },
                (160, 90),
            ),
            (
                f"real-subtitle-{index}",
                {
                    "tool_name": "add_subtitles",
                    "tool_call_id": f"real-subtitle-{index}",
                    "arguments": {
                        "input_artifact_id": reference.artifact_id,
                        "cues": [
                            {
                                "cue_id": f"real-cue-{index}",
                                "time_range": {"start_ms": 100, "end_ms": 900},
                                "text": "LICENSED VALIDATION",
                            }
                        ],
                    },
                },
                None,
            ),
        ]
        for case_id, call, dimensions in operations:
            result, record = run_case(
                registry,
                context,
                case_id=case_id,
                suite="licensed_real",
                call=call,
                expected_dimensions=dimensions,
            )
            cases.append(result)
            if call["tool_name"] == "reframe_video":
                reframed_outputs.append(
                    (
                        _output_id(record),
                        record.observation.details.get("has_audio") is True,
                    )
                )
    compatible_pair: tuple[str, str] | None = None
    for left_index, (left_id, left_audio) in enumerate(reframed_outputs):
        for right_id, right_audio in reframed_outputs[left_index + 1 :]:
            if left_audio == right_audio:
                compatible_pair = (left_id, right_id)
                break
        if compatible_pair is not None:
            break
    if compatible_pair is None:
        raise RuntimeError("licensed reframe outputs contain no concat-compatible pair")
    concat, _ = run_case(
        registry,
        context,
        case_id="real-concat",
        suite="licensed_real",
        call={
            "tool_name": "concat_videos",
            "tool_call_id": "real-concat",
            "arguments": {"input_artifact_ids": list(compatible_pair)},
        },
    )
    cases.append(concat)
    summary = summarize_cases(cases, suite="licensed_real")
    output = {
        "summary": summary.model_dump(mode="json"),
        "cases": [item.model_dump(mode="json") for item in cases],
        "licensed_provenance": provenance_rows,
    }
    _write_json(root / "licensed_real_evaluation.json", output)
    if not all(item.expectation_met and item.post_validation_passed for item in cases):
        raise RuntimeError("one or more licensed-real M3A cases failed")
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m3a"))
    parser.add_argument("--suite", choices=("synthetic", "licensed"), default="synthetic")
    parser.add_argument(
        "--licensed-provenance",
        type=Path,
        default=Path("artifacts/m1b_5/real_media/provenance.json"),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    ffmpeg = shutil.which(args.ffmpeg)
    ffprobe = shutil.which(args.ffprobe)
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("FFmpeg and ffprobe are required")
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.suite == "licensed":
        return run_licensed(
            root,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            provenance=args.licensed_provenance.resolve(),
        )
    return run_synthetic(root, ffmpeg=ffmpeg, ffprobe=ffprobe)


if __name__ == "__main__":
    raise SystemExit(main())
