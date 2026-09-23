"""Build an answer-free static review bundle for the frozen M5A human packet."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from cutagent.schemas.human_review import (
    HumanReviewCase,
    HumanReviewMediaMetadata,
    HumanReviewMediaRef,
    HumanReviewToolObservation,
    HumanReviewTrajectorySummary,
)

BUILDER_VERSION = "m10a-human-review-bundle-v1"
PACKET_VERSION = "m5a-human-calibration-v1"
PACKET_SHA256 = "a9757ef9b2b1f937d4154ab50f737a44162f8e87ab00a5a2b014aae6b8d6776e"
PACKET_CASE_COUNT = 50
RUBRIC_VERSION = "m5a-human-calibration-v1"

RATING_COLUMNS = (
    "rater_id",
    "rater_type",
    "packet_version",
    "packet_sha256",
    "case_id",
    "task_success",
    "semantic_constraint_score",
    "failure_type",
    "confidence",
    "notes",
)

FROZEN_FAILURE_LABELS = (
    "none",
    "uncertain",
    "upstream_perception_error",
    "retrieval_error",
    "planning_error",
    "wrong_tool",
    "invalid_arguments",
    "tool_execution_failure",
    "verification_error",
    "handoff_error",
    "recovery_error",
    "premature_finish",
    "premature_refusal",
    "loop_stagnation",
    "budget_exhaustion",
    "model_format_error",
)

_TEXT_SUFFIXES = frozenset({".json", ".md", ".html", ".csv", ".css", ".js"})
_FORBIDDEN_CONTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("benchmark_answer_object", re.compile(r"benchmarkgold", re.IGNORECASE)),
    ("dataset_partition_field", re.compile(r"\bsplit\b", re.IGNORECASE)),
    ("source_group_field", re.compile(r"source_group_id", re.IGNORECASE)),
    ("answer_field", re.compile(r"expected_answer", re.IGNORECASE)),
    ("scene_answer_field", re.compile(r"expected_scene", re.IGNORECASE)),
    ("time_answer_field", re.compile(r"expected_time", re.IGNORECASE)),
    ("terminal_answer_field", re.compile(r"expected_terminal", re.IGNORECASE)),
    ("failure_answer_field", re.compile(r"expected_failure", re.IGNORECASE)),
    ("evaluator_score", re.compile(r"(?:private_)?evaluator[_ -]?score", re.IGNORECASE)),
    ("protected_metric", re.compile(r"protected[_ -]?metrics?", re.IGNORECASE)),
    ("protected_result_name", re.compile(r"(?:locked_test|adversarial_test)", re.IGNORECASE)),
    (
        "private_sentinel",
        re.compile(r"(?:private_sentinel|evaluator_only|ground_truth)", re.IGNORECASE),
    ),
    ("file_uri", re.compile(r"file://", re.IGNORECASE)),
    ("unix_host_path", re.compile(r"/(?:home|root|Users)/")),
    ("windows_host_path", re.compile(r"[A-Za-z]:[\\/]")),
)
_HTML_HIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("html_comment", re.compile(r"<!--")),
    ("html_data_attribute", re.compile(r"\sdata-[\w-]+\s*=", re.IGNORECASE)),
    ("javascript_blob", re.compile(r"<script\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class ReviewBundleConfig:
    packet_path: Path
    rater_guide_path: Path
    rater_one_template_path: Path
    rater_two_template_path: Path
    public_task_paths: tuple[Path, ...]
    trajectory_directories: tuple[Path, ...]
    output_root: Path
    bundle_hash_output: Path
    rateability_output: Path
    leakage_output: Path
    ffprobe: str = "ffprobe"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _write_json(path: Path, value: object) -> None:
    _write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_object(value: Any, *, source: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return value


def _require_object_array(value: Any, *, source: Path) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{source} must contain a JSON object array")
    return list(value)


def _file_uri_path(uri: object) -> Path:
    if not isinstance(uri, str):
        raise ValueError("artifact URI is missing")
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError("only local file artifacts may be materialized")
    raw_path = unquote(parsed.path)
    if os.name == "nt" and re.match(r"^/[A-Za-z]:/", raw_path):
        raw_path = raw_path[1:]
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError("artifact URI must resolve to an absolute internal path")
    return path.resolve(strict=True)


def probe_media(path: Path, ffprobe: str) -> HumanReviewMediaMetadata:
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = _require_object(json.loads(result.stdout), source=path)
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ValueError(f"ffprobe returned no stream list for {path}")
    video_streams = [
        item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"
    ]
    audio_streams = [
        item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"
    ]
    format_payload = payload.get("format")
    duration_text = format_payload.get("duration") if isinstance(format_payload, dict) else None
    if not isinstance(duration_text, str):
        raise ValueError(f"ffprobe returned no duration for {path}")
    duration_ms = int(
        (Decimal(duration_text) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    first_video = video_streams[0] if video_streams else None
    width = first_video.get("width") if first_video else None
    height = first_video.get("height") if first_video else None
    return HumanReviewMediaMetadata(
        duration_ms=duration_ms,
        has_video=bool(video_streams),
        has_audio=bool(audio_streams),
        width=width if isinstance(width, int) else None,
        height=height if isinstance(height, int) else None,
    )


def _load_packet(path: Path) -> tuple[str, ...]:
    payload = _require_object(_read_json(path), source=path)
    if sha256_file(path) != PACKET_SHA256:
        raise ValueError("frozen packet SHA-256 changed")
    if payload.get("packet_version") != PACKET_VERSION:
        raise ValueError("frozen packet version changed")
    case_ids = payload.get("case_ids")
    if not isinstance(case_ids, list) or not all(isinstance(item, str) for item in case_ids):
        raise ValueError("frozen packet case_ids are invalid")
    if len(case_ids) != PACKET_CASE_COUNT or len(case_ids) != len(set(case_ids)):
        raise ValueError("frozen packet must contain exactly 50 unique cases")
    return tuple(case_ids)


def _validate_rater_guide(path: Path) -> None:
    guide = path.read_text(encoding="utf-8")
    required_fragments = (
        f"Packet version: `{PACKET_VERSION}`",
        f"Packet SHA-256: `{PACKET_SHA256}`",
        f"Required cases: {PACKET_CASE_COUNT}",
        "`task_success`",
        "`semantic_constraint_score`",
        "`failure_type`",
        "`confidence`",
        "`notes`",
    )
    missing = [fragment for fragment in required_fragments if fragment not in guide]
    if missing:
        raise ValueError(f"frozen rater guide is missing required fragments: {missing}")


def _load_public_tasks(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in paths:
        for task in _require_object_array(_read_json(path), source=path):
            task_id = task.get("task_id")
            if not isinstance(task_id, str):
                raise ValueError(f"public task without task_id in {path}")
            if task_id in tasks:
                raise ValueError(f"duplicate public task {task_id}")
            tasks[task_id] = task
    return tasks


def _load_trajectories(paths: tuple[Path, ...]) -> dict[str, tuple[dict[str, Any], Path]]:
    trajectories: dict[str, tuple[dict[str, Any], Path]] = {}
    for directory in paths:
        for path in sorted(directory.glob("*.json")):
            trajectory = _require_object(_read_json(path), source=path)
            task_input = trajectory.get("task_input")
            task_id = task_input.get("task_id") if isinstance(task_input, dict) else None
            if not isinstance(task_id, str):
                raise ValueError(f"trajectory without public task ID: {path}")
            if task_id in trajectories:
                raise ValueError(f"duplicate public trajectory {task_id}")
            trajectories[task_id] = (trajectory, path)
    return trajectories


def _tool_summary(trajectory: dict[str, Any]) -> HumanReviewTrajectorySummary:
    tool_records = trajectory.get("tool_records")
    if not isinstance(tool_records, list):
        tool_records = []
    observations: list[HumanReviewToolObservation] = []
    for sequence, record in enumerate(tool_records, start=1):
        if not isinstance(record, dict):
            continue
        observation = record.get("observation")
        if not isinstance(observation, dict):
            continue
        tool_name = observation.get("tool_name")
        status = observation.get("status")
        if not isinstance(tool_name, str) or status not in {
            "success",
            "error",
            "invalid",
            "timeout",
        }:
            continue
        public_summary = observation.get("public_summary")
        if not isinstance(public_summary, str) or not public_summary.strip():
            public_summary = f"tool returned status {status}"
        error_code = observation.get("error_code")
        if (
            not isinstance(error_code, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", error_code) is None
        ):
            error_code = None
        observations.append(
            HumanReviewToolObservation(
                sequence=sequence,
                tool_name=tool_name,
                status=status,
                public_summary=public_summary,
                error_code=error_code,
            )
        )
    statuses: list[str] = []
    events = trajectory.get("events")
    if isinstance(events, list):
        for envelope in events:
            event = envelope.get("event") if isinstance(envelope, dict) else None
            if isinstance(event, dict) and event.get("event_type") == "verification_result":
                value = event.get("status")
                if isinstance(value, str) and value:
                    statuses.append(value)
    policy_failures = trajectory.get("policy_failures")
    failure_count = len(policy_failures) if isinstance(policy_failures, list) else 0
    return HumanReviewTrajectorySummary(
        tool_observations=tuple(observations),
        verification_statuses=tuple(statuses),
        structured_decision_failure_count=failure_count,
        final_output_present=trajectory.get("final_output_artifact") is not None,
    )


def _assert_task_binding(task: dict[str, Any], trajectory: dict[str, Any]) -> None:
    trajectory_task = trajectory.get("task_input")
    if not isinstance(trajectory_task, dict):
        raise ValueError("trajectory has no public task input")
    for field in ("task_id", "instruction"):
        if trajectory_task.get(field) != task.get(field):
            raise ValueError(f"trajectory/public task mismatch for {field}")
    task_video = task.get("video_ref")
    trajectory_video = trajectory_task.get("video_ref")
    if not isinstance(task_video, dict) or not isinstance(trajectory_video, dict):
        raise ValueError("public task video reference is missing")
    for field in ("artifact_id", "sha256"):
        if trajectory_video.get(field) != task_video.get(field):
            raise ValueError(f"trajectory/public video mismatch for {field}")


def _instructions_markdown() -> str:
    labels = "\n".join(f"  - `{label}`" for label in FROZEN_FAILURE_LABELS)
    return f"""# CutAgentBench v0.1 Human Review Instructions

Packet version: `{PACKET_VERSION}`

Packet SHA-256: `{PACKET_SHA256}`

Cases: {PACKET_CASE_COUNT}

Work independently. Use only the instruction, playable media, public tool outcomes,
and terminal behavior shown for each case. Do not inspect another rater's CSV or any
external answer key. This bundle does not identify the policy that produced the trajectory.

Fill the unchanged CSV columns:

- `rater_id`: an identifier for the actual human reviewer; use the same value in all rows.
- `rater_type`: keep `human` unchanged.
- `packet_version`, `packet_sha256`, `case_id`: keep unchanged.
- `task_success`: `success`, `failure`, or `uncertain` based only on the presented case.
- `semantic_constraint_score`: integer 0-4; 0 means no semantic requirement is met
  and 4 means all presented semantic requirements are met.
- `failure_type`: select one frozen label:
{labels}
- `confidence`: integer 1-5.
- `notes`: concise evidence or uncertainty; do not include filesystem locations or secrets.

Case 001 maps to the first CSV data row, Case 002 to the second, and so on through
Case 050. Do not reorder rows and do not pre-fill or suggest another rater's judgments.
"""


def _case_page(case: HumanReviewCase, *, previous_page: str | None, next_page: str | None) -> str:
    source = html.escape(case.source_media_ref.relative_path, quote=True)
    output_html = "<p><strong>No final candidate output was produced.</strong></p>"
    if case.candidate_output_ref is not None:
        output = html.escape(case.candidate_output_ref.relative_path, quote=True)
        output_html = f'<video controls preload="metadata" src="{output}"></video>'
    tools = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            item.sequence,
            html.escape(item.tool_name),
            html.escape(item.status),
            html.escape(item.public_summary),
            html.escape(item.error_code or "none"),
        )
        for item in case.public_trajectory_summary.tool_observations
    )
    if not tools:
        tools = '<tr><td colspan="5">No valid tool execution was recorded.</td></tr>'
    verifications = (
        ", ".join(case.public_trajectory_summary.verification_statuses) or "none recorded"
    )
    navigation = ['<a href="../../index.html">Bundle index</a>']
    if previous_page is not None:
        navigation.append(f'<a href="../{previous_page}/page.html">Previous</a>')
    if next_page is not None:
        navigation.append(f'<a href="../{next_page}/page.html">Next</a>')
    source_meta = case.source_media_metadata
    candidate_meta = case.candidate_output_metadata
    candidate_meta_text = "none"
    if candidate_meta is not None:
        candidate_meta_text = (
            f"{candidate_meta.duration_ms} ms, {candidate_meta.width}x{candidate_meta.height}, "
            f"audio={'yes' if candidate_meta.has_audio else 'no'}"
        )
    source_audio = "yes" if source_meta.has_audio else "no"
    navigation_html = " ".join(navigation)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Case {case.display_index:03d}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;
padding:0 1rem;line-height:1.5}}
video{{max-width:100%;max-height:480px;background:#111}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #bbb;padding:.45rem;text-align:left;vertical-align:top}}
nav{{display:flex;gap:1rem;margin:1rem 0}}
.panel{{border:1px solid #ccc;border-radius:.5rem;padding:1rem;margin:1rem 0}}
code{{word-break:break-all}}
</style>
</head>
<body>
<nav>{navigation_html}</nav>
<h1>Case {case.display_index:03d}</h1>
<p><code>{html.escape(case.case_id)}</code></p>
<section class="panel"><h2>User instruction</h2>
<p>{html.escape(case.instruction)}</p></section>
<section class="panel"><h2>Source media</h2>
<video controls preload="metadata" src="{source}"></video>
<p>{source_meta.duration_ms} ms, {source_meta.width}x{source_meta.height},
audio={source_audio}</p></section>
<section class="panel"><h2>Final candidate output</h2>
{output_html}<p>{candidate_meta_text}</p></section>
<section class="panel"><h2>Public trajectory outcome</h2>
<p>Terminal behavior: <strong>{html.escape(case.terminal_behavior)}</strong></p>
<p>Structured decision failures observed:
{case.public_trajectory_summary.structured_decision_failure_count}</p>
<p>Online verification statuses: {html.escape(verifications)}</p>
<table><thead><tr><th>#</th><th>Tool</th><th>Status</th>
<th>Public observation</th><th>Error code</th></tr></thead>
<tbody>{tools}</tbody></table></section>
<section class="panel"><h2>What to record</h2>
<p>Use the unchanged row for this case in your CSV. Judge task success,
semantic constraint score, failure type, confidence, and concise notes from
the displayed material only.</p></section>
<nav>{navigation_html}</nav>
</body>
</html>
"""


def _index_page(cases: tuple[HumanReviewCase, ...]) -> str:
    rows = "".join(
        (
            f"<tr><td>{case.display_index:03d}</td>"
            f"<td><code>{html.escape(case.case_id)}</code></td>"
            f"<td>{html.escape(case.instruction)}</td>"
            f"<td>{html.escape(case.terminal_behavior)}</td>"
            f'<td><a href="cases/case_{case.display_index:03d}/page.html">'
            "Open</a></td></tr>"
        )
        for case in cases
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CutAgentBench Human Review</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;
padding:0 1rem;line-height:1.45}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #bbb;padding:.45rem;text-align:left;vertical-align:top}}
code{{word-break:break-all}}
</style>
</head>
<body>
<h1>CutAgentBench v0.1 Human Review</h1>
<p>Review all {len(cases)} cases independently. Read
<a href="RATER_INSTRUCTIONS.md">the instructions</a>, then fill one unchanged
CSV template from <code>downloads/</code>.</p>
<table><thead><tr><th>Case</th><th>Case ID</th><th>Instruction</th>
<th>Terminal behavior</th><th>Review</th></tr></thead><tbody>{rows}</tbody></table>
</body>
</html>
"""


def bundle_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_digest = bytes.fromhex(sha256_file(path))
        digest.update(file_digest)
    return digest.hexdigest()


def audit_bundle_leakage(root: Path) -> dict[str, object]:
    findings: list[dict[str, str]] = []
    files = sorted(item for item in root.rglob("*") if item.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix()
        for name, pattern in _FORBIDDEN_CONTENT_PATTERNS:
            if pattern.search(relative):
                findings.append({"category": name, "path": relative, "location": "filename"})
        if path.suffix.casefold() not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig")
        for name, pattern in _FORBIDDEN_CONTENT_PATTERNS:
            if pattern.search(text):
                findings.append({"category": name, "path": relative, "location": "content"})
        if path.suffix.casefold() == ".html":
            for name, pattern in _HTML_HIDDEN_PATTERNS:
                if pattern.search(text):
                    findings.append({"category": name, "path": relative, "location": "html"})
    return {
        "schema_version": "1.0",
        "audit_version": "m10a-review-bundle-leakage-v1",
        "status": "passed" if not findings else "failed",
        "scanned_file_count": len(files),
        "leak_count": len(findings),
        "findings": findings,
        "checks": [name for name, _ in _FORBIDDEN_CONTENT_PATTERNS]
        + [name for name, _ in _HTML_HIDDEN_PATTERNS],
    }


def _copy_verified(source: Path, destination: Path, declared_hash: object) -> str:
    observed = sha256_file(source)
    if not isinstance(declared_hash, str) or observed != declared_hash:
        raise ValueError(f"artifact digest mismatch for {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != observed:
        raise ValueError("review copy digest mismatch")
    return observed


def build_review_bundle(
    config: ReviewBundleConfig,
    *,
    media_probe: Callable[[Path, str], HumanReviewMediaMetadata] = probe_media,
) -> dict[str, object]:
    case_ids = _load_packet(config.packet_path)
    _validate_rater_guide(config.rater_guide_path)
    if config.output_root.exists():
        raise FileExistsError(f"review bundle output already exists: {config.output_root}")
    if sha256_file(config.rater_one_template_path) != sha256_file(config.rater_two_template_path):
        raise ValueError("frozen blank rater templates are not byte equivalent")
    first_header = config.rater_one_template_path.read_text(encoding="utf-8-sig").splitlines()[0]
    if tuple(first_header.split(",")) != RATING_COLUMNS:
        raise ValueError("frozen rater template columns changed")
    tasks = _load_public_tasks(config.public_task_paths)
    trajectories = _load_trajectories(config.trajectory_directories)
    config.output_root.mkdir(parents=True)
    downloads = config.output_root / "downloads"
    downloads.mkdir()
    shutil.copyfile(config.rater_one_template_path, downloads / "rater_1.csv")
    shutil.copyfile(config.rater_two_template_path, downloads / "rater_2.csv")
    _write_text(config.output_root / "RATER_INSTRUCTIONS.md", _instructions_markdown())

    rateability: list[dict[str, object]] = []
    cases: list[HumanReviewCase] = []
    manifest_cases: list[dict[str, object]] = []
    for index, case_id in enumerate(case_ids, start=1):
        missing: list[str] = []
        task = tasks.get(case_id)
        trajectory_record = trajectories.get(case_id)
        if task is None:
            missing.append("public TaskInput")
        if trajectory_record is None:
            missing.append("public frozen trajectory")
        if missing:
            rateability.append(
                {
                    "case_id": case_id,
                    "display_index": index,
                    "status": "NOT_RATEABLE_FROM_FROZEN_PUBLIC_MATERIAL",
                    "missing": missing,
                }
            )
            continue
        assert task is not None
        assert trajectory_record is not None
        trajectory, trajectory_path = trajectory_record
        try:
            _assert_task_binding(task, trajectory)
            instruction = task.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("public instruction is missing")
            video_ref = task.get("video_ref")
            if not isinstance(video_ref, dict):
                raise ValueError("public source media reference is missing")
            source_path = _file_uri_path(video_ref.get("uri"))
            case_directory = config.output_root / "cases" / f"case_{index:03d}"
            source_destination = case_directory / "source.mp4"
            source_hash = _copy_verified(source_path, source_destination, video_ref.get("sha256"))
            source_metadata = media_probe(source_destination, config.ffprobe)
            final_artifact = trajectory.get("final_output_artifact")
            candidate_ref: HumanReviewMediaRef | None = None
            candidate_metadata: HumanReviewMediaMetadata | None = None
            presentation: list[dict[str, object]] = [
                {
                    "role": "source",
                    "relative_path": f"cases/case_{index:03d}/source.mp4",
                    "sha256": source_hash,
                    "size_bytes": source_destination.stat().st_size,
                }
            ]
            if final_artifact is not None:
                if not isinstance(final_artifact, dict):
                    raise ValueError("final output reference is invalid")
                output_path = _file_uri_path(final_artifact.get("uri"))
                output_destination = case_directory / "output.mp4"
                output_hash = _copy_verified(
                    output_path, output_destination, final_artifact.get("sha256")
                )
                candidate_ref = HumanReviewMediaRef(
                    role="candidate_output", relative_path="output.mp4", media_type="video/mp4"
                )
                candidate_metadata = media_probe(output_destination, config.ffprobe)
                presentation.append(
                    {
                        "role": "candidate_output",
                        "relative_path": f"cases/case_{index:03d}/output.mp4",
                        "sha256": output_hash,
                        "size_bytes": output_destination.stat().st_size,
                    }
                )
            terminal_behavior = trajectory.get("terminal_reason")
            if not isinstance(terminal_behavior, str) or not terminal_behavior:
                raise ValueError("public terminal behavior is missing")
            if terminal_behavior == "SUCCESS" and candidate_ref is None:
                raise ValueError("successful trajectory has no final output presentation")
            case = HumanReviewCase(
                case_id=case_id,
                display_index=index,
                instruction=instruction,
                source_media_ref=HumanReviewMediaRef(
                    role="source", relative_path="source.mp4", media_type="video/mp4"
                ),
                candidate_output_ref=candidate_ref,
                source_media_metadata=source_metadata,
                candidate_output_metadata=candidate_metadata,
                public_trajectory_summary=_tool_summary(trajectory),
                terminal_behavior=terminal_behavior,
                review_instructions=(
                    "Judge only from the user instruction and displayed public evidence.",
                    "Use the unchanged CSV row with the same case ID.",
                ),
            )
            cases.append(case)
            case_json_path = case_directory / "case.json"
            _write_json(case_json_path, case.model_dump(mode="json"))
            previous_page = f"case_{index - 1:03d}" if index > 1 else None
            next_page = f"case_{index + 1:03d}" if index < PACKET_CASE_COUNT else None
            page_path = case_directory / "page.html"
            _write_text(
                page_path, _case_page(case, previous_page=previous_page, next_page=next_page)
            )
            manifest_cases.append(
                {
                    "case_id": case_id,
                    "display_index": index,
                    "csv_data_row": index,
                    "packet_case_mapping": "exact_case_id_and_original_order",
                    "rubric_version": RUBRIC_VERSION,
                    "trajectory_sha256": sha256_file(trajectory_path),
                    "presentation_artifacts": presentation,
                    "case_json": {
                        "relative_path": f"cases/case_{index:03d}/case.json",
                        "sha256": sha256_file(case_json_path),
                    },
                    "page": {
                        "relative_path": f"cases/case_{index:03d}/page.html",
                        "sha256": sha256_file(page_path),
                    },
                }
            )
            rateability.append(
                {"case_id": case_id, "display_index": index, "status": "RATEABLE", "missing": []}
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            rateability.append(
                {
                    "case_id": case_id,
                    "display_index": index,
                    "status": "NOT_RATEABLE_FROM_FROZEN_PUBLIC_MATERIAL",
                    "missing": [str(exc)],
                }
            )

    case_tuple = tuple(cases)
    _write_text(config.output_root / "index.html", _index_page(case_tuple))
    manifest = {
        "schema_version": "1.0",
        "builder_version": BUILDER_VERSION,
        "packet_version": PACKET_VERSION,
        "packet_sha256": PACKET_SHA256,
        "packet_case_count": PACKET_CASE_COUNT,
        "rubric_version": RUBRIC_VERSION,
        "rating_columns": list(RATING_COLUMNS),
        "case_count": len(cases),
        "case_order_preserved": [item.case_id for item in cases] == list(case_ids),
        "cases": manifest_cases,
        "template_copies": [
            {
                "relative_path": "downloads/rater_1.csv",
                "sha256": sha256_file(downloads / "rater_1.csv"),
            },
            {
                "relative_path": "downloads/rater_2.csv",
                "sha256": sha256_file(downloads / "rater_2.csv"),
            },
        ],
        "answer_labels_included": False,
    }
    _write_json(config.output_root / "review_manifest.json", manifest)
    rateable_count = sum(item["status"] == "RATEABLE" for item in rateability)
    rateability_payload = {
        "schema_version": "1.0",
        "audit_version": "m10a-rateability-v1",
        "status": "passed"
        if rateable_count == PACKET_CASE_COUNT
        else "HUMAN_PROTOCOL_REVISION_REQUIRED",
        "packet_case_count": PACKET_CASE_COUNT,
        "rateable_case_count": rateable_count,
        "not_rateable_case_count": PACKET_CASE_COUNT - rateable_count,
        "cases": rateability,
    }
    _write_json(config.rateability_output, rateability_payload)
    leakage_payload = audit_bundle_leakage(config.output_root)
    _write_json(config.leakage_output, leakage_payload)
    bundle_hash = bundle_tree_sha256(config.output_root)
    _write_text(config.bundle_hash_output, f"{bundle_hash}  review_bundle\n")
    return {
        "schema_version": "1.0",
        "status": "passed"
        if rateable_count == PACKET_CASE_COUNT and leakage_payload["leak_count"] == 0
        else "failed",
        "rateable_case_count": rateable_count,
        "leak_count": leakage_payload["leak_count"],
        "bundle_sha256": bundle_hash,
        "review_manifest_sha256": sha256_file(config.output_root / "review_manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--rater-guide", type=Path, required=True)
    parser.add_argument("--rater-1-template", type=Path, required=True)
    parser.add_argument("--rater-2-template", type=Path, required=True)
    parser.add_argument("--public-tasks", type=Path, action="append", required=True)
    parser.add_argument("--trajectory-directory", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle-hash-output", type=Path, required=True)
    parser.add_argument("--rateability-output", type=Path, required=True)
    parser.add_argument("--leakage-output", type=Path, required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    config = ReviewBundleConfig(
        packet_path=args.packet,
        rater_guide_path=args.rater_guide,
        rater_one_template_path=args.rater_1_template,
        rater_two_template_path=args.rater_2_template,
        public_task_paths=tuple(args.public_tasks),
        trajectory_directories=tuple(args.trajectory_directory),
        output_root=args.output_root,
        bundle_hash_output=args.bundle_hash_output,
        rateability_output=args.rateability_output,
        leakage_output=args.leakage_output,
        ffprobe=args.ffprobe,
    )
    result = build_review_bundle(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["rateable_case_count"] != PACKET_CASE_COUNT:
        return 2
    if result["leak_count"] != 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
