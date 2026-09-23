"""Offline-only validation helpers for the CutAgentBench human calibration gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

RATING_FIELDS: Final[tuple[str, ...]] = (
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
TASK_SUCCESS_LABELS: Final[frozenset[str]] = frozenset({"success", "failure", "uncertain"})
FAILURE_LABELS: Final[frozenset[str]] = frozenset(
    {
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
    }
)
_MODEL_IDENTITY_MARKERS: Final[tuple[str, ...]] = (
    "chatgpt",
    "gpt-",
    "claude",
    "gemini",
    "qwen",
    "llm",
    "language model",
    "codex",
)


@dataclass(frozen=True)
class PacketInfo:
    version: str
    sha256: str
    case_ids: tuple[str, ...]


@dataclass(frozen=True)
class RaterSubmission:
    path: Path
    sha256: str
    rater_id: str
    rows: tuple[dict[str, str], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_packet(path: Path) -> PacketInfo:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("calibration packet must be a JSON object")
    version = payload.get("packet_version")
    case_ids = payload.get("case_ids")
    required = payload.get("required_independent_raters")
    if not isinstance(version, str) or not version:
        raise ValueError("packet_version is missing")
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or not all(isinstance(item, str) and item for item in case_ids)
    ):
        raise ValueError("case_ids must be a non-empty string list")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("packet contains duplicate case IDs")
    if required != 2:
        raise ValueError("the frozen gate must require exactly two independent raters")
    return PacketInfo(
        version=version, sha256=hashlib.sha256(raw).hexdigest(), case_ids=tuple(case_ids)
    )


def template_rows(packet: PacketInfo) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "rater_id": "",
            "rater_type": "human",
            "packet_version": packet.version,
            "packet_sha256": packet.sha256,
            "case_id": case_id,
            "task_success": "",
            "semantic_constraint_score": "",
            "failure_type": "",
            "confidence": "",
            "notes": "",
        }
        for case_id in packet.case_ids
    )


def write_csv(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RATING_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_identity(value: str) -> None:
    normalized = value.strip().casefold()
    if len(normalized) < 2:
        raise ValueError("rater_id must identify a real person")
    if any(marker in normalized for marker in _MODEL_IDENTITY_MARKERS):
        raise ValueError("LLM/model identities cannot satisfy the human calibration gate")


def load_submission(path: Path, packet: PacketInfo) -> RaterSubmission:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RATING_FIELDS:
            raise ValueError(f"invalid rating columns in {path}")
        rows = tuple(dict(row) for row in reader)
    if len(rows) != len(packet.case_ids):
        raise ValueError(f"submission {path} is incomplete")
    rater_ids = {row["rater_id"].strip() for row in rows}
    if len(rater_ids) != 1:
        raise ValueError(f"submission {path} must contain one consistent rater_id")
    rater_id = next(iter(rater_ids))
    _validate_identity(rater_id)
    observed_ids = tuple(row["case_id"].strip() for row in rows)
    if observed_ids != packet.case_ids:
        raise ValueError(f"submission {path} case ordering/content differs from frozen packet")
    for row in rows:
        if row["rater_type"].strip().casefold() != "human":
            raise ValueError("rater_type must be human")
        if row["packet_version"].strip() != packet.version:
            raise ValueError("packet version mismatch")
        if row["packet_sha256"].strip().casefold() != packet.sha256:
            raise ValueError("packet hash mismatch")
        if row["task_success"].strip().casefold() not in TASK_SUCCESS_LABELS:
            raise ValueError(f"invalid task_success for {row['case_id']}")
        score_text = row["semantic_constraint_score"].strip()
        if score_text not in {"0", "1", "2", "3", "4"}:
            raise ValueError(f"invalid semantic_constraint_score for {row['case_id']}")
        if row["failure_type"].strip().casefold() not in FAILURE_LABELS:
            raise ValueError(f"invalid failure_type for {row['case_id']}")
        confidence = row["confidence"].strip()
        if confidence not in {"1", "2", "3", "4", "5"}:
            raise ValueError(f"invalid confidence for {row['case_id']}")
    return RaterSubmission(
        path=path.resolve(),
        sha256=sha256_file(path),
        rater_id=rater_id,
        rows=rows,
    )


def validate_pair(
    packet_path: Path,
    rater_one_path: Path,
    rater_two_path: Path,
) -> tuple[PacketInfo, RaterSubmission, RaterSubmission]:
    packet = load_packet(packet_path)
    first = load_submission(rater_one_path, packet)
    second = load_submission(rater_two_path, packet)
    if first.rater_id.strip().casefold() == second.rater_id.strip().casefold():
        raise ValueError("two distinct real rater identities are required")
    if first.sha256 == second.sha256:
        raise ValueError("duplicate rating files cannot count as independent submissions")
    return packet, first, second


def _cohen_kappa(left: tuple[str, ...], right: tuple[str, ...]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    labels = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right)) for label in labels
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def agreement_payload(
    packet: PacketInfo,
    first: RaterSubmission,
    second: RaterSubmission,
) -> dict[str, object]:
    first_success = tuple(row["task_success"].strip().casefold() for row in first.rows)
    second_success = tuple(row["task_success"].strip().casefold() for row in second.rows)
    first_score = tuple(row["semantic_constraint_score"].strip() for row in first.rows)
    second_score = tuple(row["semantic_constraint_score"].strip() for row in second.rows)
    first_failure = tuple(row["failure_type"].strip().casefold() for row in first.rows)
    second_failure = tuple(row["failure_type"].strip().casefold() for row in second.rows)
    disagreements = tuple(
        case_id
        for case_id, row_one, row_two in zip(packet.case_ids, first.rows, second.rows, strict=True)
        if any(
            row_one[field].strip().casefold() != row_two[field].strip().casefold()
            for field in ("task_success", "semantic_constraint_score", "failure_type")
        )
    )
    size = len(packet.case_ids)
    return {
        "schema_version": "1.0",
        "status": "validated_real_human_pair",
        "packet_version": packet.version,
        "packet_sha256": packet.sha256,
        "case_count": size,
        "rater_ids": [first.rater_id, second.rater_id],
        "rater_file_sha256": [first.sha256, second.sha256],
        "task_success_percent_agreement": sum(
            a == b for a, b in zip(first_success, second_success, strict=True)
        )
        / size,
        "task_success_cohen_kappa": _cohen_kappa(first_success, second_success),
        "semantic_score_exact_agreement": sum(
            a == b for a, b in zip(first_score, second_score, strict=True)
        )
        / size,
        "failure_type_percent_agreement": sum(
            a == b for a, b in zip(first_failure, second_failure, strict=True)
        )
        / size,
        "disagreement_case_ids": list(disagreements),
        "disagreement_count": len(disagreements),
        "protected_agent_results_used": False,
    }
