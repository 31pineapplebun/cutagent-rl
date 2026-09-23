from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cutagent.schemas.human_review import (
    HumanReviewCase,
    HumanReviewMediaMetadata,
    HumanReviewMediaRef,
)
from scripts.m10_build_human_review_bundle import (
    PACKET_CASE_COUNT,
    PACKET_SHA256,
    RATING_COLUMNS,
    ReviewBundleConfig,
    audit_bundle_leakage,
    build_review_bundle,
    bundle_tree_sha256,
    sha256_file,
)

REPOSITORY = Path(__file__).resolve().parents[2]
PACKET = REPOSITORY / "artifacts/m5a/cutagentbench_v0.1/calibration/human_calibration_packet.json"
GUIDE = REPOSITORY / "artifacts/m5a/human_calibration/RATER_GUIDE.md"
RATER_ONE = REPOSITORY / "artifacts/m5a/human_calibration/rater_1_template.csv"
RATER_TWO = REPOSITORY / "artifacts/m5a/human_calibration/rater_2_template.csv"
MANIFEST_MARKER = REPOSITORY / "artifacts/m5a/cutagentbench_v0.1/benchmark_manifest.sha256"


def _require_private_calibration_files() -> None:
    if not all(path.is_file() for path in (PACKET, GUIDE, RATER_ONE, RATER_TWO, MANIFEST_MARKER)):
        pytest.skip("requires private calibration artifacts excluded from the public snapshot")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fake_probe(_path: Path, _ffprobe: str) -> HumanReviewMediaMetadata:
    return HumanReviewMediaMetadata(
        duration_ms=15000,
        has_video=True,
        has_audio=True,
        width=384,
        height=256,
    )


def _write_fixture_inputs(root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    media = root / "public-media.mp4"
    media.write_bytes(b"public observable media fixture")
    media_hash = sha256_file(media)
    tasks: list[dict[str, object]] = []
    trajectory_directory = root / "trajectories"
    trajectory_directory.mkdir()
    for index, case_id in enumerate(packet["case_ids"], start=1):
        task = {
            "schema_version": "1.0",
            "task_id": case_id,
            "instruction": f"Review public task {index:03d}.",
            "user_constraints": [],
            "requested_output": {"schema_version": "1.0", "container": "mp4"},
            "video_ref": {
                "schema_version": "1.0",
                "artifact_id": "fixture-video",
                "uri": media.as_uri(),
                "sha256": media_hash,
                "size_bytes": media.stat().st_size,
                "media_type": "video/mp4",
            },
        }
        tasks.append(task)
        trajectory = {
            "schema_version": "1.0",
            "task_input": task,
            "terminal_reason": "MODEL_OUTPUT_FAILURE",
            "final_output_artifact": None,
            "tool_records": [],
            "events": [],
            "policy_failures": [{"operation": "decision"}],
        }
        (trajectory_directory / f"{case_id}.json").write_text(
            json.dumps(trajectory, sort_keys=True), encoding="utf-8"
        )
    first = root / "public-one.json"
    second = root / "public-two.json"
    first.write_text(json.dumps(tasks[:25], sort_keys=True), encoding="utf-8")
    second.write_text(json.dumps(tasks[25:], sort_keys=True), encoding="utf-8")
    return (first, second), (trajectory_directory,)


def _config(
    root: Path,
    public_tasks: tuple[Path, ...],
    trajectories: tuple[Path, ...],
    name: str,
) -> ReviewBundleConfig:
    return ReviewBundleConfig(
        packet_path=PACKET,
        rater_guide_path=GUIDE,
        rater_one_template_path=RATER_ONE,
        rater_two_template_path=RATER_TWO,
        public_task_paths=public_tasks,
        trajectory_directories=trajectories,
        output_root=root / name / "review_bundle",
        bundle_hash_output=root / name / "review_bundle.sha256",
        rateability_output=root / name / "rateability.json",
        leakage_output=root / name / "leakage.json",
    )


def test_frozen_packet_and_template_identity() -> None:
    _require_private_calibration_files()
    payload = json.loads(PACKET.read_text(encoding="utf-8"))
    assert sha256_file(PACKET) == PACKET_SHA256
    assert payload["packet_version"] == "m5a-human-calibration-v1"
    assert len(payload["case_ids"]) == PACKET_CASE_COUNT
    assert len(set(payload["case_ids"])) == PACKET_CASE_COUNT
    with RATER_ONE.open("r", encoding="utf-8-sig", newline="") as handle:
        first = list(csv.DictReader(handle))
    with RATER_TWO.open("r", encoding="utf-8-sig", newline="") as handle:
        second = list(csv.DictReader(handle))
    assert tuple(first[0]) == RATING_COLUMNS
    assert [row["case_id"] for row in first] == payload["case_ids"]
    assert first == second
    assert RATER_ONE.read_bytes() == RATER_TWO.read_bytes()


def test_human_review_case_forbids_hidden_or_unsafe_fields() -> None:
    payload = {
        "case_id": "case-001",
        "display_index": 1,
        "instruction": "Review the observable result.",
        "source_media_ref": {
            "role": "source",
            "relative_path": "source.mp4",
            "media_type": "video/mp4",
        },
        "source_media_metadata": {
            "duration_ms": 1000,
            "has_video": True,
            "has_audio": False,
            "width": 32,
            "height": 32,
        },
        "public_trajectory_summary": {
            "tool_observations": [],
            "verification_statuses": [],
            "structured_decision_failure_count": 1,
            "final_output_present": False,
        },
        "terminal_behavior": "MODEL_OUTPUT_FAILURE",
        "review_instructions": ["Use displayed evidence only."],
    }
    case = HumanReviewCase.model_validate(payload)
    assert case.case_id == "case-001"
    with pytest.raises(ValidationError):
        HumanReviewCase.model_validate({**payload, "split": "dev"})
    with pytest.raises(ValidationError):
        HumanReviewCase.model_validate({**payload, "expected_answer": "success"})
    with pytest.raises(ValidationError):
        HumanReviewMediaRef(role="source", relative_path="../secret.mp4", media_type="video/mp4")
    with pytest.raises(ValidationError):
        HumanReviewMediaRef(role="source", relative_path="C:/secret.mp4", media_type="video/mp4")


def test_builder_has_no_offline_or_private_imports() -> None:
    path = REPOSITORY / "scripts/m10_build_human_review_bundle.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    all_imports = modules | imported
    assert not any(name.startswith("cutagent_evaluation") for name in all_imports)
    assert not any("private" in name.casefold() for name in all_imports)


def test_review_bundle_is_complete_leak_free_and_reproducible(tmp_path: Path) -> None:
    _require_private_calibration_files()
    public_tasks, trajectories = _write_fixture_inputs(tmp_path)
    first = _config(tmp_path, public_tasks, trajectories, "first")
    second = _config(tmp_path, public_tasks, trajectories, "second")
    first_result = build_review_bundle(first, media_probe=_fake_probe)
    second_result = build_review_bundle(second, media_probe=_fake_probe)
    assert first_result["status"] == "passed"
    assert first_result["rateable_case_count"] == PACKET_CASE_COUNT
    assert first_result["leak_count"] == 0
    assert first_result["bundle_sha256"] == second_result["bundle_sha256"]
    assert bundle_tree_sha256(first.output_root) == first_result["bundle_sha256"]
    manifest = json.loads((first.output_root / "review_manifest.json").read_text())
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    assert manifest["case_count"] == PACKET_CASE_COUNT
    assert [item["case_id"] for item in manifest["cases"]] == packet["case_ids"]
    assert manifest["case_order_preserved"] is True
    assert len(list((first.output_root / "cases").glob("case_*/case.json"))) == 50
    assert len(list((first.output_root / "cases").glob("case_*/page.html"))) == 50
    assert (first.output_root / "downloads/rater_1.csv").read_bytes() == RATER_ONE.read_bytes()
    assert (first.output_root / "downloads/rater_2.csv").read_bytes() == RATER_TWO.read_bytes()
    assert audit_bundle_leakage(first.output_root)["leak_count"] == 0


def test_bundle_audit_detects_hidden_material(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "index.html").write_text(
        "<html><body><!-- hidden --><span data-answer='x'>visible</span></body></html>",
        encoding="utf-8",
    )
    audit = audit_bundle_leakage(root)
    assert audit["status"] == "failed"
    assert audit["leak_count"] == 2


def test_frozen_benchmark_manifest_hash_marker_unchanged() -> None:
    _require_private_calibration_files()
    value = MANIFEST_MARKER.read_text(encoding="utf-8").split()[0]
    assert value == "d2eecff95ac8ec974b25924be41561d49d83de13e84216728be33a311bfe4842"
    assert _digest(PACKET.read_bytes()) == PACKET_SHA256
