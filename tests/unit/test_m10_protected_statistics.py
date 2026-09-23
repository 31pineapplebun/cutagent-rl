"""Synthetic reporting-contract tests; no model or benchmark inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.m10_protected_statistics import TaskOutcome, load_outcomes, paired_source_bootstrap


def _outcomes(values: tuple[bool, ...]) -> tuple[TaskOutcome, ...]:
    return tuple(
        TaskOutcome(task_id=f"task-{i}", source_id=f"source-{i // 2}", success=value)
        for i, value in enumerate(values)
    )


def test_paired_identical_models_have_exactly_zero_difference() -> None:
    rows = _outcomes((True, True, False, False, True, False))
    first = paired_source_bootstrap(rows, rows, resamples=200)
    assert first == paired_source_bootstrap(tuple(reversed(rows)), rows, resamples=200)
    assert first["source_count"] == 3
    assert first["estimates"]["sft_minus_baseline"] == {
        "point": 0.0,
        "lower": 0.0,
        "upper": 0.0,
    }
    assert first["estimates"]["baseline"]["point"] == 0.5


def test_bootstrap_resamples_whole_sources_and_reports_degenerate_case() -> None:
    baseline = _outcomes((True, True, False, False))
    sft = _outcomes((False, False, False, False))
    result = paired_source_bootstrap(baseline, sft, resamples=1000)
    assert result["estimates"]["baseline"] == {"point": 0.5, "lower": 0.0, "upper": 1.0}
    assert result["estimates"]["sft"] == {"point": 0.0, "lower": 0.0, "upper": 0.0}
    assert result["estimates"]["sft_minus_baseline"]["point"] == -0.5
    assert "do not prove" in result["limitation"]


def test_refuses_unpaired_duplicates_and_source_changes() -> None:
    rows = _outcomes((True, False, False, True))
    with pytest.raises(ValueError, match="duplicate"):
        paired_source_bootstrap(rows + rows[:1], rows)
    with pytest.raises(ValueError, match="same nonempty"):
        paired_source_bootstrap(rows, rows[:-1])
    changed = (rows[0].model_copy(update={"source_id": "different"}), *rows[1:])
    with pytest.raises(ValueError, match="source mismatch"):
        paired_source_bootstrap(rows, changed)
    with pytest.raises(ValueError, match="two sources"):
        paired_source_bootstrap(rows[:2], rows[:2])


def _write_fixture(root: Path) -> None:
    trajectories = root / "trajectory_artifacts"
    trajectories.mkdir()
    for index in range(2):
        (trajectories / f"{index}.json").write_text(
            json.dumps(
                {"task_input": {"task_id": f"t{index}", "video_ref": {"sha256": str(index)}}}
            ),
            encoding="utf-8",
        )
    (root / "evaluations.json").write_text(
        json.dumps(
            [{"task_id": "t0", "task_success": True}, {"task_id": "t1", "task_success": False}]
        ),
        encoding="utf-8",
    )
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "status": "OFFICIAL_PROTECTED_EVALUATION",
                "task_count": 2,
                "deterministic_replay_count": 2,
                "private_trajectory_leak_count": 0,
                "protected_access_count": 1,
                "metrics": {"task_count": 2, "task_success_rate": 0.5},
            }
        ),
        encoding="utf-8",
    )


def test_loads_only_completed_outputs_and_public_source_identity(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    rows, hashes = load_outcomes(tmp_path)
    assert [row.success for row in rows] == [True, False]
    assert len(hashes) == 4
    assert all(len(digest) == 64 for digest in hashes.values())
    assert not any("gold" in path for path in hashes)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    metrics["metrics"]["task_success_rate"] = 1.0
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        load_outcomes(tmp_path)


def test_rejects_validation_and_missing_trajectories(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    metrics_path = tmp_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["status"] = "PROVISIONAL_VALIDATION_ONLY"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="official"):
        load_outcomes(tmp_path)
    metrics["status"] = "OFFICIAL_PROTECTED_EVALUATION"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    (tmp_path / "trajectory_artifacts" / "1.json").unlink()
    with pytest.raises(KeyError):
        load_outcomes(tmp_path)


def test_refuses_missing_or_wrong_model_declaration(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="declaration mismatch"):
        load_outcomes(tmp_path, expected_split="locked_test", expected_adapter="none/base")
