from __future__ import annotations

import json
from pathlib import Path

import pytest
from cutagent_evaluation.m5a_schemas import (
    AbsenceGoldConstraint,
    BenchmarkFailureCause,
    CutAgentBenchGold,
    DifficultyLevel,
    ExpectedTerminalBehavior,
    TaskFamily,
)
from cutagent_evaluation.m5b_baseline import (
    M5BRunDeclaration,
    assert_split_access,
    config_sha256,
    load_cases,
)
from cutagent_evaluation.schemas import DatasetSplit

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.task_input import TaskInput


def _declaration(split: DatasetSplit) -> M5BRunDeclaration:
    return M5BRunDeclaration(
        split=split,
        config_sha256=config_sha256({"config": "frozen"}),
        declared_metrics=("task_success_rate",),
        access_reason="unit contract test",
        output_directory="artifacts/test-output",
    )


def _write_dev_case(root: Path) -> None:
    artifact = ArtifactRef(
        artifact_id="video-unit",
        uri="file:///opaque/source.mp4",
        sha256="0" * 64,
        media_type="video/mp4",
        size_bytes=1,
    )
    task = TaskInput(task_id="task-unit", video_ref=artifact, instruction="Find absent event")
    gold = CutAgentBenchGold(
        gold_id="gold-task-unit",
        task_id=task.task_id,
        source_group_id="source-unit",
        split=DatasetSplit.DEV,
        task_family=TaskFamily.IMPOSSIBLE,
        task_subtype="unavailable_evidence",
        difficulty=DifficultyLevel.L1,
        relevant_video_ids=(),
        relevant_scene_ids=(),
        acceptable_time_ranges=(),
        required_tool_capabilities=("retrieval.read",),
        required_tools=("search_video",),
        objective_constraints=(
            AbsenceGoldConstraint(requested_description="an event that is absent"),
        ),
        expected_terminal_behavior=ExpectedTerminalBehavior.CANNOT_COMPLETE,
        primary_failure_if_unsolved=BenchmarkFailureCause.PREMATURE_FINISH,
    )
    (root / "public").mkdir(parents=True)
    (root / "private").mkdir(parents=True)
    (root / "public" / "dev.json").write_text(
        json.dumps([task.model_dump(mode="json")]), encoding="utf-8"
    )
    (root / "private" / "dev_gold.json").write_text(
        json.dumps([gold.model_dump(mode="json")]), encoding="utf-8"
    )


def test_dev_loads_without_human_gate(tmp_path: Path) -> None:
    _write_dev_case(tmp_path)
    cases = load_cases(
        tmp_path,
        _declaration(DatasetSplit.DEV),
        human_gate_path=tmp_path / "missing-gate.json",
    )
    assert len(cases) == 1
    assert cases[0][0].task_id == cases[0][1].task_id


def test_protected_split_rejected_before_gold_read(tmp_path: Path) -> None:
    declaration = _declaration(DatasetSplit.LOCKED_TEST)
    with pytest.raises(PermissionError, match="real-human"):
        load_cases(
            tmp_path,
            declaration,
            human_gate_path=tmp_path / "missing-gate.json",
        )


def test_protected_access_requires_complete_matching_record(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    declaration = _declaration(DatasetSplit.ADVERSARIAL_TEST)
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"split": "adversarial_test"}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        assert_split_access(
            declaration,
            human_gate_path=gate,
            access_record_path=record,
        )
    record.write_text(
        json.dumps(
            {
                "git_commit": "a" * 40,
                "model_revision": declaration.model_revision,
                "adapter_or_checkpoint_hash": declaration.adapter_hash,
                "config_sha256": declaration.config_sha256,
                "benchmark_version": declaration.benchmark_version,
                "access_reason": declaration.access_reason,
                "declared_metrics": list(declaration.declared_metrics),
                "split": declaration.split.value,
            }
        ),
        encoding="utf-8",
    )
    assert_split_access(
        declaration,
        human_gate_path=gate,
        access_record_path=record,
    )


def test_run_declaration_has_no_private_gold_fields() -> None:
    serialized = _declaration(DatasetSplit.VALIDATION).model_dump_json().casefold()
    assert "benchmarkgold" not in serialized
    assert "source_group_id" not in serialized
    assert "gold_id" not in serialized
