from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from cutagent_evaluation.m5a_evaluator import CutAgentBenchMetricsSummary

from scripts.finalize_after_human_gate import (
    _DECLARED_JOBS,
    _DECLARED_SPLITS,
    _FROZEN_PLAN_CONSTRAINTS,
    _FROZEN_PLAN_METRIC_SEMANTICS,
    _NO_MUTATION_CONSTRAINTS,
    _access_record,
    _plan,
    _protected_jobs,
    _protected_preparation_command,
    _runner_config_hash,
)
from scripts.m10_compare_protected_plan import compare_declaration


def test_finalization_declaration_selects_exactly_two_official_models() -> None:
    plan = _plan()
    selected = plan["selected_models"]
    assert isinstance(selected, list)
    assert [item["official_id"] for item in selected] == [
        "prompt_only_m4b_handoff",
        "m6_sft",
    ]
    assert len(selected) == 2
    excluded = plan["excluded_models"]
    assert excluded == {
        "m7_dpo": "NOT_SELECTED_FOR_PROTECTED_EVALUATION",
        "m8_reward_model": "NOT_SELECTED_FOR_PROTECTED_EVALUATION",
        "m9_grpo": "NOT_SELECTED_FOR_PROTECTED_EVALUATION",
    }
    assert not ({item["official_id"] for item in selected} & set(excluded))


def test_declaration_splits_jobs_and_metrics_match_frozen_contract() -> None:
    plan = _plan()
    assert (
        plan["declared_splits"]
        == list(_DECLARED_SPLITS)
        == [
            "locked_test",
            "adversarial_test",
        ]
    )
    assert plan["declared_jobs"] == list(_DECLARED_JOBS)
    assert len(plan["declared_jobs"]) == 4
    evaluator_metrics = [
        name for name in CutAgentBenchMetricsSummary.model_fields if name != "schema_version"
    ]
    assert plan["declared_metrics"] == evaluator_metrics
    assert plan["frozen_plan_metric_semantics"] == list(_FROZEN_PLAN_METRIC_SEMANTICS)
    assert plan["unsupported_requested_metrics"] == {
        "secondary_failure_distribution": (
            "Per-task secondary_failures exist, but the frozen "
            "CutAgentBenchMetricsSummary has no aggregate secondary distribution field."
        )
    }


def test_declaration_emits_all_no_mutation_constraints() -> None:
    plan = _plan()
    assert plan["constraints"] == _NO_MUTATION_CONSTRAINTS
    assert all(value is False for value in _NO_MUTATION_CONSTRAINTS.values())
    assert plan["frozen_plan_constraints"] == _FROZEN_PLAN_CONSTRAINTS
    assert plan["protected_job_rerun_policy"] == "one_official_declared_run_per_model_split_pair"
    assert plan["protected_access_count_during_dry_run"] == 0


def test_protected_jobs_are_predeclared_without_gold_paths(tmp_path: Path) -> None:
    jobs = _protected_jobs(
        python="python",
        repository=tmp_path,
        model_cache=tmp_path / "models",
        sft_adapter=tmp_path / "adapter",
        sft_hash="a" * 64,
        gate_path=tmp_path / "gate.json",
        output_root=tmp_path / "outputs",
        access_root=tmp_path / "access",
        prepared_root=tmp_path / "prepared",
    )
    assert len(jobs) == 4
    assert {job["split"] for job in jobs} == {"locked_test", "adversarial_test"}
    serialized = json.dumps(jobs).casefold()
    assert "_gold.json" not in serialized
    assert "private/sealed" not in serialized
    assert "benchmarkgold" not in serialized


def test_access_record_matches_frozen_runner_identity() -> None:
    record = _access_record(
        git_commit="b" * 40,
        adapter_hash="c" * 64,
        split="locked_test",
        reason="official_test",
    )
    assert record["config_sha256"] == _runner_config_hash()
    assert record["split"] == "locked_test"
    assert record["adapter_or_checkpoint_hash"] == "c" * 64
    evaluator_metrics = [
        name for name in CutAgentBenchMetricsSummary.model_fields if name != "schema_version"
    ]
    assert record["declared_metrics"] == evaluator_metrics
    assert record["constraints"] == _NO_MUTATION_CONSTRAINTS


def test_machine_comparator_accepts_only_exact_execution_semantics() -> None:
    declaration = _plan()
    frozen_plan = {
        key: value
        for key, value in declaration.items()
        if key
        in {
            "git_commit",
            "human_gate_acceptance_sha256",
            "base_model",
            "runtime",
            "benchmark",
            "evaluator",
            "selected_models",
            "declared_splits",
            "declared_jobs",
            "access_reason",
        }
    }
    frozen_plan["declared_metrics"] = list(_FROZEN_PLAN_METRIC_SEMANTICS)
    frozen_plan["constraints"] = dict(_FROZEN_PLAN_CONSTRAINTS)
    dry_run = {
        "status": "human_gate_inputs_valid_dry_run",
        "protected_access_count": 0,
        "protected_execution_arguments_complete": True,
        "would_run": declaration,
    }
    result = compare_declaration(frozen_plan, dry_run)
    assert result["status"] == "EXACT_EXECUTION_SEMANTIC_MATCH"
    assert result["mismatches"] == []

    declaration["declared_splits"] = ["locked_test"]
    mismatch = compare_declaration(frozen_plan, dry_run)
    assert mismatch["status"] == "EXECUTION_SEMANTIC_MISMATCH"
    assert "declared_splits_exact" in mismatch["mismatches"]


def test_declaration_build_is_read_only_for_frozen_inputs(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    checkpoint = tmp_path / "adapter.safetensors"
    config = tmp_path / "config.json"
    for path, payload in (
        (benchmark, b"benchmark"),
        (checkpoint, b"checkpoint"),
        (config, b"config"),
    ):
        path.write_bytes(payload)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (benchmark, checkpoint, config)
    }
    declaration = _plan(sft_adapter_checkpoint=str(checkpoint))
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (benchmark, checkpoint, config)
    }
    assert declaration["protected_access_count_during_dry_run"] == 0
    assert before == after


def test_finalizer_wires_complete_protected_preparation_authorization(tmp_path: Path) -> None:
    command = _protected_preparation_command(
        python="python",
        split="locked_test",
        benchmark_root=tmp_path / "benchmark",
        artifact_root=tmp_path / "artifacts",
        model_cache=tmp_path / "models",
        authorization_path=tmp_path / "authorization.json",
        human_gate_path=tmp_path / "human-gate.json",
        protected_plan_path=tmp_path / "protected-plan.json",
        access_record_path=tmp_path / "public-preparation-access.json",
    )
    assert command[:5] == [
        "python",
        "-m",
        "scripts.m5b_prepare_public_media",
        "--split",
        "locked_test",
    ]
    for flag in (
        "--benchmark-root",
        "--protected-authorization",
        "--human-gate",
        "--protected-plan",
        "--preparation-access-record",
    ):
        assert flag in command
    serialized = json.dumps(command).casefold()
    assert "_gold.json" not in serialized
    assert "private/sealed" not in serialized


def test_finalizer_direct_script_import_path_is_supported() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/finalize_after_human_gate.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--execute-protected" in result.stdout
