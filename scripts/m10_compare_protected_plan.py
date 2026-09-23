"""Compare a protected dry-run declaration with the frozen M10 execution plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cutagent_evaluation.m5a_evaluator import CutAgentBenchMetricsSummary

from scripts.finalize_after_human_gate import (
    _DECLARED_JOBS,
    _DECLARED_SPLITS,
    _FROZEN_PLAN_CONSTRAINTS,
    _FROZEN_PLAN_METRIC_SEMANTICS,
    _NO_MUTATION_CONSTRAINTS,
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def compare_declaration(
    frozen_plan: dict[str, Any],
    dry_run: dict[str, Any],
) -> dict[str, Any]:
    declaration = dry_run.get("would_run")
    if not isinstance(declaration, dict):
        declaration = {}
    evaluator_registry = [
        name for name in CutAgentBenchMetricsSummary.model_fields if name != "schema_version"
    ]
    selected_models = declaration.get("selected_models")
    selected_ids = (
        [str(item.get("official_id")) for item in selected_models]
        if isinstance(selected_models, list)
        and all(isinstance(item, dict) for item in selected_models)
        else []
    )
    excluded_models = declaration.get("excluded_models")
    excluded_ids = sorted(excluded_models) if isinstance(excluded_models, dict) else []
    excluded_values = (
        set(str(value) for value in excluded_models.values())
        if isinstance(excluded_models, dict)
        else set()
    )
    comparisons = {
        "dry_run_status": dry_run.get("status") == "human_gate_inputs_valid_dry_run",
        "protected_access_zero": dry_run.get("protected_access_count") == 0
        and declaration.get("protected_access_count_during_dry_run") == 0,
        "execution_arguments_complete": dry_run.get("protected_execution_arguments_complete")
        is True,
        "selected_models_exact": selected_models == frozen_plan.get("selected_models"),
        "selected_model_count": len(selected_ids) == 2,
        "selected_model_identities": selected_ids == ["prompt_only_m4b_handoff", "m6_sft"],
        "excluded_models_separate": excluded_ids == ["m7_dpo", "m8_reward_model", "m9_grpo"]
        and excluded_values == {"NOT_SELECTED_FOR_PROTECTED_EVALUATION"},
        "excluded_models_not_selected": not (set(selected_ids) & set(excluded_ids)),
        "declared_splits_exact": declaration.get("declared_splits")
        == list(_DECLARED_SPLITS)
        == frozen_plan.get("declared_splits"),
        "declared_jobs_exact": declaration.get("declared_jobs")
        == list(_DECLARED_JOBS)
        == frozen_plan.get("declared_jobs"),
        "official_job_count": len(declaration.get("declared_jobs", [])) == 4,
        "evaluator_metric_registry_exact": declaration.get("declared_metrics")
        == evaluator_registry,
        "frozen_metric_semantics_exact": declaration.get("frozen_plan_metric_semantics")
        == list(_FROZEN_PLAN_METRIC_SEMANTICS)
        == frozen_plan.get("declared_metrics"),
        "unsupported_metric_explicit": declaration.get("unsupported_requested_metrics")
        == {
            "secondary_failure_distribution": (
                "Per-task secondary_failures exist, but the frozen "
                "CutAgentBenchMetricsSummary has no aggregate secondary distribution field."
            )
        },
        "benchmark_exact": declaration.get("benchmark") == frozen_plan.get("benchmark"),
        "evaluator_exact": declaration.get("evaluator") == frozen_plan.get("evaluator"),
        "base_model_exact": declaration.get("base_model") == frozen_plan.get("base_model"),
        "runtime_exact": declaration.get("runtime") == frozen_plan.get("runtime"),
        "git_commit_exact": declaration.get("git_commit") == frozen_plan.get("git_commit"),
        "human_gate_hash_exact": declaration.get("human_gate_acceptance_sha256")
        == frozen_plan.get("human_gate_acceptance_sha256"),
        "access_reason_exact": declaration.get("access_reason") == frozen_plan.get("access_reason"),
        "frozen_constraints_exact": declaration.get("frozen_plan_constraints")
        == dict(_FROZEN_PLAN_CONSTRAINTS)
        == frozen_plan.get("constraints"),
        "expanded_no_mutation_constraints_exact": declaration.get("constraints")
        == dict(_NO_MUTATION_CONSTRAINTS),
        "rerun_policy_exact": declaration.get("protected_job_rerun_policy")
        == "one_official_declared_run_per_model_split_pair",
        "protected_results_excluded_from_selection": declaration.get(
            "protected_results_may_not_feed_training_or_selection"
        )
        is True,
    }
    mismatches = [name for name, passed in comparisons.items() if not passed]
    return {
        "schema_version": "1.0",
        "status": (
            "EXACT_EXECUTION_SEMANTIC_MATCH" if not mismatches else "EXECUTION_SEMANTIC_MISMATCH"
        ),
        "comparisons": comparisons,
        "mismatches": mismatches,
        "selected_model_count": len(selected_ids),
        "official_job_count": len(declaration.get("declared_jobs", [])),
        "declared_split_count": len(declaration.get("declared_splits", [])),
        "evaluator_metric_count": len(evaluator_registry),
        "allowed_nonsemantic_differences": [
            "timestamp",
            "dry_run_artifact_id",
            "environment_diagnostics",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--dry-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_declaration(
        _load_object(args.frozen_plan.resolve(strict=True)),
        _load_object(args.dry_run.resolve(strict=True)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["status"] == "EXACT_EXECUTION_SEMANTIC_MATCH" else 2


if __name__ == "__main__":
    raise SystemExit(main())
