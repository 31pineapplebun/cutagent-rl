"""Validate the human gate and prepare (never silently execute) protected evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from cutagent.schemas.m4b_agent import M4BRuntimeConfig
from cutagent.schemas.retrieval import AdaptiveRetrievalConfig, RetrievalConfig

try:
    from scripts.m10_protected_preparation import (
        PROTECTED_PREPARATION_SPLITS,
        ProtectedPublicPreparationAccessRecord,
        issue_protected_preparation_authorization,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from m10_protected_preparation import (  # type: ignore[no-redef,import-not-found]
        PROTECTED_PREPARATION_SPLITS,
        ProtectedPublicPreparationAccessRecord,
        issue_protected_preparation_authorization,
    )

try:
    from scripts.m5a_human_gate_lib import agreement_payload, validate_pair
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from m5a_human_gate_lib import (  # type: ignore[no-redef,import-not-found]
        agreement_payload,
        validate_pair,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
_MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
_FROZEN_GIT_COMMIT = "d5abc3f81d6431bbcbf9549fa624869148379f30"
_FROZEN_M6_ADAPTER = "<local-adapter-checkpoint-required>"
_FROZEN_M6_ADAPTER_SHA256 = "5dd1546e01f2b8f8f27652f9d949199442b65ec345e8b32447bbf6a3ad1e0a4a"
_BENCHMARK_MANIFEST_SHA256 = "d2eecff95ac8ec974b25924be41561d49d83de13e84216728be33a311bfe4842"
_LOCKED_TEST_SEAL_SHA256 = "5c4bfe0a1b5215965423013245d8c06ae5d4fdf7cfb71798a46d2e001d8d8443"
_ADVERSARIAL_TEST_SEAL_SHA256 = "8c931e4837c832a4a6815797f665176fe0543ca1e259b744826ac0172b190ac6"
_EVALUATOR_CONTRACT_SHA256 = "d785867c3036ec31a52859ed0d44eff81bb99167153daaf5830fa634d9edd685"
_EVALUATOR_SOURCE_SHA256 = "95292c8ada7ecb4640182453899249b747d573f1a4885b8a668b80fbd4e32f3b"
_HUMAN_GATE_ACCEPTANCE_SHA256 = "35132c3ace2257c0e2e5d2e28a63266422e56376ee8ee2189fbb09950756ab16"
_DECLARED_SPLITS = ("locked_test", "adversarial_test")
_DECLARED_JOBS = (
    "prompt_only_m4b_handoff_x_locked_test",
    "m6_sft_x_locked_test",
    "prompt_only_m4b_handoff_x_adversarial_test",
    "m6_sft_x_adversarial_test",
)
_FROZEN_PLAN_METRIC_SEMANTICS = (
    "task_success_rate",
    "hard_constraint_satisfaction",
    "correct_final_artifact_rate",
    "correct_impossible_refusal_rate",
    "recall_at_1",
    "recall_at_5",
    "mrr",
    "temporal_iou",
    "tool_selection_accuracy",
    "tool_argument_validity",
    "invalid_tool_call_rate",
    "structured_output_validity",
    "average_agent_steps",
    "average_tool_calls",
    "average_search_calls",
    "repeated_action_rate",
    "budget_exhaustion_rate",
    "correct_termination_rate",
    "premature_finish_rate",
    "premature_refusal_rate",
    "loop_stagnation_rate",
    "recovery_opportunity_count",
    "recovery_trigger_rate",
    "local_recovery_rate",
    "conditional_full_recovery_rate",
    "primary_failure_distribution",
    "secondary_failure_distribution",
)
_EVALUATOR_METRIC_REGISTRY = (
    "task_count",
    "task_success_rate",
    "hard_constraint_satisfaction",
    "correct_final_artifact_rate",
    "correct_impossible_refusal_rate",
    "recall_at_1",
    "recall_at_5",
    "mrr",
    "temporal_iou",
    "tool_selection_accuracy",
    "tool_argument_validity",
    "invalid_tool_call_rate",
    "structured_output_validity",
    "average_agent_steps",
    "average_tool_calls",
    "average_search_calls",
    "repeated_action_rate",
    "budget_exhaustion_rate",
    "correct_termination_rate",
    "premature_finish_rate",
    "premature_refusal_rate",
    "loop_stagnation_rate",
    "recovery_opportunity_count",
    "recovery_trigger_rate",
    "local_recovery_rate",
    "conditional_full_recovery_rate",
    "success_by_family",
    "failure_counts",
)
_NO_MUTATION_CONSTRAINTS: dict[str, bool] = {
    "training_allowed": False,
    "model_weight_mutation_allowed": False,
    "checkpoint_selection_allowed": False,
    "prompt_tuning_allowed": False,
    "retrieval_tuning_allowed": False,
    "reward_tuning_allowed": False,
    "benchmark_mutation_allowed": False,
    "evaluator_mutation_allowed": False,
    "model_selection_mutation_allowed": False,
    "protected_result_driven_rerun_allowed": False,
    "protected_gold_policy_visibility": False,
}
_FROZEN_PLAN_CONSTRAINTS: dict[str, bool] = {
    "training_allowed": False,
    "prompt_or_config_mutation_allowed": False,
    "checkpoint_selection_allowed": False,
    "benchmark_or_evaluator_mutation_allowed": False,
    "protected_result_driven_reruns_allowed": False,
    "protected_gold_policy_visibility": False,
}
_DECLARED_METRICS = _EVALUATOR_METRIC_REGISTRY


def _plan(
    *,
    git_commit: str = _FROZEN_GIT_COMMIT,
    sft_adapter_checkpoint: str = _FROZEN_M6_ADAPTER,
    sft_adapter_hash: str = _FROZEN_M6_ADAPTER_SHA256,
    human_gate_acceptance_sha256: str = _HUMAN_GATE_ACCEPTANCE_SHA256,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "plan_version": "cutagentbench-v0.1-protected-final-plan-v1",
        "git_commit": git_commit,
        "human_gate_acceptance_sha256": human_gate_acceptance_sha256,
        "base_model": {
            "model_id": _MODEL_ID,
            "revision": _MODEL_REVISION,
            "dtype": "bfloat16",
            "inference_backend": "local_transformers_eager",
        },
        "runtime": {
            "agent": "M4B_handoff_only_hierarchical_structured",
            "protocol_variant": "handoff_only",
            "policy_view": "structured_state",
            "prompt_template_version": "m4b-handoff-policy-v1",
            "runner_config_sha256": _runner_config_hash(),
            "retrieval_policy": (
                "frozen_query_aware_weighted_with_deterministic_evidence_reranking"
            ),
            "native_video_verification_enabled": False,
            "tool_execution_interface": "frozen_M3A_ToolRegistry_only",
        },
        "benchmark": {
            "version": "cutagentbench-v0.1",
            "manifest_sha256": _BENCHMARK_MANIFEST_SHA256,
            "locked_test_seal_sha256": _LOCKED_TEST_SEAL_SHA256,
            "adversarial_test_seal_sha256": _ADVERSARIAL_TEST_SEAL_SHA256,
        },
        "evaluator": {
            "version": "m5a-objective-evaluator-v1",
            "contract_sha256": _EVALUATOR_CONTRACT_SHA256,
            "source_sha256": _EVALUATOR_SOURCE_SHA256,
        },
        "selected_models": [
            {
                "official_id": "prompt_only_m4b_handoff",
                "selection": "frozen_preprotected",
                "adapter_or_checkpoint": "none/base",
                "adapter_sha256": None,
                "access_reason_prefix": "official_cutagentbench_v0.1",
            },
            {
                "official_id": "m6_sft",
                "selection": "frozen_preprotected",
                "adapter_or_checkpoint": sft_adapter_checkpoint,
                "adapter_sha256": sft_adapter_hash,
                "access_reason_prefix": "official_cutagentbench_v0.1",
            },
        ],
        "excluded_models": {
            "m7_dpo": "NOT_SELECTED_FOR_PROTECTED_EVALUATION",
            "m8_reward_model": "NOT_SELECTED_FOR_PROTECTED_EVALUATION",
            "m9_grpo": "NOT_SELECTED_FOR_PROTECTED_EVALUATION",
        },
        "declared_splits": list(_DECLARED_SPLITS),
        "declared_jobs": list(_DECLARED_JOBS),
        "declared_metrics": list(_EVALUATOR_METRIC_REGISTRY),
        "frozen_plan_metric_semantics": list(_FROZEN_PLAN_METRIC_SEMANTICS),
        "failure_analysis_outputs": {
            "primary_failure_distribution": "failure_counts",
            "secondary_failure_distribution": "unsupported_aggregate_metric",
        },
        "unsupported_requested_metrics": {
            "secondary_failure_distribution": (
                "Per-task secondary_failures exist, but the frozen "
                "CutAgentBenchMetricsSummary has no aggregate secondary distribution field."
            )
        },
        "access_reason": "single_official_final_cutagentbench_v0.1_protected_comparison",
        "constraints": dict(_NO_MUTATION_CONSTRAINTS),
        "frozen_plan_constraints": dict(_FROZEN_PLAN_CONSTRAINTS),
        "protected_job_rerun_policy": "one_official_declared_run_per_model_split_pair",
        "protected_results_may_not_feed_training_or_selection": True,
        "protected_access_count_during_dry_run": 0,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runner_config_hash() -> str:
    identity = {
        "runtime_config": M4BRuntimeConfig(protocol_variant="handoff_only").model_dump(mode="json"),
        "retrieval_config": RetrievalConfig().model_dump(mode="json"),
        "adaptive_config": AdaptiveRetrievalConfig(native_video_enabled=False).model_dump(
            mode="json"
        ),
        "model_revision": _MODEL_REVISION,
        "prompt_version": "m4b-handoff-policy-v1",
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _access_record(
    *,
    git_commit: str,
    adapter_hash: str,
    split: str,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "git_commit": git_commit,
        "model_revision": _MODEL_REVISION,
        "adapter_or_checkpoint_hash": adapter_hash,
        "config_sha256": _runner_config_hash(),
        "benchmark_version": "cutagentbench-v0.1",
        "benchmark_manifest_sha256": _BENCHMARK_MANIFEST_SHA256,
        "evaluator_version": "m5a-objective-evaluator-v1",
        "evaluator_contract_sha256": _EVALUATOR_CONTRACT_SHA256,
        "access_reason": reason,
        "declared_metrics": list(_DECLARED_METRICS),
        "unsupported_requested_metrics": ["secondary_failure_distribution"],
        "constraints": dict(_NO_MUTATION_CONSTRAINTS),
        "protected_job_rerun_policy": "one_official_declared_run_per_model_split_pair",
        "split": split,
        "created_at": datetime.now(UTC).isoformat(),
    }


def _run_checked(command: list[str], *, repository: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        subprocess.run(
            command,
            cwd=repository,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _protected_preparation_command(
    *,
    python: str,
    split: str,
    benchmark_root: Path,
    artifact_root: Path,
    model_cache: Path,
    authorization_path: Path,
    human_gate_path: Path,
    protected_plan_path: Path,
    access_record_path: Path,
) -> list[str]:
    return [
        python,
        "-m",
        "scripts.m5b_prepare_public_media",
        "--split",
        split,
        "--benchmark-root",
        str(benchmark_root),
        "--artifact-root",
        str(artifact_root),
        "--model-cache",
        str(model_cache),
        "--protected-authorization",
        str(authorization_path),
        "--human-gate",
        str(human_gate_path),
        "--protected-plan",
        str(protected_plan_path),
        "--preparation-access-record",
        str(access_record_path),
    ]


def _protected_jobs(
    *,
    python: str,
    repository: Path,
    model_cache: Path,
    sft_adapter: Path,
    sft_hash: str,
    gate_path: Path,
    output_root: Path,
    access_root: Path,
    prepared_root: Path,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for split in ("locked_test", "adversarial_test"):
        for model_name, module, adapter_hash, prefix in (
            ("prompt_only_handoff", "scripts.m5b_run_baseline", "none/base", []),
            (
                "selected_m6_sft",
                "scripts.m6_run_validation",
                sft_hash,
                ["--adapter-checkpoint", str(sft_adapter)],
            ),
        ):
            reason = f"official_cutagentbench_v0.1_{split}_{model_name}"
            output = output_root / split / model_name
            access_path = access_root / f"{split}_{model_name}.json"
            jobs.append(
                {
                    "model": model_name,
                    "split": split,
                    "adapter_hash": adapter_hash,
                    "access_reason": reason,
                    "output": str(output),
                    "access_record": str(access_path),
                    "command": [
                        python,
                        "-m",
                        module,
                        *prefix,
                        "--split",
                        split,
                        "--prepared-root",
                        str(prepared_root),
                        "--output-directory",
                        str(output),
                        "--model-cache",
                        str(model_cache),
                        "--adapter-hash",
                        adapter_hash,
                        "--access-reason",
                        reason,
                        "--human-gate",
                        str(gate_path),
                        "--access-record",
                        str(access_path),
                        "--variant",
                        "handoff_only",
                    ],
                }
            )
    return jobs


def _protected_report(output_root: Path, jobs: list[dict[str, Any]]) -> str:
    lines = [
        "# CutAgentBench v0.1 Official Protected Comparison",
        "",
        "This report was generated only after the frozen two-human calibration gate passed.",
        "Protected results were not used for training or selection.",
        "",
        (
            "| Model | Split | TSR | Constraint Sat. | Correct Artifact | "
            "Format Validity | Loop | Avg Steps |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for job in jobs:
        metrics_path = Path(job["output"]) / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        lines.append(
            "| {model} | {split} | {tsr:.4f} | {constraints:.4f} | {artifact:.4f} | "
            "{format_valid:.4f} | {loop:.4f} | {steps:.2f} |".format(
                model=job["model"],
                split=job["split"],
                tsr=metrics["task_success_rate"],
                constraints=metrics["hard_constraint_satisfaction"],
                artifact=metrics["correct_final_artifact_rate"],
                format_valid=metrics["structured_output_validity"],
                loop=metrics["loop_stagnation_rate"],
                steps=metrics["average_agent_steps"],
            )
        )
    lines.extend(
        [
            "",
            "M7 DPO, M8 RM-assisted selection, and M9 GRPO were not selected because their "
            "pre-protected validation/controlled results were negative.",
            "",
            f"Raw official outputs: `{output_root}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--packet",
        type=Path,
        default=Path("artifacts/m5a/cutagentbench_v0.1/calibration/human_calibration_packet.json"),
    )
    parser.add_argument(
        "--rater-1",
        type=Path,
        default=Path("artifacts/m5a/human_calibration/submitted/rater_1.csv"),
    )
    parser.add_argument(
        "--rater-2",
        type=Path,
        default=Path("artifacts/m5a/human_calibration/submitted/rater_2.csv"),
    )
    parser.add_argument(
        "--adjudication",
        type=Path,
        default=Path("artifacts/m5a/human_calibration/final/adjudication.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/m5a/human_calibration/final")
    )
    parser.add_argument("--execute-protected", action="store_true")
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--sft-adapter-checkpoint", type=Path)
    parser.add_argument("--sft-adapter-hash")
    parser.add_argument(
        "--protected-output-root",
        type=Path,
        default=Path("artifacts/final_protected/cutagentbench_v0.1"),
    )
    parser.add_argument("--prepared-root", type=Path, default=Path("artifacts/m5b/prepared"))
    parser.add_argument(
        "--final-report",
        type=Path,
        default=Path("reports/FINAL_CUTAGENTBENCH_V0.1_PROTECTED_COMPARISON.md"),
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    gate_acceptance = repository / "artifacts/m10/human_gate_acceptance.json"
    gate_acceptance_hash = (
        hashlib.sha256(gate_acceptance.read_bytes()).hexdigest()
        if gate_acceptance.is_file()
        else _HUMAN_GATE_ACCEPTANCE_SHA256
    )
    infrastructure_git_commit = _git_commit(repository)
    plan = _plan(
        git_commit=_FROZEN_GIT_COMMIT,
        sft_adapter_checkpoint=(
            str(args.sft_adapter_checkpoint)
            if args.sft_adapter_checkpoint is not None
            else _FROZEN_M6_ADAPTER
        ),
        sft_adapter_hash=args.sft_adapter_hash or _FROZEN_M6_ADAPTER_SHA256,
        human_gate_acceptance_sha256=gate_acceptance_hash,
    )
    missing = [str(path) for path in (args.rater_1, args.rater_2) if not path.is_file()]
    if missing:
        result = {
            "status": "blocked_human_calibration_pending",
            "valid_real_raters": 0,
            "required_real_raters": 2,
            "missing_submission_paths": missing,
            "protected_access_count": 0,
            "would_run": plan,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if args.dry_run else 2
    packet, first, second = validate_pair(
        args.packet.resolve(strict=True),
        args.rater_1.resolve(strict=True),
        args.rater_2.resolve(strict=True),
    )
    agreement = agreement_payload(packet, first, second)
    if agreement["disagreement_count"] and not args.adjudication.is_file():
        result = {
            "status": "blocked_adjudication_pending",
            "agreement": agreement,
            "protected_access_count": 0,
            "would_run": plan,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if args.dry_run else 2
    if args.dry_run:
        execution_ready = all(
            value is not None
            for value in (args.model_cache, args.sft_adapter_checkpoint, args.sft_adapter_hash)
        )
        print(
            json.dumps(
                {
                    "status": "human_gate_inputs_valid_dry_run",
                    "agreement": agreement,
                    "protected_access_count": 0,
                    "protected_execution_arguments_complete": execution_ready,
                    "infrastructure_git_commit": infrastructure_git_commit,
                    "would_run": plan,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    adjudication_sha256 = None
    if args.adjudication.is_file():
        adjudication_sha256 = hashlib.sha256(args.adjudication.read_bytes()).hexdigest()
    gate = {
        "schema_version": "1.0",
        "status": "passed",
        "passed_at": datetime.now(UTC).isoformat(),
        "agreement": agreement,
        "adjudication_sha256": adjudication_sha256,
        "protected_access_count_at_gate": 0,
        "protected_evaluation_plan": plan,
        "note": "Gate finalization does not itself deserialize Gold or run protected models.",
    }
    payload = json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = root / "human_calibration.json"
    output.write_bytes(payload)
    (root / "human_calibration.sha256").write_text(
        f"{_sha256_bytes(payload)}  human_calibration.json\n", encoding="utf-8"
    )
    _write_json(
        root / "m5a_completion.json",
        {
            "schema_version": "1.0",
            "status": "COMPLETE_HUMAN_CALIBRATED",
            "human_calibration_sha256": _sha256_bytes(payload),
            "protected_access_count": 0,
        },
    )
    if not args.execute_protected:
        print(
            json.dumps(
                {
                    **gate,
                    "status": "human_gate_passed_protected_execution_not_requested",
                    "next_command_requires": [
                        "--execute-protected",
                        "--model-cache",
                        "--sft-adapter-checkpoint",
                        "--sft-adapter-hash",
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.model_cache is None or args.sft_adapter_checkpoint is None or not args.sft_adapter_hash:
        parser.error(
            "protected execution requires --model-cache, --sft-adapter-checkpoint, and "
            "--sft-adapter-hash"
        )
    model_cache = args.model_cache.resolve(strict=True)
    sft_adapter = args.sft_adapter_checkpoint.resolve(strict=True)
    prepared_root = args.prepared_root.resolve()
    protected_root = args.protected_output_root.resolve()
    access_root = protected_root / "access_records"
    ledger_path = protected_root / "execution_ledger.json"
    benchmark_root = repository / "artifacts/m5a/cutagentbench_v0.1"
    protected_plan_path = repository / "artifacts/m10/protected_evaluation_plan.json"
    gate_acceptance_path = repository / "artifacts/m10/human_gate_acceptance.json"
    finalization_run_id = (
        f"m10c-{infrastructure_git_commit[:12]}-{int(datetime.now(UTC).timestamp())}"
    )
    jobs = _protected_jobs(
        python=sys.executable,
        repository=repository,
        model_cache=model_cache,
        sft_adapter=sft_adapter,
        sft_hash=args.sft_adapter_hash,
        gate_path=output,
        output_root=protected_root / "runs",
        access_root=access_root,
        prepared_root=prepared_root,
    )
    commit = _git_commit(repository)
    _write_json(
        protected_root / "predeclared_execution_manifest.json",
        {
            "schema_version": "1.0",
            "created_before_protected_execution": True,
            "git_commit": commit,
            "benchmark_version": "cutagentbench-v0.1",
            "finalization_run_id": finalization_run_id,
            "jobs": [
                {key: value for key, value in job.items() if key != "command"} for job in jobs
            ],
            "not_selected": {
                "m7_dpo": "negative validation result",
                "m8_rm": "not a full Agent variant and negative decision selection result",
                "m9_grpo": "negative controlled holdout result",
            },
        },
    )
    ledger: list[dict[str, object]] = []
    preparation_records: list[ProtectedPublicPreparationAccessRecord] = []
    for split in PROTECTED_PREPARATION_SPLITS:
        authorization = issue_protected_preparation_authorization(
            split=split,
            benchmark_root=benchmark_root,
            human_gate_path=gate_acceptance_path,
            protected_plan_path=protected_plan_path,
            finalization_run_id=finalization_run_id,
        )
        authorization_path = (
            protected_root / "control_plane" / "preparation_authorizations" / f"{split}.json"
        )
        preparation_access_path = access_root / f"public_preparation_{split}.json"
        _write_json(authorization_path, authorization.model_dump(mode="json"))
        _run_checked(
            _protected_preparation_command(
                python=sys.executable,
                split=split,
                benchmark_root=benchmark_root,
                artifact_root=prepared_root.parent,
                model_cache=model_cache,
                authorization_path=authorization_path,
                human_gate_path=gate_acceptance_path,
                protected_plan_path=protected_plan_path,
                access_record_path=preparation_access_path,
            ),
            repository=repository,
            log_path=protected_root / "logs" / f"prepare_{split}.log",
        )
        preparation_record = ProtectedPublicPreparationAccessRecord.model_validate_json(
            preparation_access_path.read_text(encoding="utf-8")
        )
        if preparation_record.status != "completed":
            raise RuntimeError(f"protected public preparation did not complete: {split}")
        preparation_records.append(preparation_record)
    _write_json(
        protected_root / "access_accounting_pre_model.json",
        {
            "schema_version": "1.0",
            "finalization_run_id": finalization_run_id,
            "protected_public_preparation_events": len(preparation_records),
            "protected_gold_evaluator_accesses": 0,
            "protected_agent_model_runs": 0,
            "preparation_splits": [item.split for item in preparation_records],
        },
    )
    for job in jobs:
        output_directory = Path(job["output"])
        metrics_path = output_directory / "metrics.json"
        if metrics_path.is_file():
            ledger.append(
                {
                    "model": job["model"],
                    "split": job["split"],
                    "status": "resumed_completed_result",
                    "protected_access_count": 1,
                }
            )
            continue
        access_path = Path(job["access_record"])
        _write_json(
            access_path,
            _access_record(
                git_commit=commit,
                adapter_hash=job["adapter_hash"],
                split=job["split"],
                reason=job["access_reason"],
            ),
        )
        _run_checked(
            cast(list[str], job["command"]),
            repository=repository,
            log_path=protected_root / "logs" / f"{job['split']}_{job['model']}.log",
        )
        ledger.append(
            {
                "model": job["model"],
                "split": job["split"],
                "status": "completed",
                "protected_access_count": 1,
            }
        )
        _write_json(ledger_path, {"schema_version": "1.0", "jobs": ledger})
    report = _protected_report(protected_root, jobs)
    report_path = args.final_report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    _write_json(
        ledger_path,
        {
            "schema_version": "1.0",
            "status": "FULLY_COMPLETE",
            "finalization_run_id": finalization_run_id,
            "jobs": ledger,
            "protected_access_count": len(jobs),
            "access_accounting": {
                "protected_public_preparation_events": len(preparation_records),
                "protected_gold_evaluator_accesses": len(jobs),
                "protected_agent_model_runs": len(jobs),
            },
            "final_report": str(report_path),
        },
    )
    print(
        json.dumps(
            {
                "status": "FULLY_COMPLETE",
                "protected_access_count": len(jobs),
                "final_report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
