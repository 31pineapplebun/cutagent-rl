"""Smoke-test a frozen M5A benchmark without accessing protected Gold or running an Agent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cutagent_evaluation.m5a_calibration import handcrafted_facts
from cutagent_evaluation.m5a_evaluator import evaluate_facts
from cutagent_evaluation.m5a_schemas import (
    BenchmarkManifest,
    CutAgentBenchCase,
    CutAgentBenchGold,
)

from cutagent.schemas.task_input import TaskInput


def _load_list(path: Path) -> list[object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise RuntimeError(f"expected a JSON list: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/m5a/cutagentbench_v0.1"),
    )
    args = parser.parse_args()
    root = args.artifact_root.resolve(strict=True)
    manifest_path = root / "benchmark_manifest.json"
    manifest_bytes = manifest_path.read_bytes().rstrip(b"\n")
    manifest = BenchmarkManifest.model_validate_json(manifest_bytes)
    expected_manifest_hash = (
        (root / "benchmark_manifest.sha256").read_text(encoding="utf-8").split()[0]
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_hash:
        raise RuntimeError("benchmark manifest hash verification failed")
    public_dev = _load_list(root / "public/dev.json")
    private_dev = _load_list(root / "private/dev_gold.json")
    tasks = tuple(TaskInput.model_validate(item) for item in public_dev)
    gold = tuple(CutAgentBenchGold.model_validate(item) for item in private_dev)
    if tuple(task.task_id for task in tasks) != tuple(item.task_id for item in gold):
        raise RuntimeError("dev public/private ordering differs")
    case = CutAgentBenchCase(public_task=tasks[0], private_gold=gold[0])
    evaluation = evaluate_facts(handcrafted_facts(case, "exact_success"), case.private_gold)
    if not evaluation.task_success:
        raise RuntimeError("objective evaluator rejected the exact-success smoke case")
    locked_path = root / "private/sealed/locked_test_gold.json"
    locked_observed = hashlib.sha256(locked_path.read_bytes().rstrip(b"\n")).hexdigest()
    if locked_observed != manifest.locked_test_seal_sha256:
        raise RuntimeError("locked-test seal hash verification failed")
    result = {
        "status": "passed",
        "benchmark_version": manifest.benchmark_version,
        "manifest_sha256": expected_manifest_hash,
        "task_count": manifest.task_count,
        "dev_calibration_task_id": case.public_task.task_id,
        "locked_gold_deserialized": False,
        "agent_model_runs": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
