"""Compare frozen M6 SFT and M7 DPO results on the same validation tasks.

This is an evaluator-only command.  Private validation annotations are used
solely to aggregate already-produced evaluations by difficulty; no private
field or per-task label is copied into the emitted summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _rates_by(
    evaluations: list[dict[str, Any]],
    labels: dict[str, str],
) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[bool]] = defaultdict(list)
    for item in evaluations:
        values[labels[item["task_id"]]].append(bool(item["task_success"]))
    return {
        key: {"count": len(rows), "task_success_rate": sum(rows) / len(rows)}
        for key, rows in sorted(values.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-directory", type=Path, required=True)
    parser.add_argument("--dpo-directory", type=Path, required=True)
    parser.add_argument("--validation-gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for directory in (args.sft_directory, args.dpo_directory):
        declaration = _load(directory / "run_declaration.json")
        if declaration.get("split") != "validation":
            raise ValueError(f"comparison accepts validation runs only: {directory}")

    gold_rows = _load(args.validation_gold)
    if any(item.get("split") != "validation" for item in gold_rows):
        raise ValueError("validation Gold file contains a non-validation record")
    difficulty = {item["task_id"]: item["difficulty"] for item in gold_rows}

    sft_evaluations = _load(args.sft_directory / "evaluations.json")
    dpo_evaluations = _load(args.dpo_directory / "evaluations.json")
    sft_ids = [item["task_id"] for item in sft_evaluations]
    dpo_ids = [item["task_id"] for item in dpo_evaluations]
    if sft_ids != dpo_ids or set(sft_ids) != set(difficulty):
        raise ValueError("SFT, DPO, and validation Gold task sets/order must match exactly")

    sft_summary = _load(args.sft_directory / "metrics.json")["metrics"]
    dpo_summary = _load(args.dpo_directory / "metrics.json")["metrics"]
    scalar_metrics = (
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
        "structured_output_validity",
        "loop_stagnation_rate",
        "average_agent_steps",
        "average_tool_calls",
    )
    comparison = {
        key: {
            "sft": sft_summary[key],
            "dpo": dpo_summary[key],
            "delta_dpo_minus_sft": dpo_summary[key] - sft_summary[key],
        }
        for key in scalar_metrics
    }
    payload = {
        "schema_version": "1.0",
        "comparison": "M6_SFT_vs_M7_DPO_same_frozen_validation",
        "task_count": len(sft_ids),
        "task_order_identical": True,
        "metrics": comparison,
        "success_by_family": {
            "sft": sft_summary["success_by_family"],
            "dpo": dpo_summary["success_by_family"],
        },
        "success_by_difficulty": {
            "sft": _rates_by(sft_evaluations, difficulty),
            "dpo": _rates_by(dpo_evaluations, difficulty),
        },
        "terminal_failure_counts": {
            "sft": sft_summary["failure_counts"],
            "dpo": dpo_summary["failure_counts"],
        },
        "private_gold_usage": "evaluator_only_aggregation",
        "private_gold_sha256": _sha256(args.validation_gold),
        "protected_access_count": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
