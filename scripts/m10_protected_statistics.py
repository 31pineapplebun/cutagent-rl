"""Descriptive paired source-bootstrap intervals from completed evaluation outputs.

Reads evaluator results and public trajectory inputs only, never sealed Gold.
This reporting analysis does not change frozen evaluator metrics or model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from scripts.finalize_after_human_gate import _FROZEN_M6_ADAPTER_SHA256, _MODEL_ID, _MODEL_REVISION


class TaskOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    success: StrictBool


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_source_bootstrap(
    baseline: Sequence[TaskOutcome],
    sft: Sequence[TaskOutcome],
    *,
    seed: int = 20260922,
    resamples: int = 10000,
) -> dict[str, Any]:
    if resamples < 100 or seed < 0:
        raise ValueError("require at least 100 resamples and a nonnegative seed")
    left = {item.task_id: item for item in baseline}
    right = {item.task_id: item for item in sft}
    if len(left) != len(baseline) or len(right) != len(sft):
        raise ValueError("duplicate task IDs")
    if not left or left.keys() != right.keys():
        raise ValueError("paired models must have the same nonempty task set")
    groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for task_id in sorted(left):
        first, second = left[task_id], right[task_id]
        if first.source_id != second.source_id:
            raise ValueError("paired task source mismatch")
        groups[first.source_id].append((first.success, second.success))
    if len(groups) < 2:
        raise ValueError("source-level intervals require at least two sources")
    counts = [
        (len(rows), sum(a for a, _ in rows), sum(b for _, b in rows))
        for _, rows in sorted(groups.items())
    ]
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {"baseline": [], "sft": [], "sft_minus_baseline": []}
    for _ in range(resamples):
        drawn = [counts[rng.randrange(len(counts))] for _ in counts]
        denominator = sum(row[0] for row in drawn)
        first_rate = sum(row[1] for row in drawn) / denominator
        second_rate = sum(row[2] for row in drawn) / denominator
        samples["baseline"].append(first_rate)
        samples["sft"].append(second_rate)
        samples["sft_minus_baseline"].append(second_rate - first_rate)
    first_point = sum(item.success for item in baseline) / len(baseline)
    second_point = sum(item.success for item in sft) / len(sft)
    points = {
        "baseline": first_point,
        "sft": second_point,
        "sft_minus_baseline": second_point - first_point,
    }
    return {
        "method": "paired_source_cluster_percentile_bootstrap_v1",
        "purpose": "posthoc_descriptive_uncertainty_not_model_selection",
        "seed": seed,
        "resamples": resamples,
        "confidence_level": 0.95,
        "task_count": len(left),
        "source_count": len(groups),
        "estimates": {
            name: {
                "point": points[name],
                "lower": _percentile(values, 0.025),
                "upper": _percentile(values, 0.975),
            }
            for name, values in samples.items()
        },
        "limitation": (
            "Few generated sources; descriptive percentile intervals only. "
            "Degenerate all-zero intervals do not prove a zero population success rate."
        ),
    }


def load_outcomes(
    root: Path, *, expected_split: str | None = None, expected_adapter: str | None = None
) -> tuple[tuple[TaskOutcome, ...], dict[str, str]]:
    hashes: dict[str, str] = {}

    def read(path: Path) -> Any:
        raw = path.read_bytes()
        hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    summary = read(root / "metrics.json")
    if summary.get("status") != "OFFICIAL_PROTECTED_EVALUATION":
        raise ValueError("require completed official metrics, not validation")
    if expected_split is not None or expected_adapter is not None:
        declaration = summary.get("declaration", {})
        if (
            declaration.get("split") != expected_split
            or declaration.get("adapter_hash") != expected_adapter
            or declaration.get("model_id") != _MODEL_ID
            or declaration.get("model_revision") != _MODEL_REVISION
            or declaration.get("benchmark_version") != "cutagentbench-v0.1"
            or declaration.get("protocol_variant") != "handoff_only"
        ):
            raise ValueError("official model/split declaration mismatch")
    evaluations = read(root / "evaluations.json")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("require nonempty evaluation records")
    sources: dict[str, str] = {}
    for path in sorted((root / "trajectory_artifacts").glob("*.json")):
        trajectory = read(path)
        task = trajectory["task_input"]
        task_id = task["task_id"]
        if task_id in sources:
            raise ValueError("duplicate trajectory task")
        sources[task_id] = task["video_ref"]["sha256"]
    outcomes = tuple(
        TaskOutcome(
            task_id=item["task_id"],
            source_id=sources[item["task_id"]],
            success=item["task_success"],
        )
        for item in evaluations
    )
    ids = {item.task_id for item in outcomes}
    if len(ids) != len(outcomes) or ids != sources.keys():
        raise ValueError("evaluation/trajectory task coverage mismatch")
    count = len(outcomes)
    if (
        summary["task_count"] != count
        or summary["metrics"]["task_count"] != count
        or summary["deterministic_replay_count"] != count
        or summary["private_trajectory_leak_count"] != 0
        or summary["protected_access_count"] != 1
    ):
        raise ValueError("incomplete or invalid official run integrity checks")
    observed = sum(item.success for item in outcomes) / count
    if not math.isclose(observed, summary["metrics"]["task_success_rate"], abs_tol=1e-12):
        raise ValueError("per-task outcomes disagree with frozen TSR")
    return outcomes, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.protected_root.resolve(strict=True)
    ledger = json.loads((root / "execution_ledger.json").read_text(encoding="utf-8"))
    expected_jobs = {
        (split, model)
        for split in ("locked_test", "adversarial_test")
        for model in ("prompt_only_handoff", "selected_m6_sft")
    }
    jobs = ledger.get("jobs", [])
    if (
        ledger.get("status") != "FULLY_COMPLETE"
        or len(jobs) != 4
        or {(job["split"], job["model"]) for job in jobs} != expected_jobs
        or any(job["status"] not in {"completed", "resumed_completed_result"} for job in jobs)
        or ledger.get("protected_access_count") != 4
    ):
        raise ValueError("all four official jobs must be complete before reporting")
    results: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    for split, expected_count, expected_sources in (
        ("locked_test", 60, 12),
        ("adversarial_test", 30, 6),
    ):
        baseline, baseline_hashes = load_outcomes(
            root / "runs" / split / "prompt_only_handoff",
            expected_split=split,
            expected_adapter="none/base",
        )
        sft, sft_hashes = load_outcomes(
            root / "runs" / split / "selected_m6_sft",
            expected_split=split,
            expected_adapter=_FROZEN_M6_ADAPTER_SHA256,
        )
        if len(baseline) != expected_count or len(sft) != expected_count:
            raise ValueError("official split size mismatch")
        results[split] = paired_source_bootstrap(baseline, sft)
        if results[split]["source_count"] != expected_sources:
            raise ValueError("official source count mismatch")
        inputs[split] = {"baseline": baseline_hashes, "sft": sft_hashes}
    payload = {
        "schema_version": "1.0",
        "benchmark_version": "cutagentbench-v0.1",
        "status": "descriptive_analysis_complete",
        "gold_read": False,
        "model_inference_performed": False,
        "input_sha256": inputs,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
