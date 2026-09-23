"""Recompute evaluator-private M4A metrics from persisted public trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cutagent_evaluation.m4a_agent import (
    M4ATaskGold,
    evaluate_trajectory,
    summarize_evaluations,
)

from cutagent.schemas.agent import AgentTrajectory


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m4a"))
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    raw_gold = json.loads((root / "private" / "development_gold.json").read_text("utf-8"))
    if not isinstance(raw_gold, list):
        raise RuntimeError("M4A private Gold must be a list")
    gold = {item.task_id: item for item in map(M4ATaskGold.model_validate, raw_gold)}
    summary_path = root / "experiment_summary.json"
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    variants = summary.get("variants")
    if not isinstance(variants, dict):
        raise RuntimeError("M4A experiment summary has no variants")
    for variant in sorted(variants):
        trajectories = [
            AgentTrajectory.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted((root / "progress" / variant).glob("*.json"))
        ]
        if len(trajectories) != len(gold):
            raise RuntimeError(
                f"variant {variant} has {len(trajectories)} trajectories for {len(gold)} tasks"
            )
        evaluations = [
            evaluate_trajectory(item, gold[item.task_input.task_id]) for item in trajectories
        ]
        metrics = summarize_evaluations(evaluations)
        _write_json(
            root / "evaluations" / f"{variant}.json",
            [item.model_dump(mode="json") for item in evaluations],
        )
        metrics_path = root / "metrics" / f"{variant}.json"
        payload: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["metrics"] = metrics.model_dump(mode="json")
        performance = payload.get("performance")
        if not isinstance(performance, dict):
            raise RuntimeError(f"variant {variant} performance summary is malformed")
        failures = [failure for item in trajectories for failure in item.policy_failures]
        steps = [step for item in trajectories for step in item.policy_steps]
        inference_count = len(steps) + len(failures)
        total_inference_ms = sum(step.stats.latency_ms for step in steps) + sum(
            failure.latency_ms for failure in failures
        )
        visual_counts = [len(step.context.visual_evidence_ids) for step in steps]
        performance.update(
            {
                "policy_inference_count": inference_count,
                "policy_failure_count": len(failures),
                "policy_failure_latency_total_ms": sum(failure.latency_ms for failure in failures),
                "mean_policy_latency_including_failures_ms": (
                    total_inference_ms / max(inference_count, 1)
                ),
                "visual_evidence_step_count": sum(count > 0 for count in visual_counts),
                "visual_evidence_image_count": sum(visual_counts),
            }
        )
        _write_json(metrics_path, payload)
        variant_payload = variants[variant]
        if not isinstance(variant_payload, dict):
            raise RuntimeError(f"variant {variant} summary is malformed")
        variant_payload["metrics"] = metrics.model_dump(mode="json")
        variant_payload["performance"] = performance
    summary["offline_metrics_recomputed"] = True
    _write_json(summary_path, summary)
    print(json.dumps({"variants": variants, "recomputed": True}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
