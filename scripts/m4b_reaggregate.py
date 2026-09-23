"""Recompute M4B private metrics from persisted trajectories without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cutagent_evaluation.m4b_agent import (
    M4BRecoveryCaseRecord,
    M4BTaskGold,
    M4BTrajectoryEvaluation,
    analyze_m4b_recovery_case,
    evaluate_m4b_trajectory,
    summarize_m4b_evaluations,
)

from cutagent.schemas.agent import AgentTrajectory
from cutagent.schemas.m4b_agent import M4BAgentTrajectory

VARIANTS = ("frozen_m4a", "handoff_only", "compact_recovery")
EVALUATOR_VERSION = "m4b-private-evaluator-v1.1"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_trajectory(path: Path, variant: str) -> AgentTrajectory | M4BAgentTrajectory:
    serialized = path.read_text(encoding="utf-8")
    if variant == "frozen_m4a":
        return AgentTrajectory.model_validate_json(serialized)
    return M4BAgentTrajectory.model_validate_json(serialized)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m4b"))
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    gold = tuple(
        M4BTaskGold.model_validate(item)
        for item in json.loads(
            (root / "private" / "validation_gold.json").read_text(encoding="utf-8")
        )
    )
    gold_by_task = {item.task_id: item for item in gold}
    if len(gold_by_task) != len(gold):
        raise RuntimeError("M4B private Gold task identifiers must be unique")
    experiment = json.loads((root / "experiment_summary.json").read_text(encoding="utf-8"))
    input_hash = hashlib.sha256()
    variant_counts: dict[str, int] = {}

    for variant in VARIANTS:
        paths = sorted((root / "progress" / variant).glob("*.json"))
        if {path.stem for path in paths} != set(gold_by_task):
            raise RuntimeError(f"{variant} trajectory set does not match private Gold")
        evaluations: list[M4BTrajectoryEvaluation] = []
        recovery_cases: list[M4BRecoveryCaseRecord] = []
        for path in paths:
            serialized = path.read_bytes()
            input_hash.update(variant.encode("utf-8"))
            input_hash.update(path.name.encode("utf-8"))
            input_hash.update(serialized)
            trajectory = _load_trajectory(path, variant)
            task_gold = gold_by_task[path.stem]
            evaluation = evaluate_m4b_trajectory(trajectory, task_gold, variant=variant)
            evaluations.append(evaluation)
            recovery_case = analyze_m4b_recovery_case(trajectory, task_gold, evaluation)
            if recovery_case is not None:
                recovery_cases.append(recovery_case)
        summary = summarize_m4b_evaluations(evaluations).model_dump(mode="json")
        _write_json(
            root / "evaluations" / f"{variant}.json",
            [item.model_dump(mode="json") for item in evaluations],
        )
        _write_json(
            root / "recovery_cases" / f"{variant}.json",
            [item.model_dump(mode="json") for item in recovery_cases],
        )
        metrics_path = root / "metrics" / f"{variant}.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["metrics"] = summary
        metrics["evaluator_version"] = EVALUATOR_VERSION
        _write_json(metrics_path, metrics)
        experiment["variants"][variant]["metrics"] = summary
        experiment["variants"][variant]["evaluator_version"] = EVALUATOR_VERSION
        variant_counts[variant] = len(evaluations)

    experiment["evaluator_version"] = EVALUATOR_VERSION
    _write_json(root / "experiment_summary.json", experiment)
    evidence = {
        "schema_version": "1.0",
        "evaluator_version": EVALUATOR_VERSION,
        "trajectory_input_sha256": input_hash.hexdigest(),
        "variant_trajectory_counts": variant_counts,
        "model_inference_performed": False,
        "reason": (
            "Correct legacy ReplanDecision patch acceptance by matching emitted "
            "PlanPatch revisions."
        ),
    }
    _write_json(root / "evaluation_reaggregation.json", evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
