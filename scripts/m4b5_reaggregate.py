"""Recompute M4B.5 evaluator-private recovery metrics from persisted trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cutagent_evaluation.m4b5_recovery import (
    M4B5_DATASET_VERSION,
    M4B5_EVALUATOR_VERSION,
    DeterministicFailureInjectingRegistry,
    FailureInjectionTrigger,
    M4B5RecoveryEvaluation,
    M4B5TaskGold,
    M4B5Variant,
    evaluate_m4b5_trajectory,
    finalize_trigger,
    summarize_m4b5_evaluations,
)

from cutagent.schemas.m4b_agent import M4BAgentTrajectory, M4BRuntimeConfig
from scripts.m4b_run_experiment import _write_json

VARIANTS: tuple[M4B5Variant, ...] = ("handoff_only", "compact_recovery")


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m4b_5"))
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    raw_gold = _load_json(root / "private" / "validation_gold_and_injection.json")
    if not isinstance(raw_gold, list):
        raise RuntimeError("M4B.5 private validation annotation must be a list")
    loaded_gold = tuple(M4B5TaskGold.model_validate(item) for item in raw_gold)
    gold_values: list[M4B5TaskGold] = []
    for item in loaded_gold:
        expected: tuple[str, ...] | None = None
        if item.task_gold.category == "impossible":
            expected = ("cannot_recover",)
        elif item.injection.failure_type == "search_no_results":
            expected = ("retry_current_node", "modify_current_node")
        gold_values.append(
            item.model_copy(
                update={
                    "injection": item.injection.model_copy(
                        update={"expected_recovery_operations": expected}
                    )
                }
            )
            if expected is not None
            else item
        )
    gold = tuple(gold_values)
    _write_json(
        root / "private" / "validation_gold_and_injection.json",
        [item.model_dump(mode="json") for item in gold],
    )
    selected = tuple(item for item in VARIANTS if args.variant is None or item == args.variant)
    experiment_path = root / "experiment_summary.json"
    experiment = _load_json(experiment_path) if experiment_path.is_file() else None
    for variant in selected:
        evaluations: list[M4B5RecoveryEvaluation] = []
        triggers: list[FailureInjectionTrigger] = []
        for item in gold:
            trajectory_path = root / "progress" / variant / f"{item.task_id}.json"
            trigger_path = root / "private" / "progress" / variant / f"{item.task_id}.json"
            if not trajectory_path.is_file() or not trigger_path.is_file():
                raise RuntimeError(f"missing persisted M4B.5 evidence for {variant}/{item.task_id}")
            trajectory = M4BAgentTrajectory.model_validate_json(
                trajectory_path.read_text(encoding="utf-8")
            )
            trigger = FailureInjectionTrigger.model_validate_json(
                trigger_path.read_text(encoding="utf-8")
            )
            trigger = finalize_trigger(trigger, trajectory)
            triggers.append(trigger)
            _write_json(trigger_path, trigger.model_dump(mode="json"))
            evaluations.append(evaluate_m4b5_trajectory(trajectory, item, trigger, variant=variant))
        _write_json(
            root / "private" / "triggers" / f"{variant}.json",
            [item.model_dump(mode="json") for item in triggers],
        )
        summary = summarize_m4b5_evaluations(evaluations)
        _write_json(
            root / "evaluations" / f"{variant}.json",
            [item.model_dump(mode="json") for item in evaluations],
        )
        _write_json(root / "metrics" / f"{variant}.json", summary.model_dump(mode="json"))
        if isinstance(experiment, dict):
            variants = experiment.get("variants")
            if isinstance(variants, dict) and isinstance(variants.get(variant), dict):
                variants[variant]["metrics"] = summary.model_dump(mode="json")
    if isinstance(experiment, dict):
        experiment["dataset_version"] = M4B5_DATASET_VERSION
        experiment["evaluator_version"] = M4B5_EVALUATOR_VERSION
        experiment["failure_injector_version"] = DeterministicFailureInjectingRegistry.version
        experiment["runtime_configs"] = {
            variant: M4BRuntimeConfig(protocol_variant=variant).model_dump(mode="json")
            for variant in VARIANTS
        }
        _write_json(experiment_path, experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
