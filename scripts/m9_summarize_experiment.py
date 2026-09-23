"""Consolidate M9 trainer, holdout, reward, and curriculum evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _policy_detail(root: Path, policy: str) -> dict[str, object]:
    values = _load(root / policy / "private_scores.json")
    count = len(values)
    return {
        "case_count": count,
        "decision_kind_accuracy": sum(
            item["reward_breakdown"]["decision_kind_reward"] > 0 for item in values
        )
        / count,
        "tool_or_operation_accuracy": sum(
            item["reward_breakdown"]["tool_or_operation_reward"] > 0 for item in values
        )
        / count,
        "exact_argument_accuracy": sum(
            item["reward_breakdown"]["argument_reward"] == 0.5 for item in values
        )
        / count,
        "invalid_action_rate": sum(not item["structured_valid"] for item in values) / count,
        "premature_terminal_count": sum(
            item["reward_breakdown"]["premature_terminal_penalty"] < 0 for item in values
        ),
        "repeated_action_count": sum(
            item["reward_breakdown"]["repetition_penalty"] < 0 for item in values
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--hacking-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = args.training.resolve(strict=True)
    comparison_root = args.comparison.resolve(strict=True)
    train_result = _load(training / "training_result.json")
    trainer_history = _load(training / "trainer_log_history.json")
    comparison = _load(comparison_root / "comparison.json")
    audit = _load(args.hacking_audit.resolve(strict=True))
    kl_values = [float(item["kl"]) for item in trainer_history if "kl" in item]
    loss_values = [float(item["loss"]) for item in trainer_history if "loss" in item]
    result = {
        "schema_version": "1.0",
        "training": train_result,
        "optimizer": {
            "finite_step_count": len(loss_values),
            "mean_loss": statistics.mean(loss_values),
            "first_loss": loss_values[0],
            "last_loss": loss_values[-1],
            "mean_kl": statistics.mean(kl_values),
            "maximum_kl": max(kl_values),
            "kl_is_optimizer_regularization_only": True,
        },
        "holdout": {
            "comparison": comparison,
            "initialization_action_metrics": _policy_detail(comparison_root, "initialization"),
            "grpo_action_metrics": _policy_detail(comparison_root, "grpo"),
        },
        "curriculum": {
            "single_decision": "completed_negative_result",
            "one_tool_step": "pilot_compatibility_only_structured_validity_zero",
            "two_step_sequence": "not_started_due_to_unstable_earlier_stage",
            "observable_recovery": "not_started_due_to_unstable_earlier_stage",
        },
        "reward_hacking": {key: value for key, value in audit.items() if key != "reviews"},
        "full_validation_agent_transfer": "not_run_because_controlled_stage_was_unstable",
        "protected_access_count": 0,
    }
    _write(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
