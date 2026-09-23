"""Audit train-side M8 preference shortcuts without accessing protected data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from cutagent_training.m8_contracts import M8RMInput, M8RMLabel


def _decision_type(candidate: Mapping[str, object]) -> str:
    value = candidate.get("decision_type") or candidate.get("recovery_type")
    return str(value or "unknown")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = tuple(
        M8RMInput.model_validate_json(line)
        for line in (args.dataset_root / "inputs" / "holdout.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    )
    labels = tuple(
        M8RMLabel.model_validate_json(line)
        for line in (args.dataset_root / "labels" / "holdout.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    )
    label_by_id = {item.input_id: item for item in labels}
    shorter = longer = nonterminal = tool = 0
    type_pairs: Counter[str] = Counter()
    for item in inputs:
        label = label_by_id[item.input_id]
        preferred_a = label.preferred_candidate == "a"
        length_a, length_b = len(item.render_candidate("a")), len(item.render_candidate("b"))
        shorter += (length_a <= length_b) == preferred_a
        longer += (length_a >= length_b) == preferred_a
        type_a, type_b = _decision_type(item.candidate_a), _decision_type(item.candidate_b)
        terminal = {"finish", "cannot_complete"}
        predicts_a_nonterminal = type_a not in terminal and type_b in terminal
        predicts_b_nonterminal = type_b not in terminal and type_a in terminal
        if predicts_a_nonterminal or predicts_b_nonterminal:
            nonterminal += predicts_a_nonterminal == preferred_a
        is_tool_a, is_tool_b = type_a == "tool", type_b == "tool"
        if is_tool_a != is_tool_b:
            tool += is_tool_a == preferred_a
        type_pairs[f"{type_a}|{type_b}"] += 1
    count = len(inputs)
    payload = {
        "schema_version": "1.0",
        "split": "train_source_disjoint_holdout",
        "pair_count": count,
        "shorter_output_heuristic_accuracy": shorter / count,
        "longer_output_heuristic_accuracy": longer / count,
        "nonterminal_heuristic_correct_count": nonterminal,
        "tool_heuristic_correct_count": tool,
        "candidate_type_pair_counts": dict(type_pairs),
        "preferred_side_counts": dict(Counter(item.preferred_candidate for item in labels)),
        "label_model_input_physical_separation": True,
        "protected_access_count": 0,
        "warning": (
            "Terminal-vs-tool heuristics are confounded by controlled counterfactual templates; "
            "these diagnostics prohibit a human-alignment claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(cast(object, payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
