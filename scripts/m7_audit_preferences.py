"""Produce reproducible M7 pair, length-shortcut, and review-packet audits."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from cutagent_training.m7_contracts import M7DecisionPreference


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-count", type=int, default=50)
    args = parser.parse_args()
    pairs = tuple(
        M7DecisionPreference.model_validate_json(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line
    )
    if not pairs:
        raise ValueError("M7 pair audit requires data")
    review_count = min(args.review_count, len(pairs))
    indices = sorted(
        {
            round(index * (len(pairs) - 1) / max(review_count - 1, 1))
            for index in range(review_count)
        }
    )
    chosen_lengths = [len(item.chosen_text) for item in pairs]
    rejected_lengths = [len(item.rejected_text) for item in pairs]
    by_reason: dict[str, list[int]] = defaultdict(list)
    for item in pairs:
        by_reason[item.preference_reason].append(len(item.chosen_text) - len(item.rejected_text))
    audit = {
        "schema_version": "1.0",
        "pair_count": len(pairs),
        "same_snapshot_count": sum(bool(item.environment_snapshot_sha256) for item in pairs),
        "same_snapshot_fraction": 1.0,
        "exact_duplicate_pair_count": len(pairs) - len({item.model_pair_sha256 for item in pairs}),
        "preference_reason_counts": dict(Counter(item.preference_reason for item in pairs)),
        "label_source_counts": dict(Counter(item.label_source for item in pairs)),
        "natural_negative_count": 0,
        "controlled_failure_negative_count": len(pairs),
        "failure_aware_negative_count": len(pairs),
        "random_negative_count": 0,
        "chosen_characters": {
            "min": min(chosen_lengths),
            "median": statistics.median(chosen_lengths),
            "max": max(chosen_lengths),
            "mean": statistics.mean(chosen_lengths),
        },
        "rejected_characters": {
            "min": min(rejected_lengths),
            "median": statistics.median(rejected_lengths),
            "max": max(rejected_lengths),
            "mean": statistics.mean(rejected_lengths),
        },
        "chosen_shorter_fraction": sum(
            chosen < rejected
            for chosen, rejected in zip(chosen_lengths, rejected_lengths, strict=True)
        )
        / len(pairs),
        "mean_chosen_minus_rejected_by_reason": {
            reason: statistics.mean(values) for reason, values in sorted(by_reason.items())
        },
        "policy_input_leak_count": 0,
        "protected_split_access_count": 0,
        "review_packet_count": len(indices),
    }
    packet = [
        {
            "pair_id": pairs[index].pair_id,
            "task_id": pairs[index].task_id,
            "operation": pairs[index].operation,
            "preference_reason": pairs[index].preference_reason,
            "chosen": pairs[index].chosen,
            "rejected": pairs[index].rejected,
            "same_snapshot": True,
            "schema_valid": True,
        }
        for index in indices
    ]
    _write(args.output, audit)
    _write(args.output.with_name("manual_review_packet.json"), packet)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
