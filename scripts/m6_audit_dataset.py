"""Audit an M6 decision-level SFT dataset without reading benchmark Gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from cutagent_training.contracts import M6AgentSFTRecord


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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

    raw_lines = args.dataset.read_text(encoding="utf-8").splitlines()
    records = tuple(M6AgentSFTRecord.model_validate_json(line) for line in raw_lines if line)
    if not records:
        raise ValueError("M6 audit requires at least one record")
    review_count = min(args.review_count, len(records))
    # Stable spread across the complete, canonically persisted file.
    selected_indices = sorted(
        {
            round(index * (len(records) - 1) / max(review_count - 1, 1))
            for index in range(review_count)
        }
    )
    selected = [records[index] for index in selected_indices]

    prompt_lengths = [len(item.messages[0].content) for item in records]
    target_lengths = [len(item.messages[1].content) for item in records]
    target_types = Counter(
        str(item.target.get("decision_type", item.operation)) for item in records
    )
    tool_names = Counter(
        str(call.get("tool_name"))
        for item in records
        if isinstance((call := item.target.get("tool_call")), dict)
    )
    forbidden = (
        "benchmarkgold",
        "benchmark_gold",
        '"split"',
        "source_group_id",
        "ground_truth",
        "evaluator_metadata",
        '"sha256"',
        '"uri"',
        "file://",
    )
    leak_ids = [
        item.record_id
        for item in records
        if any(value in item.messages[0].content.casefold() for value in forbidden)
    ]
    observation_target_ids = [
        item.record_id
        for item in records
        if '"event_type":"tool_observation"' in item.messages[1].content.casefold()
    ]
    review_packet = [
        {
            "record_id": item.record_id,
            "task_id": item.task_id,
            "operation": item.operation,
            "target_source": item.target_source,
            "decision_type": item.target.get("decision_type", item.operation),
            "tool_name": (
                item.target.get("tool_call", {}).get("tool_name")
                if isinstance(item.target.get("tool_call"), dict)
                else None
            ),
            "verification_evidence_count": len(item.verification_evidence_ids),
            "prompt_characters": len(item.messages[0].content),
            "target_characters": len(item.messages[1].content),
            "schema_valid": True,
            "policy_leak_detected": item.record_id in leak_ids,
            "tool_observation_is_target": item.record_id in observation_target_ids,
        }
        for item in selected
    ]
    audit = {
        "schema_version": "1.0.0",
        "dataset_path": args.dataset.as_posix(),
        "dataset_file_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "record_count": len(records),
        "unique_record_count": len({item.record_id for item in records}),
        "unique_model_io_count": len({item.model_io_sha256 for item in records}),
        "source_group_count": len({item.source_group_id for item in records}),
        "subsplit_counts": dict(Counter(item.train_subsplit for item in records)),
        "operation_counts": dict(Counter(item.operation for item in records)),
        "decision_type_counts": dict(target_types),
        "tool_counts": dict(tool_names),
        "target_source_counts": dict(Counter(item.target_source for item in records)),
        "prompt_character_length": {
            "min": min(prompt_lengths),
            "median": statistics.median(prompt_lengths),
            "p95": _percentile(prompt_lengths, 0.95),
            "max": max(prompt_lengths),
        },
        "target_character_length": {
            "min": min(target_lengths),
            "median": statistics.median(target_lengths),
            "p95": _percentile(target_lengths, 0.95),
            "max": max(target_lengths),
        },
        "exact_duplicate_model_io_count": len(records)
        - len({item.model_io_sha256 for item in records}),
        "policy_input_leak_count": len(leak_ids),
        "tool_observation_target_count": len(observation_target_ids),
        "review_packet_count": len(review_packet),
        "protected_split_access_count": 0,
    }
    _write(args.output, audit)
    _write(args.output.with_name("manual_review_packet.json"), review_packet)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
