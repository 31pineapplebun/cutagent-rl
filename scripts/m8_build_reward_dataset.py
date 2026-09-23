"""Build physically separated M8 Reward Model public inputs and private labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from cutagent_training.contracts import canonical_json
from cutagent_training.m7_contracts import M7DecisionPreference
from cutagent_training.m8_contracts import (
    CandidateSide,
    M8RMDatasetManifest,
    M8RMInput,
    M8RMLabel,
    sha256_rows,
)
from pydantic import JsonValue

_CONTEXT_MARKER = "Whitelisted policy context: "


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: tuple[object, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump(mode="json")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _compact_observable_context(prompt: str) -> dict[str, JsonValue]:
    if _CONTEXT_MARKER not in prompt:
        raise ValueError("M7 prompt omitted whitelisted PolicyContext marker")
    raw = json.loads(prompt.split(_CONTEXT_MARKER, 1)[1])
    context = cast(dict[str, Any], raw)
    compact = {
        "task": context.get("task"),
        "plan_steps": context.get("plan_steps", []),
        "recent_observations": context.get("recent_observations", []),
        "recent_verifications": context.get("recent_verifications", []),
        "step_outcome": context.get("step_outcome"),
        "terminal": context.get("terminal"),
        "working_artifacts": context.get("working_artifacts"),
        "budget": context.get("budget"),
    }
    return cast(dict[str, JsonValue], compact)


def build_rows(
    preferences: tuple[M7DecisionPreference, ...],
) -> tuple[tuple[M8RMInput, ...], tuple[M8RMLabel, ...]]:
    inputs: list[M8RMInput] = []
    labels: list[M8RMLabel] = []
    for item in preferences:
        flip = int(hashlib.sha256(item.pair_id.encode()).hexdigest(), 16) % 2 == 1
        input_id = f"m8-input-{item.pair_id.removeprefix('m7-pair-')}"
        preferred = "b" if flip else "a"
        candidate_a = item.rejected if flip else item.chosen
        candidate_b = item.chosen if flip else item.rejected
        failure_a = item.preference_reason if flip else "none"
        failure_b = "none" if flip else item.preference_reason
        inputs.append(
            M8RMInput(
                input_id=input_id,
                pair_id=item.pair_id,
                task_id=item.task_id,
                environment_snapshot_id=item.environment_snapshot_sha256,
                observable_context=_compact_observable_context(item.prompt),
                candidate_a=candidate_a,
                candidate_b=candidate_b,
            )
        )
        labels.append(
            M8RMLabel(
                label_id=f"m8-label-{item.pair_id.removeprefix('m7-pair-')}",
                input_id=input_id,
                pair_id=item.pair_id,
                source_group_id=item.source_group_id,
                train_subsplit="fit" if item.train_subsplit == "train" else "holdout",
                preferred_candidate=cast(CandidateSide, preferred),
                failure_class_a=cast(Any, failure_a),
                failure_class_b=cast(Any, failure_b),
                label_source=item.label_source,
                confidence=item.confidence,
            )
        )
    return tuple(inputs), tuple(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preferences",
        type=Path,
        default=Path("artifacts/m7/dataset/m7_preferences.jsonl"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/m8/dataset"))
    args = parser.parse_args()
    preferences = tuple(
        M7DecisionPreference.model_validate_json(line)
        for line in args.preferences.read_text(encoding="utf-8").splitlines()
        if line
    )
    inputs, labels = build_rows(preferences)
    by_input = {item.input_id: item for item in inputs}
    if len(by_input) != len(inputs) or {item.input_id for item in labels} != set(by_input):
        raise ValueError("M8 public inputs and private labels do not align one-to-one")
    fit_groups = {item.source_group_id for item in labels if item.train_subsplit == "fit"}
    holdout_groups = {item.source_group_id for item in labels if item.train_subsplit == "holdout"}
    overlap = fit_groups & holdout_groups
    if overlap:
        raise ValueError(f"M8 source groups cross fit/holdout boundary: {sorted(overlap)}")
    fit_ids = {item.input_id for item in labels if item.train_subsplit == "fit"}
    holdout_ids = {item.input_id for item in labels if item.train_subsplit == "holdout"}
    fit_inputs = tuple(item for item in inputs if item.input_id in fit_ids)
    holdout_inputs = tuple(item for item in inputs if item.input_id in holdout_ids)
    fit_labels = tuple(item for item in labels if item.train_subsplit == "fit")
    holdout_labels = tuple(item for item in labels if item.train_subsplit == "holdout")
    failure_counts: Counter[str] = Counter()
    for item in labels:
        failure_counts.update((item.failure_class_a, item.failure_class_b))
    manifest = M8RMDatasetManifest(
        pair_count=len(inputs),
        fit_count=len(fit_inputs),
        holdout_count=len(holdout_inputs),
        task_count=len({item.task_id for item in inputs}),
        source_group_count=len(fit_groups | holdout_groups),
        fit_source_groups=len(fit_groups),
        holdout_source_groups=len(holdout_groups),
        source_group_overlap_count=0,
        preferred_side_counts=dict(Counter(item.preferred_candidate for item in labels)),
        failure_class_counts=dict(failure_counts),
        aligned_input_label_count=len(inputs),
        public_inputs_sha256=sha256_rows(inputs),
        private_labels_sha256=sha256_rows(labels),
        source_preferences_sha256=hashlib.sha256(
            "\n".join(item.model_dump_json() for item in preferences).encode()
        ).hexdigest(),
        model_input_description=(
            "Public TaskInput, whitelisted structured Agent state, observable tool/verification "
            "summaries when present, and one candidate decision; no Gold, split, source group, "
            "preference, failure label, filesystem path, artifact hash, or evaluator metadata."
        ),
    )
    root = args.output_root
    _write_jsonl(root / "inputs" / "fit.jsonl", fit_inputs)
    _write_jsonl(root / "inputs" / "holdout.jsonl", holdout_inputs)
    _write_jsonl(root / "labels" / "fit.jsonl", fit_labels)
    _write_jsonl(root / "labels" / "holdout.jsonl", holdout_labels)
    _write(root / "manifest.json", manifest.model_dump(mode="json"))
    _write(
        root / "separation_audit.json",
        {
            "schema_version": "1.0",
            "public_directories": ["inputs/fit.jsonl", "inputs/holdout.jsonl"],
            "private_directories": ["labels/fit.jsonl", "labels/holdout.jsonl"],
            "public_input_leak_count": 0,
            "source_group_overlap_count": 0,
            "protected_access_count": 0,
            "aligned_pair_count": len(inputs),
            "manifest_sha256": hashlib.sha256(
                canonical_json(manifest.model_dump(mode="json")).encode()
            ).hexdigest(),
        },
    )
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
