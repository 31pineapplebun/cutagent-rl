"""Build train-side same-snapshot M7 decision preference pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

from cutagent_training.contracts import M6AgentSFTRecord, canonical_json
from cutagent_training.m7_contracts import (
    M7DecisionPreference,
    PreferenceReason,
    build_m7_manifest,
    environment_snapshot_sha256,
)
from pydantic import JsonValue

from cutagent.schemas.agent import POLICY_DECISION_ADAPTER, FinishDecision, ToolDecision
from cutagent.schemas.m4b_agent import RECOVERY_DECISION_ADAPTER


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump(mode="json")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _premature_refusal() -> dict[str, JsonValue]:
    return {
        "schema_version": "1.0",
        "decision_type": "cannot_complete",
        "reason": (
            "Stop despite an executable public plan and available typed tools; this is a "
            "controlled premature refusal rather than a missing-capability conclusion."
        ),
        "missing_evidence_or_capability": ["none established by observable state"],
        "attempted_action_ids": [],
    }


def _premature_finish() -> dict[str, JsonValue]:
    return {
        "schema_version": "1.0",
        "decision_type": "finish",
        "output_artifact_id": "unverified-premature-artifact",
        "completion_summary": (
            "Claim completion before the required observable editing and validation criteria "
            "have been satisfied."
        ),
        "evidence_ids": ["unsupported-premature-evidence"],
    }


def _failure_aware_tool(chosen: ToolDecision) -> dict[str, JsonValue]:
    payload = chosen.model_dump(mode="json")
    arguments = cast(dict[str, JsonValue], payload["tool_call"]["arguments"])
    name = chosen.tool_call.tool_name
    if name == "search_video":
        arguments["query"] = "unrelated observable scene with a different entity and action"
        arguments["top_k"] = 1
    elif name == "trim_video":
        interval = cast(dict[str, JsonValue], arguments["time_range"])
        start = int(cast(int, interval["start_ms"]))
        end = int(cast(int, interval["end_ms"]))
        interval["end_ms"] = start + max(1, (end - start) // 2)
    elif name == "concat_videos":
        values = cast(list[JsonValue], arguments["input_artifact_ids"])
        arguments["input_artifact_ids"] = list(reversed(values))
    elif name == "change_speed":
        speed = float(cast(float, arguments["speed_factor"]))
        arguments["speed_factor"] = 0.5 if speed >= 1 else 2.0
    elif name == "add_subtitles":
        cues = cast(list[dict[str, JsonValue]], arguments["cues"])
        cues[0]["text"] = "incorrect controlled subtitle"
    elif name == "reframe_video":
        width = int(cast(int, arguments["width"]))
        height = int(cast(int, arguments["height"]))
        arguments["width"], arguments["height"] = height, width
    elif name == "validate_media":
        arguments["require_audio"] = not bool(arguments.get("require_audio", False))
    else:
        raise ValueError(f"unsupported M7 controlled tool perturbation: {name}")
    payload["rationale"] = (
        "Controlled schema-valid action that violates an observable task constraint."
    )
    payload["expected_observation"] = "A real ToolObservation for the wrong behavior."
    payload["success_condition"] = "The unrelated or altered constraint is applied."
    POLICY_DECISION_ADAPTER.validate_python(payload)
    return cast(dict[str, JsonValue], payload)


def _unnecessary_validation(chosen: FinishDecision) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "schema_version": "1.0",
        "decision_type": "tool",
        "tool_call": {
            "schema_version": "1.0",
            "tool_name": "validate_media",
            "tool_call_id": "controlled-repeat-validation",
            "arguments": {
                "schema_version": "1.0",
                "input_artifact_id": chosen.output_artifact_id,
                "decode_entire_video": True,
                "require_audio": False,
            },
        },
        "rationale": (
            "Repeat a completed validation even though runtime completion criteria are already "
            "satisfied and no new evidence can be gained."
        ),
        "expected_observation": "The same observable validation result repeats.",
        "success_condition": "No new completion evidence is added.",
    }
    POLICY_DECISION_ADAPTER.validate_python(payload)
    return payload


def _rejections(record: M6AgentSFTRecord) -> list[tuple[dict[str, JsonValue], PreferenceReason]]:
    if record.operation == "recover":
        chosen = RECOVERY_DECISION_ADAPTER.validate_python(record.target)
        node_id = getattr(chosen, "node_id", getattr(chosen, "affected_node_id", "active-node"))
        values: list[tuple[dict[str, JsonValue], PreferenceReason]] = [
            (
                {
                    "schema_version": "1.0",
                    "recovery_type": "cannot_recover",
                    "reason": "Refuse recovery although the observed failure is retryable.",
                    "missing_evidence_or_capability": ["none demonstrated"],
                },
                "invalid_recovery",
            ),
            (
                {
                    "schema_version": "1.0",
                    "recovery_type": "retry_current_node",
                    "node_id": f"unrelated-{node_id}",
                    "reason": "Retry an unrelated node and produce no useful local state change.",
                    "preferred_capability": None,
                },
                "invalid_recovery",
            ),
        ]
        for value, _ in values:
            RECOVERY_DECISION_ADAPTER.validate_python(value)
        return values
    decision = POLICY_DECISION_ADAPTER.validate_python(record.target)
    if isinstance(decision, ToolDecision):
        return [
            (_premature_refusal(), "premature_refusal"),
            (_premature_finish(), "premature_finish"),
            (_failure_aware_tool(decision), "wrong_tool_or_arguments"),
        ]
    if isinstance(decision, FinishDecision):
        return [
            (_premature_refusal(), "premature_refusal"),
            (_unnecessary_validation(decision), "repeated_action"),
        ]
    return [(_premature_finish(), "premature_finish")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m6-dataset", type=Path, default=Path("artifacts/m6/dataset/m6_agent_sft.jsonl")
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/m7/dataset"))
    args = parser.parse_args()
    records = tuple(
        M6AgentSFTRecord.model_validate_json(line)
        for line in args.m6_dataset.read_text(encoding="utf-8").splitlines()
        if line
    )
    pairs: list[M7DecisionPreference] = []
    for record in records:
        for index, (rejected, reason) in enumerate(_rejections(record), start=1):
            fingerprint = hashlib.sha256(
                canonical_json([record.record_id, record.target, rejected, reason, index]).encode()
            ).hexdigest()
            pairs.append(
                M7DecisionPreference(
                    pair_id=f"m7-pair-{fingerprint[:24]}",
                    source_record_id=record.record_id,
                    task_id=record.task_id,
                    source_group_id=record.source_group_id,
                    train_subsplit=record.train_subsplit,
                    operation=record.operation,
                    policy_context_sha256=record.policy_context_sha256,
                    public_state_sha256=record.public_state_sha256,
                    environment_snapshot_sha256=environment_snapshot_sha256(
                        task_id=record.task_id,
                        public_state_sha256=record.public_state_sha256,
                        policy_context_sha256=record.policy_context_sha256,
                    ),
                    prompt=record.messages[0].content,
                    chosen=record.target,
                    rejected=rejected,
                    preference_reason=reason,
                    label_source="executed_oracle_vs_controlled_failure",
                    rejected_evaluation="schema_valid_failure_aware_counterfactual",
                    score_margin=1.0,
                    confidence="high" if reason != "repeated_action" else "medium",
                )
            )
    result = tuple(pairs)
    manifest = build_m7_manifest(result)
    root = args.output_root
    _write_jsonl(root / "m7_preferences.jsonl", list(result))
    _write_jsonl(
        root / "dpo_train.jsonl",
        [item.swift_row() for item in result if item.train_subsplit == "train"],
    )
    _write_jsonl(
        root / "dpo_holdout.jsonl",
        [item.swift_row() for item in result if item.train_subsplit == "holdout"],
    )
    _write(root / "manifest.json", manifest.model_dump(mode="json"))
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
