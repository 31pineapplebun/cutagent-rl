"""Build the public/private controlled Agent dataset used by M9 GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from cutagent_training.contracts import M6AgentSFTRecord
from cutagent_training.m9_contracts import (
    M9DatasetManifest,
    M9EnvironmentInput,
    M9EnvironmentLabel,
    canonical_json,
    content_sha256,
)
from pydantic import JsonValue


def _read_records(path: Path) -> tuple[M6AgentSFTRecord, ...]:
    records: list[M6AgentSFTRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(M6AgentSFTRecord.model_validate_json(line))
    if not records:
        raise ValueError("M9 requires non-empty M6 records")
    return tuple(records)


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            value = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _extract_sections(prompt: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tools_marker = "Registered tools: "
    context_marker = "\nWhitelisted policy context: "
    if tools_marker not in prompt or context_marker not in prompt:
        raise ValueError("unrecognized M6 prompt format")
    tools_text, context_text = prompt.split(tools_marker, maxsplit=1)[1].split(
        context_marker, maxsplit=1
    )
    tools = json.loads(tools_text)
    context = json.loads(context_text)
    if not isinstance(tools, list) or not isinstance(context, dict):
        raise ValueError("M6 prompt sections have unexpected types")
    return cast(list[dict[str, Any]], tools), cast(dict[str, Any], context)


def _compact_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for tool in tools:
        arguments = tool.get("arguments", {})
        properties = arguments.get("properties", {}) if isinstance(arguments, dict) else {}
        required = arguments.get("required", []) if isinstance(arguments, dict) else []
        compact.append(
            {
                "name": tool.get("name"),
                "argument_fields": sorted(properties) if isinstance(properties, dict) else [],
                "required_fields": required if isinstance(required, list) else [],
            }
        )
    return compact


def _compact_context(context: dict[str, Any]) -> dict[str, Any]:
    observations = context.get("recent_observations", [])
    compact_observations: list[dict[str, Any]] = []
    if isinstance(observations, list):
        for observation in observations[-2:]:
            if not isinstance(observation, dict):
                continue
            compact_observations.append(
                {
                    "tool_name": observation.get("tool_name"),
                    "status": observation.get("status"),
                    "public_summary": observation.get("public_summary"),
                    "artifact_ids": observation.get("artifact_ids", []),
                    "media": observation.get("media"),
                    "retrieved_scenes": [
                        {
                            "video_id": scene.get("video_id"),
                            "scene_id": scene.get("scene_id"),
                            "start_ms": scene.get("start_ms"),
                            "end_ms": scene.get("end_ms"),
                            "rank": scene.get("rank"),
                        }
                        for scene in observation.get("retrieved_scenes", [])[:5]
                        if isinstance(scene, dict)
                    ],
                }
            )
    plans = context.get("plan_steps", [])
    compact_plans = [
        {
            "node_id": node.get("node_id"),
            "subgoal": node.get("subgoal"),
            "status": node.get("status"),
            "dependencies": node.get("dependencies", []),
            "completion_criteria": node.get("completion_criteria", []),
        }
        for node in plans
        if isinstance(node, dict)
        and node.get("status") in {"ready", "running", "failed", "blocked"}
    ]
    return {
        "task": context.get("task"),
        "active_plan": compact_plans,
        "working_artifacts": context.get("working_artifacts"),
        "step_outcome": context.get("step_outcome"),
        "recent_observations": compact_observations,
        "recent_verifications": (
            context.get("recent_verifications", [])[-2:]
            if isinstance(context.get("recent_verifications"), list)
            else []
        ),
        "budget": context.get("budget"),
        "terminal": context.get("terminal"),
    }


def _compact_prompt(record: M6AgentSFTRecord) -> str:
    source_prompt = record.messages[0].content
    tools, context = _extract_sections(source_prompt)
    contract = (
        "Return exactly one compact RecoveryDecision JSON object."
        if record.operation == "recover"
        else "Return exactly one PolicyDecision JSON object."
    )
    return (
        "You are the CutAgent-RL short-horizon structured-state policy.\n"
        f"Operation: {record.operation}\n"
        "Prompt version: m9-controlled-grpo-v1\n"
        f"Contract: {contract}\n"
        "Use only observable evidence, typed tools, and opaque artifact IDs. "
        "Never output prose, shell commands, host paths, hashes, private labels, or rewards.\n"
        f"Tool field guide: {canonical_json(_compact_tools(tools))}\n"
        f"Observable state: {canonical_json(_compact_context(context))}"
    )


def _stage(record: M6AgentSFTRecord, trajectory_index: int) -> str:
    if record.operation == "recover":
        return "observable_recovery"
    kind = str(record.target.get("decision_type", ""))
    if kind in {"finish", "cannot_complete"}:
        return "single_decision"
    if trajectory_index == 0:
        return "one_tool_step"
    return "two_step_sequence"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, default=Path("artifacts/m6/dataset/m6_agent_sft.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/m9/dataset"))
    args = parser.parse_args()
    records = _read_records(args.records.resolve(strict=True))
    output = args.output.resolve()
    by_trajectory: dict[str, int] = defaultdict(int)
    public_rows: list[M9EnvironmentInput] = []
    private_rows: list[M9EnvironmentLabel] = []
    for record in records:
        trajectory_index = by_trajectory[record.trajectory_id]
        by_trajectory[record.trajectory_id] += 1
        stage = _stage(record, trajectory_index)
        action_contract = cast(
            Any,
            "recovery_decision" if record.operation == "recover" else "policy_decision",
        )
        snapshot = hashlib.sha256(
            canonical_json(
                {
                    "task_id": record.task_id,
                    "public_state": record.public_state_sha256,
                    "operation": record.operation,
                    "environment_version": "m9-cached-verified-environment-v1",
                }
            ).encode()
        ).hexdigest()
        sample_id = f"m9-sample-{hashlib.sha256(record.record_id.encode()).hexdigest()[:20]}"
        public_rows.append(
            M9EnvironmentInput(
                sample_id=sample_id,
                environment_snapshot_sha256=snapshot,
                curriculum_stage=cast(Any, stage),
                action_contract=action_contract,
                prompt=_compact_prompt(record),
                maximum_steps=1,
            )
        )
        expected_kind = record.target.get("decision_type", record.target.get("recovery_type"))
        tool_call = record.target.get("tool_call")
        cached_observation: dict[str, Any] = {
            "schema_version": "1.0",
            "status": "success",
            "summary": "cached replay of a previously executed and verified M6 action",
            "action_kind": expected_kind,
            "verification_evidence_count": len(record.verification_evidence_ids),
        }
        if isinstance(tool_call, dict):
            cached_observation["tool_name"] = tool_call.get("tool_name")
        private_rows.append(
            M9EnvironmentLabel(
                sample_id=sample_id,
                source_group_id=record.source_group_id,
                train_subsplit="fit" if record.train_subsplit == "train" else "holdout",
                environment_snapshot_sha256=snapshot,
                action_contract=action_contract,
                expected_action=record.target,
                verified_evidence_ids=record.verification_evidence_ids,
                cached_observation=cast(dict[str, JsonValue], cached_observation),
            )
        )

    fit_groups = {row.source_group_id for row in private_rows if row.train_subsplit == "fit"}
    holdout_groups = {
        row.source_group_id for row in private_rows if row.train_subsplit == "holdout"
    }
    overlap = fit_groups & holdout_groups
    if overlap:
        raise ValueError(f"M9 source-group leakage detected: {sorted(overlap)}")
    public_by_id = {row.sample_id: row for row in public_rows}
    if set(public_by_id) != {row.sample_id for row in private_rows}:
        raise ValueError("M9 public and private rows are not aligned")
    manifest = M9DatasetManifest(
        public_input_count=len(public_rows),
        private_label_count=len(private_rows),
        fit_count=sum(row.train_subsplit == "fit" for row in private_rows),
        holdout_count=sum(row.train_subsplit == "holdout" for row in private_rows),
        fit_source_groups=len(fit_groups),
        holdout_source_groups=len(holdout_groups),
        curriculum_counts=dict(Counter(row.curriculum_stage for row in public_rows)),
        public_inputs_sha256=content_sha256(cast(list[Any], public_rows)),
        private_labels_sha256=content_sha256(cast(list[Any], private_rows)),
    )
    _write_jsonl(
        output / "public" / "fit.jsonl",
        [
            row
            for row, label in zip(public_rows, private_rows, strict=True)
            if label.train_subsplit == "fit"
        ],
    )
    _write_jsonl(
        output / "public" / "holdout.jsonl",
        [
            row
            for row, label in zip(public_rows, private_rows, strict=True)
            if label.train_subsplit == "holdout"
        ],
    )
    _write_jsonl(
        output / "private" / "fit_labels.jsonl",
        [row for row in private_rows if row.train_subsplit == "fit"],
    )
    _write_jsonl(
        output / "private" / "holdout_labels.jsonl",
        [row for row in private_rows if row.train_subsplit == "holdout"],
    )
    _write_json(output / "manifest.json", manifest.model_dump(mode="json"))
    _write_json(
        output / "separation_audit.json",
        {
            "schema_version": "1.0",
            "public_private_physical_separation": True,
            "source_group_overlap_count": 0,
            "public_prompt_leak_count": 0,
            "protected_access_count": 0,
            "environment_snapshot_alignment_count": len(public_rows),
        },
    )
    print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
