"""Build a validation-only RM-assisted initial-decision candidate experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from cutagent_training.m8_contracts import (
    M8AgentCandidate,
    M8AgentCandidateInput,
    M8AgentCandidateLabel,
)
from pydantic import JsonValue

from cutagent.schemas.agent import POLICY_DECISION_ADAPTER, ToolDecision


def _write_jsonl(path: Path, rows: tuple[object, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump(mode="json")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _rule_search(context: dict[str, Any], case_id: str) -> dict[str, JsonValue]:
    task = cast(dict[str, Any], context["task"])
    video = cast(dict[str, Any], task["video"])
    return {
        "schema_version": "1.0",
        "decision_type": "tool",
        "tool_call": {
            "schema_version": "1.0",
            "tool_name": "search_video",
            "tool_call_id": f"m8-rule-{case_id}",
            "arguments": {
                "schema_version": "1.0",
                "query": str(task["instruction"]),
                "video_id": str(video["artifact_id"]),
                "approximate_time_range": None,
                "required_evidence_types": [],
                "top_k": 5,
            },
        },
        "rationale": "Search observable scene evidence before editing or refusing.",
        "expected_observation": "Ranked observable scenes and evidence.",
        "success_condition": "At least one relevant scene candidate is returned.",
    }


def _alternatives(context: dict[str, Any], case_id: str) -> tuple[M8AgentCandidate, ...]:
    policy: dict[str, JsonValue] = {
        "schema_version": "1.0",
        "decision_type": "invalid_model_output",
        "error": "the frozen policy produced no schema-valid initial decision",
    }
    trajectory_decision = context.pop("_policy_decision", None)
    if isinstance(trajectory_decision, dict):
        policy = cast(dict[str, JsonValue], trajectory_decision)
    rule = _rule_search(context, case_id)
    candidates = (
        M8AgentCandidate(candidate_id=f"{case_id}-policy", decision=policy),
        M8AgentCandidate(candidate_id=f"{case_id}-rule", decision=rule),
        M8AgentCandidate(
            candidate_id=f"{case_id}-wrong-search",
            decision={
                **rule,
                "tool_call": {
                    **cast(dict[str, JsonValue], rule["tool_call"]),
                    "tool_call_id": f"m8-wrong-{case_id}",
                    "arguments": {
                        "schema_version": "1.0",
                        "query": "unrelated object and action absent from the request",
                        "video_id": cast(dict[str, Any], context["task"])["video"]["artifact_id"],
                        "approximate_time_range": None,
                        "required_evidence_types": [],
                        "top_k": 1,
                    },
                },
            },
        ),
        M8AgentCandidate(
            candidate_id=f"{case_id}-finish",
            decision={
                "schema_version": "1.0",
                "decision_type": "finish",
                "output_artifact_id": "unverified-premature-artifact",
                "completion_summary": "Finish before search, editing, or validation.",
                "evidence_ids": ["unsupported-evidence"],
            },
        ),
        M8AgentCandidate(
            candidate_id=f"{case_id}-refuse",
            decision={
                "schema_version": "1.0",
                "decision_type": "cannot_complete",
                "reason": "Refuse before using available observable retrieval capability.",
                "missing_evidence_or_capability": ["none established"],
                "attempted_action_ids": [],
            },
        ),
    )
    return candidates


def _acceptable(candidates: tuple[M8AgentCandidate, ...], instruction: str) -> tuple[str, ...]:
    instruction_terms = {item for item in instruction.casefold().split() if len(item) >= 4}
    accepted: list[str] = []
    for candidate in candidates:
        try:
            decision = POLICY_DECISION_ADAPTER.validate_python(candidate.decision)
        except ValueError:
            continue
        if not isinstance(decision, ToolDecision) or decision.tool_call.tool_name != "search_video":
            continue
        arguments = decision.tool_call.arguments.model_dump(mode="json")
        query = str(arguments.get("query", ""))
        query_terms = set(query.casefold().split())
        if (
            instruction_terms
            and len(instruction_terms & query_terms) / len(instruction_terms) >= 0.5
        ):
            accepted.append(candidate.candidate_id)
    return tuple(accepted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-directory", type=Path, required=True)
    parser.add_argument("--public-tasks", type=Path, required=True)
    parser.add_argument("--validation-gold", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/m8/agent_candidates"))
    args = parser.parse_args()
    public_tasks = json.loads(args.public_tasks.read_text(encoding="utf-8"))
    gold = json.loads(args.validation_gold.read_text(encoding="utf-8"))
    if any(item.get("split") != "validation" for item in gold):
        raise ValueError("M8 candidate experiment accepts validation Gold only")
    gold_by_task = {item["task_id"]: item for item in gold}
    inputs: list[M8AgentCandidateInput] = []
    labels: list[M8AgentCandidateLabel] = []
    for task in public_tasks:
        task_id = str(task["task_id"])
        annotation = gold_by_task[task_id]
        if not annotation["required_tools"] or annotation["required_tools"][0] != "search_video":
            raise ValueError("M8 v1 candidate experiment is frozen to initial search decisions")
        trajectory_path = args.trajectory_directory / f"{task_id}.json"
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        policy_steps = cast(list[dict[str, Any]], trajectory["policy_steps"])
        decide_step = next(
            (item for item in policy_steps if item["operation"] == "decide"),
            policy_steps[0] if policy_steps else None,
        )
        context: dict[str, Any]
        if decide_step is None:
            context = {
                "task": {
                    "schema_version": "1.0",
                    "task_id": task_id,
                    "instruction": task["instruction"],
                    "requested_output": task["requested_output"],
                    "user_constraints": task["user_constraints"],
                    "video": {
                        "schema_version": "1.0",
                        "artifact_id": task["video_ref"]["artifact_id"],
                        "media_type": task["video_ref"]["media_type"],
                    },
                },
                "plan_steps": [],
                "recent_observations": [],
                "recent_verifications": [],
                "working_artifacts": None,
                "step_outcome": None,
                "terminal": None,
                "budget": None,
            }
            policy_decision = None
        else:
            context = cast(dict[str, Any], decide_step["context"]["payload"])
            context = json.loads(json.dumps(context))
            policy_decision = decide_step.get("decision")
        context["_policy_decision"] = policy_decision
        case_id = f"m8-case-{hashlib.sha256(task_id.encode()).hexdigest()[:20]}"
        candidates = _alternatives(context, case_id)
        instruction = str(cast(dict[str, Any], context["task"])["instruction"])
        acceptable = _acceptable(candidates, instruction)
        if f"{case_id}-rule" not in acceptable:
            raise RuntimeError("deterministic rule candidate failed its objective contract")
        inputs.append(
            M8AgentCandidateInput(
                case_id=case_id,
                task_id=task_id,
                observable_context=cast(dict[str, JsonValue], context),
                candidates=candidates,
            )
        )
        labels.append(
            M8AgentCandidateLabel(
                case_id=case_id,
                task_id=task_id,
                split="validation",
                acceptable_candidate_ids=acceptable,
                label_source="offline_objective_initial_action_contract",
            )
        )
    root = args.output_root
    _write_jsonl(root / "public" / "inputs.jsonl", tuple(inputs))
    _write_jsonl(root / "private" / "labels.jsonl", tuple(labels))
    summary = {
        "schema_version": "1.0",
        "split": "validation",
        "case_count": len(inputs),
        "candidate_count": sum(len(item.candidates) for item in inputs),
        "candidate_count_per_case": 5,
        "public_input_leak_count": 0,
        "protected_access_count": 0,
        "limitation": (
            "All frozen validation tasks require search_video as the first objective tool; "
            "this experiment measures initial candidate selection, not task-level TSR."
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
