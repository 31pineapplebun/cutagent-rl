"""Build leakage-audited M6 decision records from executed train trajectories."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Literal

from cutagent_training.contracts import (
    M6AgentSFTRecord,
    TrainingMessage,
    canonical_json,
)
from cutagent_training.prompts import render_m6_prompt

from cutagent.schemas.agent import CannotCompleteDecision, FinishDecision, ToolDecision
from cutagent.schemas.m4b_agent import M4BAgentTrajectory
from cutagent.schemas.tools import ToolManifest
from cutagent_evaluation.m5a_schemas import CutAgentBenchGold


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _target_source(
    decision: object, operation: str
) -> Literal[
    "executed_oracle",
    "verified_repair",
    "observable_recovery",
    "correct_cannot_complete",
    "valid_finish",
]:
    if operation == "recover":
        return "observable_recovery"
    if isinstance(decision, FinishDecision):
        return "valid_finish"
    if isinstance(decision, CannotCompleteDecision):
        return "correct_cannot_complete"
    if isinstance(decision, ToolDecision):
        return "executed_oracle"
    return "verified_repair"


def _verification_ids(
    trajectory: M4BAgentTrajectory,
    *,
    state_version: int,
) -> tuple[str, ...]:
    values: list[str] = []
    for envelope in trajectory.events[:state_version]:
        if envelope.event.event_type == "verification_result":
            values.append(envelope.event.verification_id)
    return tuple(dict.fromkeys(values[-3:]))


def records_from_trajectories(
    trajectories: Iterable[M4BAgentTrajectory],
    *,
    gold_by_task: dict[str, CutAgentBenchGold],
    holdout_source_groups: frozenset[str],
    tool_manifest: ToolManifest,
) -> tuple[M6AgentSFTRecord, ...]:
    records: list[M6AgentSFTRecord] = []
    for trajectory in trajectories:
        gold = gold_by_task[trajectory.task_input.task_id]
        expected_terminal = gold.expected_terminal_behavior.value
        if trajectory.terminal_reason != expected_terminal:
            continue
        for step in trajectory.policy_steps:
            if step.operation == "plan" or step.decision is None:
                continue
            context_payload = step.context.payload
            serialized_context = canonical_json(context_payload)
            prompt = render_m6_prompt(
                operation=step.operation,
                serialized_context=serialized_context,
                tool_manifest=tool_manifest,
            )
            target = step.decision.model_dump(mode="json")
            state_version = context_payload.get("state_version")
            if not isinstance(state_version, int) or isinstance(state_version, bool):
                raise ValueError("M6 context state_version is not an integer")
            fingerprint = _digest(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "state": step.context.context_sha256,
                    "target": target,
                }
            )
            record = M6AgentSFTRecord(
                record_id=f"m6-sft-{fingerprint[:24]}",
                trajectory_id=trajectory.trajectory_id,
                task_id=trajectory.task_input.task_id,
                source_group_id=gold.source_group_id,
                train_subsplit=(
                    "holdout" if gold.source_group_id in holdout_source_groups else "train"
                ),
                operation=step.operation,
                policy_context_sha256=step.context.context_sha256,
                public_state_sha256=_digest(context_payload),
                target_source=_target_source(step.decision, step.operation),
                verification_evidence_ids=_verification_ids(
                    trajectory,
                    state_version=state_version,
                ),
                messages=(
                    TrainingMessage(role="user", content=prompt),
                    TrainingMessage(role="assistant", content=canonical_json(target)),
                ),
                target=target,
            )
            records.append(record)
    identifiers = [item.record_id for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("M6 decision record IDs are not unique")
    return tuple(records)


def assert_executable_targets(
    records: Iterable[M6AgentSFTRecord],
    tool_manifest: ToolManifest,
) -> int:
    available = {item.name for item in tool_manifest.tools}
    count = 0
    for record in records:
        if record.operation == "decide" and record.target.get("decision_type") == "tool":
            call = record.target.get("tool_call")
            if not isinstance(call, dict) or call.get("tool_name") not in available:
                raise ValueError("M6 tool target is not executable by the frozen manifest")
            count += 1
        else:
            count += 1
    return count
