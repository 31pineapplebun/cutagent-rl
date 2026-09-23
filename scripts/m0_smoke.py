#!/usr/bin/env python3
"""Run the synthetic M0 infrastructure smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cutagent.agent.policy_context import PolicyContextBuilder, PolicyViewSerializer
from cutagent.agent.state_reducer import StateReducer
from cutagent.core.artifacts import ArtifactRef
from cutagent.core.run_context import RunContext
from cutagent.schemas.event import (
    AgentEvent,
    AgentEventEnvelope,
    BudgetUpdate,
    PlanPatch,
    PlanStep,
    RuntimeCheckResult,
    TerminalEvent,
    ToolObservation,
    VerificationResult,
)
from cutagent.schemas.state import AgentState, ExecutionBudget
from cutagent.schemas.task_input import (
    AspectRatioConstraint,
    DurationConstraint,
    OutputRequest,
    TaskInput,
)
from scripts.probe_environment import collect_environment, write_manifest

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _event(
    *,
    sequence_no: int,
    event: AgentEvent,
    emitted_by: str,
) -> AgentEventEnvelope:
    return AgentEventEnvelope.model_validate(
        {
            "event_id": f"event-{sequence_no}",
            "task_id": "task-m0-smoke",
            "sequence_no": sequence_no,
            "event": event.model_dump(mode="python"),
            "emitted_by": emitted_by,
            "created_at": BASE_TIME + timedelta(seconds=sequence_no),
            "parent_state_version": sequence_no - 1,
        }
    )


def _synthetic_sequence() -> tuple[AgentEventEnvelope, ...]:
    return (
        _event(
            sequence_no=1,
            emitted_by="planner",
            event=PlanPatch(
                revision=1,
                reason="M0 synthetic plan patch",
                steps=(
                    PlanStep(
                        node_id="step-contract",
                        subgoal="exercise infrastructure contracts",
                        status="running",
                    ),
                ),
            ),
        ),
        _event(
            sequence_no=2,
            emitted_by="tool",
            event=ToolObservation(
                call_id="call-synthetic",
                tool_name="synthetic_contract_tool",
                status="success",
                public_summary="synthetic contract observation completed",
                details={"synthetic": True},
            ),
        ),
        _event(
            sequence_no=3,
            emitted_by="verifier",
            event=VerificationResult(
                verification_id="verification-synthetic",
                status="passed",
                checks=(
                    RuntimeCheckResult(
                        check_name="contract_check",
                        passed=True,
                        summary="synthetic state is internally consistent",
                    ),
                ),
            ),
        ),
        _event(
            sequence_no=4,
            emitted_by="budget",
            event=BudgetUpdate(
                used_steps_delta=1,
                used_tool_calls_delta=1,
                used_wall_time_ms_delta=10,
            ),
        ),
        _event(
            sequence_no=5,
            emitted_by="runtime",
            event=TerminalEvent(
                status="succeeded",
                reason="M0 synthetic infrastructure sequence completed",
            ),
        ),
    )


def run_smoke(
    *, repository_root: Path, environment_output: Path, output: Path
) -> dict[str, object]:
    environment = collect_environment()
    write_manifest(environment, str(environment_output))
    environment_ref = ArtifactRef.from_path(
        environment_output,
        artifact_id="environment-m0-smoke",
        media_type="application/json",
    )

    task = TaskInput(
        task_id="task-m0-smoke",
        video_ref=ArtifactRef(
            artifact_id="video-synthetic",
            uri="artifact://synthetic/video",
            sha256="0" * 64,
            media_type="video/mp4",
            size_bytes=0,
        ),
        instruction="Exercise M0 contracts without model or tool inference.",
        user_constraints=(
            DurationConstraint(min_ms=1_000, max_ms=2_000),
            AspectRatioConstraint(width=9, height=16),
        ),
        requested_output=OutputRequest(container="mp4"),
    )
    initial_state = AgentState.initial(
        task,
        ExecutionBudget(max_steps=8, max_tool_calls=4, max_wall_time_ms=60_000),
    )
    events = _synthetic_sequence()
    final_state = StateReducer.replay(initial_state, events)

    policy_context = PolicyContextBuilder().build(final_state)
    serialized_policy_context = PolicyViewSerializer.to_dict(policy_context)

    run_context = RunContext.create(
        repository_root=repository_root,
        seed=42,
        config={"smoke_test": True, "event_count": len(events)},
        environment_manifest_ref=environment_ref,
        run_id="run-m0-smoke",
        created_at=BASE_TIME,
    )
    run_manifest = run_context.create_manifest(
        model_versions={"policy": "not-applicable-m0"},
        dataset_versions={"dataset": "not-applicable-m0"},
    )

    replayed_state = StateReducer.replay(initial_state, events)
    final_json = final_state.model_dump_json()
    replayed_json = replayed_state.model_dump_json()
    if final_json != replayed_json:
        raise RuntimeError("deterministic replay produced a different final state")

    result: dict[str, object] = {
        "schema_version": "1.0",
        "status": "passed",
        "synthetic_events_only": True,
        "event_count": len(events),
        "final_state_sha256": hashlib.sha256(final_json.encode("utf-8")).hexdigest(),
        "replay_identical": True,
        "policy_context": serialized_policy_context,
        "run_manifest": run_manifest.model_dump(mode="json"),
        "environment_manifest": str(environment_output.resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return result


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-output",
        type=Path,
        default=Path("artifacts/environment/m0_smoke_environment.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m0/smoke_result.json"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    repository_root = Path(__file__).resolve().parents[1]
    result = run_smoke(
        repository_root=repository_root,
        environment_output=args.environment_output,
        output=args.output,
    )
    print(
        f"M0 smoke test {result['status']}: replay_identical={result['replay_identical']} "
        f"output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
