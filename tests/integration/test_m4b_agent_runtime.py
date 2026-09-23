"""ToolRegistry integration tests for M4B handoff and compact recovery."""

from __future__ import annotations

import json
from pathlib import Path

from cutagent.agent.m4b_protocols import M4BPolicyModelRequest, M4BPolicyModelResult
from cutagent.agent.m4b_runtime import M4BAgentRuntime
from cutagent.agent.m4b_state_reducer import M4BStateReducer
from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.agent import FinishDecision, PlanGraph, PolicyDecision, ToolDecision
from cutagent.schemas.event import PlanNode
from cutagent.schemas.m4b_agent import (
    M4BAgentTrajectory,
    M4BPolicyInferenceStats,
    M4BRuntimeConfig,
    RecoveryDecision,
    RetryCurrentNode,
)
from cutagent.schemas.media import TimeRange
from cutagent.schemas.task_input import DurationConstraint, TaskInput
from cutagent.schemas.tools import (
    TrimVideoArgs,
    TrimVideoCall,
    ValidateMediaArgs,
    ValidateMediaCall,
)
from tests.tool_fixtures import build_tool_runtime, generate_tool_media


class HandoffPolicy:
    backend_version = "test-m4b-handoff-policy-v1"

    def __init__(self, *, fail_first_trim: bool = False) -> None:
        self.fail_first_trim = fail_first_trim
        self.trim_attempts = 0
        self.requests: list[M4BPolicyModelRequest] = []

    @staticmethod
    def _artifact(directory: Path, index: int) -> ArtifactRef:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"fake-m4b-{index}.json"
        path.write_text('{"fake_policy":true}\n', encoding="utf-8")
        return ArtifactRef.from_path(
            path,
            artifact_id=f"fake-m4b-policy-{index}",
            media_type="application/json",
        )

    @staticmethod
    def _plan() -> PlanGraph:
        return PlanGraph(
            revision=1,
            nodes=(
                PlanNode(
                    node_id="trim",
                    subgoal="Trim the requested public interval.",
                    status="ready",
                    preferred_capability="media.write",
                    completion_criteria=("edited clip exists",),
                ),
                PlanNode(
                    node_id="validate",
                    subgoal="Independently validate current working media.",
                    dependencies=("trim",),
                    preferred_capability="media.inspect",
                    completion_criteria=("current media validates",),
                ),
            ),
        )

    def infer(self, request: M4BPolicyModelRequest) -> M4BPolicyModelResult:
        self.requests.append(request)
        artifact = self._artifact(request.output_directory, len(self.requests))
        plan = self._plan() if request.operation == "plan" else None
        decision: PolicyDecision | RecoveryDecision | None = None
        if request.operation == "recover":
            decision = RetryCurrentNode(
                node_id="trim",
                reason="retry with an interval inside the observable source duration",
            )
        elif request.operation == "replan":
            raise AssertionError("compact integration policy must not use legacy replan")
        elif request.operation == "decide":
            context = json.loads(request.serialized_context)
            working = context["working_artifacts"]
            current_id = working["current_working_media"]["artifact_id"]
            candidate_id = working["final_candidate_artifact_id"]
            if candidate_id is not None and context["step_outcome"]["completion_ready"]:
                decision = FinishDecision(
                    output_artifact_id=candidate_id,
                    completion_summary="Current working media is independently validated.",
                    evidence_ids=(context["task"]["video"]["artifact_id"],),
                )
            elif working["latest_generated_artifact_id"] is not None:
                decision = ToolDecision(
                    tool_call=ValidateMediaCall(
                        tool_call_id=f"validate-{len(self.requests)}",
                        arguments=ValidateMediaArgs(input_artifact_id=current_id),
                    ),
                    rationale="Validate the explicit current working artifact.",
                    expected_observation="Independent decode and metadata checks.",
                    success_condition="Current working media validates.",
                )
            else:
                self.trim_attempts += 1
                interval = (
                    TimeRange(start_ms=4500, end_ms=5500)
                    if self.fail_first_trim and self.trim_attempts == 1
                    else TimeRange(start_ms=500, end_ms=2000)
                )
                decision = ToolDecision(
                    tool_call=TrimVideoCall(
                        tool_call_id=f"trim-{len(self.requests)}",
                        arguments=TrimVideoArgs(
                            input_artifact_id=current_id,
                            time_range=interval,
                        ),
                    ),
                    rationale="Create the requested clip from observable media.",
                    expected_observation="A validated opaque output artifact.",
                    success_condition="The trim duration check passes.",
                )
        return M4BPolicyModelResult(
            operation=request.operation,
            decision=decision,
            proposed_plan=plan,
            raw_output_artifact=artifact,
            stats=M4BPolicyInferenceStats(
                operation=request.operation,
                latency_ms=1,
                input_tokens=25,
                output_tokens=15,
                repair_count=0,
            ),
        )


class ExplodingPolicy(HandoffPolicy):
    backend_version = "test-m4b-exploding-policy-v1"

    def infer(self, request: M4BPolicyModelRequest) -> M4BPolicyModelResult:
        if request.operation == "plan":
            return super().infer(request)
        raise ValueError("failure at /tmp/fixture/private password=do-not-persist")


def _run(tmp_path: Path, *, fail_first_trim: bool) -> tuple[HandoffPolicy, M4BAgentTrajectory]:
    source_path = generate_tool_media(tmp_path / "source.mp4")
    tools = build_tool_runtime(tmp_path / "tools", source_path)
    policy = HandoffPolicy(fail_first_trim=fail_first_trim)
    runtime = M4BAgentRuntime(
        registry=tools.registry,
        policy_model=policy,
        artifact_root=tmp_path / "agent",
    )
    trajectory = runtime.run(
        TaskInput(
            task_id="m4b-integration-task",
            video_ref=tools.source,
            instruction="Trim 500-2000 ms, validate it, and return only the output.",
            user_constraints=(DurationConstraint(min_ms=1400, max_ms=1600),),
        ),
        config=M4BRuntimeConfig(
            protocol_variant="compact_recovery",
            max_wall_time_ms=120_000,
            max_model_tokens=20_000,
        ),
        run_id=f"m4b-integration-{int(fail_first_trim)}",
    )
    return policy, trajectory


def test_handoff_runtime_completes_through_registry(tmp_path: Path) -> None:
    policy, trajectory = _run(tmp_path, fail_first_trim=False)
    assert trajectory.terminal_reason == "SUCCESS"
    assert [record.trace.tool_name for record in trajectory.tool_records] == [
        "trim_video",
        "validate_media",
    ]
    assert trajectory.final_state.step_outcomes[-1].completion_ready
    assert trajectory.final_state.working_artifacts.final_candidate_artifact is not None
    assert (
        M4BStateReducer.replay(trajectory.initial_state, trajectory.events)
        == trajectory.final_state
    )
    contexts = "\n".join(request.serialized_context for request in policy.requests).casefold()
    assert "source_group_id" not in contexts
    assert '"split"' not in contexts
    assert "sha256" not in contexts


def test_compact_recovery_converts_a_tool_failure_into_success(tmp_path: Path) -> None:
    policy, trajectory = _run(tmp_path, fail_first_trim=True)
    assert trajectory.terminal_reason == "SUCCESS"
    assert [request.operation for request in policy.requests].count("recover") == 1
    assert len(trajectory.final_state.recovery_events) == 1
    recovery = trajectory.final_state.recovery_events[0]
    assert recovery.accepted
    assert recovery.generated_plan_revision is not None
    assert trajectory.tool_records[0].observation.status == "invalid"
    assert trajectory.tool_records[1].observation.status == "success"
    assert (
        M4BStateReducer.replay(trajectory.initial_state, trajectory.events)
        == trajectory.final_state
    )


def test_runtime_persists_only_sanitized_system_error_detail(tmp_path: Path) -> None:
    source_path = generate_tool_media(tmp_path / "system-error.mp4")
    tools = build_tool_runtime(tmp_path / "system-error-tools", source_path)
    runtime = M4BAgentRuntime(
        registry=tools.registry,
        policy_model=ExplodingPolicy(),
        artifact_root=tmp_path / "system-error-agent",
    )
    trajectory = runtime.run(
        TaskInput(
            task_id="m4b-system-error-task",
            video_ref=tools.source,
            instruction="Exercise safe error persistence.",
        ),
        config=M4BRuntimeConfig(
            protocol_variant="compact_recovery",
            max_wall_time_ms=120_000,
        ),
        run_id="m4b-system-error-run",
    )
    assert trajectory.terminal_reason == "SYSTEM_ERROR"
    diagnostic = trajectory.final_state.system_diagnostics[0]
    assert diagnostic.error_category == "runtime_validation_error"
    serialized = diagnostic.model_dump_json().casefold()
    assert "/tmp/fixture/" not in serialized
    assert "password" not in serialized
    assert "do-not-persist" not in serialized
