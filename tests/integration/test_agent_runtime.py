"""Real ToolRegistry integration for both M4A Agent baselines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cutagent_evaluation.m4a_agent import (
    M4ATaskGold,
    evaluate_trajectory,
    summarize_evaluations,
)

from cutagent.agent.protocols import PolicyModelRequest, PolicyModelResult
from cutagent.agent.runtime import AgentRuntime
from cutagent.agent.state_reducer import StateReducer
from cutagent.agent.trajectory import TrajectoryStore
from cutagent.core.artifacts import ArtifactRef
from cutagent.core.errors import StructuredOutputError
from cutagent.schemas.agent import (
    AgentBaseline,
    AgentRuntimeConfig,
    AgentTrajectory,
    FinishDecision,
    PlanGraph,
    PolicyDecision,
    PolicyInferenceStats,
    ToolDecision,
)
from cutagent.schemas.event import PlanNode
from cutagent.schemas.media import TimeRange
from cutagent.schemas.task_input import DurationConstraint, TaskInput
from cutagent.schemas.tools import (
    TrimVideoArgs,
    TrimVideoCall,
    ValidateMediaArgs,
    ValidateMediaCall,
)
from cutagent.tools.errors import ToolFailure
from tests.tool_fixtures import build_tool_runtime, generate_tool_media


class ContextDrivenPolicy:
    """Deterministic policy test double; never used by production paths."""

    backend_version = "test-context-policy-v1"

    def __init__(self) -> None:
        self.requests: list[PolicyModelRequest] = []

    @staticmethod
    def _artifact(directory: Path, index: int) -> ArtifactRef:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"fake-output-{index}.json"
        path.write_text('{"synthetic_policy_test_double":true}\n', encoding="utf-8")
        return ArtifactRef.from_path(
            path,
            artifact_id=f"fake-policy-output-{index}",
            media_type="application/json",
        )

    @staticmethod
    def _plan() -> PlanGraph:
        return PlanGraph(
            revision=1,
            nodes=(
                PlanNode(
                    node_id="trim",
                    subgoal="Trim the requested interval.",
                    status="ready",
                    preferred_capability="media.write",
                    expected_evidence=("new clip artifact",),
                    completion_criteria=("trim succeeds",),
                ),
                PlanNode(
                    node_id="validate",
                    subgoal="Validate the edited clip.",
                    dependencies=("trim",),
                    preferred_capability="media.inspect",
                    expected_evidence=("decode check",),
                    completion_criteria=("media validates",),
                ),
            ),
        )

    def infer(self, request: PolicyModelRequest) -> PolicyModelResult:
        self.requests.append(request)
        index = len(self.requests)
        artifact = self._artifact(request.output_directory, index)
        if request.operation == "plan":
            return PolicyModelResult(
                operation="plan",
                decision=None,
                proposed_plan=self._plan(),
                raw_output_artifact=artifact,
                stats=PolicyInferenceStats(
                    operation="plan",
                    latency_ms=1,
                    input_tokens=10,
                    output_tokens=10,
                    repair_count=0,
                ),
            )
        context = json.loads(request.serialized_context)
        observations = context["recent_observations"]
        source_id = context["task"]["video"]["artifact_id"]
        output_id = next(
            (
                item["artifact_ids"][0]
                for item in observations
                if item["tool_name"] == "trim_video" and item["artifact_ids"]
            ),
            None,
        )
        decision: PolicyDecision
        if output_id is None:
            decision = ToolDecision(
                tool_call=TrimVideoCall(
                    tool_call_id=f"trim-{index}",
                    arguments=TrimVideoArgs(
                        input_artifact_id=source_id,
                        time_range=TimeRange(start_ms=500, end_ms=2000),
                    ),
                ),
                rationale="The public task specifies a 1.5 second clip.",
                expected_observation="A new opaque video artifact.",
                success_condition="Post-execution duration checks pass.",
            )
        elif not any(item["tool_name"] == "validate_media" for item in observations):
            decision = ToolDecision(
                tool_call=ValidateMediaCall(
                    tool_call_id=f"validate-{index}",
                    arguments=ValidateMediaArgs(input_artifact_id=output_id),
                ),
                rationale="The output must be independently decoded.",
                expected_observation="Successful full media validation.",
                success_condition="The artifact decodes and meets its media contract.",
            )
        else:
            decision = FinishDecision(
                output_artifact_id=output_id,
                completion_summary="The requested clip was created and validated.",
                evidence_ids=(source_id,),
            )
        return PolicyModelResult(
            operation=request.operation,
            decision=decision,
            proposed_plan=None,
            raw_output_artifact=artifact,
            stats=PolicyInferenceStats(
                operation=request.operation,
                latency_ms=1,
                input_tokens=10,
                output_tokens=10,
                repair_count=0,
            ),
        )


class RepeatedPolicy(ContextDrivenPolicy):
    def infer(self, request: PolicyModelRequest) -> PolicyModelResult:
        result = super().infer(request)
        if request.operation == "plan":
            return result
        context = json.loads(request.serialized_context)
        source_id = context["task"]["video"]["artifact_id"]
        decision = ToolDecision(
            tool_call=TrimVideoCall(
                tool_call_id=f"repeat-{len(self.requests)}",
                arguments=TrimVideoArgs(
                    input_artifact_id=source_id,
                    time_range=TimeRange(start_ms=500, end_ms=2000),
                ),
            ),
            rationale="Repeated action for loop-guard contract testing.",
            expected_observation="Same semantic result.",
            success_condition="This should be rejected by the loop guard.",
        )
        return result.__class__(
            operation=result.operation,
            decision=decision,
            proposed_plan=None,
            raw_output_artifact=result.raw_output_artifact,
            stats=result.stats,
        )


class InvalidOutputPolicy:
    backend_version = "test-invalid-output-policy-v1"

    def infer(self, request: PolicyModelRequest) -> PolicyModelResult:
        del request
        raise StructuredOutputError(
            "controlled invalid structured output",
            attempts=("not json", '{"still":"wrong"}'),
        )


def _run(tmp_path: Path, baseline: AgentBaseline) -> tuple[AgentRuntime, AgentTrajectory]:
    source_path = generate_tool_media(tmp_path / f"{baseline}.mp4")
    tools = build_tool_runtime(tmp_path / f"tools-{baseline}", source_path)
    task = TaskInput(
        task_id=f"task-{baseline}",
        video_ref=tools.source,
        instruction="Trim 500-2000 ms, validate it, and return the output.",
        user_constraints=(DurationConstraint(min_ms=1400, max_ms=1600),),
    )
    runtime = AgentRuntime(
        registry=tools.registry,
        policy_model=ContextDrivenPolicy(),
        artifact_root=tmp_path / f"agent-{baseline}",
    )
    trajectory = runtime.run(
        task,
        config=AgentRuntimeConfig(
            baseline=baseline,
            policy_view_mode="structured_state",
            max_wall_time_ms=120_000,
            max_model_tokens=10_000,
        ),
        run_id=f"run-{baseline}",
    )
    return runtime, trajectory


def test_react_and_hierarchical_execute_only_through_registry(tmp_path: Path) -> None:
    for baseline in ("react", "hierarchical"):
        runtime, trajectory = _run(tmp_path, baseline)
        assert trajectory.terminal_reason == "SUCCESS"
        assert [item.trace.tool_name for item in trajectory.tool_records] == [
            "trim_video",
            "validate_media",
        ]
        replayed = StateReducer.replay(trajectory.initial_state, trajectory.events)
        assert replayed == trajectory.final_state
        assert trajectory.final_output_artifact is not None
        runtime.registry.artifact_store.get(trajectory.final_output_artifact.artifact_id)
        reference = TrajectoryStore(tmp_path / "trajectories").write(trajectory)
        assert reference.media_type == "application/json"
        evaluated = evaluate_trajectory(
            trajectory,
            M4ATaskGold(
                gold_id=f"gold-task-{baseline}",
                task_id=f"task-{baseline}",
                source_group_id="private-source-group",
                category="search_trim_validate",
                expected_terminal_reason="SUCCESS",
                relevant_time_ranges=(TimeRange(start_ms=500, end_ms=2000),),
                required_tools=("trim_video", "validate_media"),
                expected_duration_ms=1500,
            ),
        )
        assert evaluated.task_success
        summary = summarize_evaluations((evaluated,))
        assert summary.task_success_rate == 1.0
        serialized = trajectory.model_dump_json().casefold()
        assert "private-source-group" not in serialized
        assert "source_group_id" not in serialized
        assert '"split"' not in serialized


def test_repeated_action_terminates_with_typed_loop_reason(tmp_path: Path) -> None:
    source_path = generate_tool_media(tmp_path / "loop.mp4")
    tools = build_tool_runtime(tmp_path / "loop-tools", source_path)
    runtime = AgentRuntime(
        registry=tools.registry,
        policy_model=RepeatedPolicy(),
        artifact_root=tmp_path / "loop-agent",
    )
    trajectory = runtime.run(
        TaskInput(
            task_id="task-loop",
            video_ref=tools.source,
            instruction="Keep trying the same trim for the loop test.",
        ),
        config=AgentRuntimeConfig(
            baseline="react",
            policy_view_mode="structured_state",
            max_repeated_identical_actions=1,
            max_wall_time_ms=120_000,
        ),
        run_id="run-loop",
    )
    assert trajectory.terminal_reason == "LOOP_DETECTED"
    assert trajectory.diagnostics[-1].diagnostic_type == "identical_tool_call"
    assert len(trajectory.tool_records) == 1


def test_policy_provenance_is_namespaced_by_run_id(tmp_path: Path) -> None:
    source_path = generate_tool_media(tmp_path / "namespaces.mp4")
    tools = build_tool_runtime(tmp_path / "namespaces-tools", source_path)
    policy = ContextDrivenPolicy()
    runtime = AgentRuntime(
        registry=tools.registry,
        policy_model=policy,
        artifact_root=tmp_path / "namespaces-agent",
    )
    task = TaskInput(
        task_id="task-namespaces",
        video_ref=tools.source,
        instruction="Trim 500-2000 ms, validate it, and return the output.",
        user_constraints=(DurationConstraint(min_ms=1400, max_ms=1600),),
    )
    config = AgentRuntimeConfig(
        baseline="react",
        policy_view_mode="structured_state",
        max_wall_time_ms=120_000,
    )
    first = runtime.run(task, config=config, run_id="namespace-run-a")
    second = runtime.run(task, config=config, run_id="namespace-run-b")
    assert first.terminal_reason == second.terminal_reason == "SUCCESS"
    assert {request.seed for request in policy.requests} == {20_260_823}
    assert (tmp_path / "namespaces-agent" / "policy" / "namespace-run-a").is_dir()
    assert (tmp_path / "namespaces-agent" / "policy" / "namespace-run-b").is_dir()


def test_failed_policy_output_is_linked_from_trajectory(tmp_path: Path) -> None:
    source_path = generate_tool_media(tmp_path / "invalid-policy.mp4")
    tools = build_tool_runtime(tmp_path / "invalid-policy-tools", source_path)
    runtime = AgentRuntime(
        registry=tools.registry,
        policy_model=InvalidOutputPolicy(),
        artifact_root=tmp_path / "invalid-policy-agent",
    )
    trajectory = runtime.run(
        TaskInput(
            task_id="task-invalid-policy",
            video_ref=tools.source,
            instruction="Exercise the invalid policy output trajectory contract.",
        ),
        config=AgentRuntimeConfig(
            baseline="react",
            policy_view_mode="structured_state",
            max_wall_time_ms=120_000,
        ),
        run_id="invalid-policy-run",
    )
    assert trajectory.terminal_reason == "MODEL_OUTPUT_FAILURE"
    assert len(trajectory.policy_failures) == 1
    failure = trajectory.policy_failures[0]
    assert failure.attempt_count == 2
    assert failure.repair_count == 1
    assert failure.context.context_sha256
    assert failure.raw_failure_artifact.media_type == "application/json"


def test_unknown_final_artifact_is_not_silently_accepted(tmp_path: Path) -> None:
    source_path = generate_tool_media(tmp_path / "unknown.mp4")
    tools = build_tool_runtime(tmp_path / "unknown-tools", source_path)
    with pytest.raises(ToolFailure):
        tools.store.get("not-an-artifact")
