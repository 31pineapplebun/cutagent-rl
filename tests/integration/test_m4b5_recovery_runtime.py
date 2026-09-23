"""M4B.5 observable failure to compact-recovery integration through ToolRegistry."""

from __future__ import annotations

from pathlib import Path

from cutagent_evaluation.m4b5_recovery import (
    DeterministicFailureInjectingRegistry,
    FailureInjectionConfig,
    M4B5TaskGold,
    evaluate_m4b5_trajectory,
    finalize_trigger,
    summarize_m4b5_evaluations,
    trajectory_has_private_injection_data,
)
from cutagent_evaluation.m4b_agent import M4BTaskGold

from cutagent.agent.m4b_runtime import M4BAgentRuntime
from cutagent.agent.m4b_state_reducer import M4BStateReducer
from cutagent.schemas.m4b_agent import M4BRuntimeConfig
from cutagent.schemas.media import TimeRange
from cutagent.schemas.task_input import DurationConstraint, TaskInput
from tests.integration.test_m4b_agent_runtime import HandoffPolicy
from tests.tool_fixtures import build_tool_runtime, generate_tool_media


def test_real_registry_failure_triggers_recovery_and_replays(tmp_path: Path) -> None:
    source_path = generate_tool_media(tmp_path / "source.mp4")
    tools = build_tool_runtime(tmp_path / "tools", source_path)
    config = FailureInjectionConfig(
        injection_id="failure-integration-timeout",
        task_id="m4b5-integration-task",
        failure_type="tool_timeout",
        trigger_mode="pre_execute_failure",
        trigger_tool_names=("trim_video",),
        expected_recovery_operations=("retry_current_node",),
    )
    injected = DeterministicFailureInjectingRegistry(tools.registry, config)
    policy = HandoffPolicy()
    runtime = M4BAgentRuntime(
        registry=injected,
        policy_model=policy,
        artifact_root=tmp_path / "agent",
    )
    trajectory = runtime.run(
        TaskInput(
            task_id=config.task_id,
            video_ref=tools.source,
            instruction="Trim 500-2000 ms, validate it, and return only the output.",
            user_constraints=(DurationConstraint(min_ms=1400, max_ms=1600),),
        ),
        config=M4BRuntimeConfig(
            protocol_variant="compact_recovery",
            max_wall_time_ms=120_000,
            max_model_tokens=20_000,
        ),
        run_id="m4b5-integration-run",
    )
    trigger = finalize_trigger(injected.private_trigger(), trajectory)
    assert trigger.triggered
    assert trigger.trigger_source == "tool_observation"
    assert trigger.active_plan_node == "trim"
    assert [item.operation for item in trajectory.policy_steps].count("recover") == 1
    assert trajectory.final_state.recovery_events[0].accepted
    assert trajectory.tool_records[0].observation.status == "timeout"
    assert trajectory.tool_records[1].observation.status == "success"
    assert trajectory.terminal_reason == "SUCCESS"
    assert (
        M4BStateReducer.replay(trajectory.initial_state, trajectory.events)
        == trajectory.final_state
    )
    assert not trajectory_has_private_injection_data(trajectory)
    gold = M4B5TaskGold(
        gold_id="private-m4b5-integration-task",
        task_id=config.task_id,
        source_group_id="m4b5-integration-source",
        task_gold=M4BTaskGold(
            gold_id="gold-m4b5-integration-task",
            task_id=config.task_id,
            source_group_id="m4b5-integration-source",
            category="search_trim_validate",
            expected_terminal_reason="SUCCESS",
            relevant_time_ranges=(TimeRange(start_ms=500, end_ms=2000),),
            required_tools=("trim_video", "validate_media"),
            expected_duration_ms=1500,
            designed_recovery_opportunity=True,
        ),
        injection=config,
    )
    evaluation = evaluate_m4b5_trajectory(
        trajectory,
        gold,
        trigger,
        variant="compact_recovery",
    )
    summary = summarize_m4b5_evaluations((evaluation,))
    assert evaluation.maximum_recovery_level == 5
    assert evaluation.recovery_operation == "retry_current_node"
    assert summary.overall.cumulative_level_counts == {level: 1 for level in range(6)}
    assert summary.overall.recovery_success_rate == 1.0
    recovery_context = next(
        item.serialized_context for item in policy.requests if item.operation == "recover"
    ).casefold()
    assert config.injection_id.casefold() not in recovery_context
    assert config.failure_type.casefold() not in recovery_context
    assert "expected_recovery_operations" not in recovery_context
    assert "retry_current_node" not in recovery_context
    unrelated = trajectory.tool_records[-1]
    colliding_observation = unrelated.observation.model_copy(
        update={"call_id": trigger.trigger_tool_call_id}
    )
    collision_trajectory = trajectory.model_copy(
        update={
            "tool_records": (
                unrelated.model_copy(update={"observation": colliding_observation}),
                *trajectory.tool_records,
            )
        }
    )
    collision_evaluation = evaluate_m4b5_trajectory(
        collision_trajectory,
        gold,
        trigger,
        variant="compact_recovery",
    )
    assert collision_evaluation.maximum_recovery_level == 5
