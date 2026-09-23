"""M4A policy, trajectory, and tool-execution isolation tests."""

from __future__ import annotations

import inspect
import json

from cutagent.agent.policy_context import PolicyContextBuilder, PolicyViewSerializer
from cutagent.agent.runtime import AgentRuntime
from cutagent.schemas.event import PlanNode


def test_plan_expected_evidence_is_public_but_private_labels_are_absent(initial_state) -> None:
    state = initial_state.model_copy(
        update={
            "plan_revision": 1,
            "plan_steps": (
                PlanNode(
                    node_id="inspect",
                    subgoal="Inspect observable media.",
                    status="ready",
                    expected_evidence=("observable media dimensions",),
                ),
            ),
        }
    )
    serialized = PolicyViewSerializer.to_json(PolicyContextBuilder().build(state))
    assert "observable media dimensions" in serialized
    lowered = serialized.casefold()
    for forbidden in (
        "benchmarkgold",
        "benchmark_gold",
        "source_group_id",
        '"split"',
        "ground_truth",
        "evaluator_metadata",
    ):
        assert forbidden not in lowered


def test_agent_runtime_has_one_auditable_tool_execution_boundary() -> None:
    source = inspect.getsource(AgentRuntime)
    assert source.count("self.registry.execute(") == 1
    assert "subprocess" not in source
    assert "shell=True" not in source
    assert "ffmpeg" not in source.casefold()


def test_policy_context_snapshot_contains_no_host_paths(initial_state) -> None:
    payload = PolicyViewSerializer.to_dict(PolicyContextBuilder().build(initial_state))
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "file://" not in serialized
    assert initial_state.task_input.video_ref.sha256 not in serialized
    assert initial_state.task_input.video_ref.uri not in serialized
