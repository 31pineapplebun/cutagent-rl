"""M4A planning, decision, loop, and trajectory contract tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cutagent.agent.loop_detection import LoopDetector
from cutagent.agent.planning import patch_node_status, validate_targeted_patch
from cutagent.schemas.agent import POLICY_DECISION_ADAPTER, PlanGraph
from cutagent.schemas.event import PlanNode, VerificationResult
from cutagent.schemas.tools import TOOL_CALL_ADAPTER


def _graph() -> PlanGraph:
    return PlanGraph(
        revision=1,
        nodes=(
            PlanNode(
                node_id="find",
                subgoal="Find the target scene.",
                status="ready",
                expected_evidence=("retrieved scene",),
                preferred_capability="retrieval.read",
                completion_criteria=("one supported scene",),
            ),
            PlanNode(
                node_id="edit",
                subgoal="Trim the target scene.",
                dependencies=("find",),
                expected_evidence=("output artifact",),
                completion_criteria=("valid clip",),
            ),
        ),
    )


def test_policy_decision_union_is_strict_and_discriminated() -> None:
    decision = POLICY_DECISION_ADAPTER.validate_python(
        {
            "decision_type": "tool",
            "tool_call": {
                "tool_name": "trim_video",
                "tool_call_id": "trim-001",
                "arguments": {
                    "input_artifact_id": "source-001",
                    "time_range": {"start_ms": 100, "end_ms": 900},
                },
            },
            "rationale": "The requested scene range is known.",
            "expected_observation": "A new clip artifact.",
            "success_condition": "The clip is decodable.",
        }
    )
    assert decision.decision_type == "tool"
    with pytest.raises(ValidationError):
        POLICY_DECISION_ADAPTER.validate_python(
            {
                "decision_type": "tool",
                "tool_call": {
                    "tool_name": "trim_video",
                    "tool_call_id": "trim-002",
                    "arguments": {
                        "input_artifact_id": "source-001",
                        "time_range": {"start_ms": 100, "end_ms": 900},
                    },
                    "command": "ffmpeg -i secret",
                },
                "rationale": "invalid",
                "expected_observation": "invalid",
                "success_condition": "invalid",
            }
        )


def test_plan_graph_rejects_cycle_and_preserves_successful_history() -> None:
    with pytest.raises(ValidationError, match="acyclic"):
        PlanGraph(
            revision=1,
            nodes=(
                PlanNode(node_id="a", subgoal="A", dependencies=("b",)),
                PlanNode(node_id="b", subgoal="B", dependencies=("a",)),
            ),
        )
    succeeded_patch = patch_node_status(_graph(), "find", "succeeded", reason="done")
    succeeded = PlanGraph(revision=2, nodes=succeeded_patch.steps)
    rewritten = tuple(
        node.model_copy(update={"subgoal": "Rewrite forbidden"}) if node.node_id == "find" else node
        for node in succeeded.nodes
    )
    with pytest.raises(ValueError, match="successful"):
        validate_targeted_patch(
            succeeded,
            succeeded.as_patch(reason="wrong").model_copy(
                update={"revision": 3, "steps": rewritten}
            ),
            ("find",),
        )


def test_loop_detector_counts_normalized_actions_not_call_ids() -> None:
    detector = LoopDetector(maximum_identical_actions=2)
    for index in range(2):
        call = TOOL_CALL_ADAPTER.validate_python(
            {
                "tool_name": "inspect_media",
                "tool_call_id": f"inspect-{index}",
                "arguments": {"input_artifact_id": "source-001"},
            }
        )
        assert detector.observe_tool_call(call) is None
    third = TOOL_CALL_ADAPTER.validate_python(
        {
            "tool_name": "inspect_media",
            "tool_call_id": "inspect-3",
            "arguments": {"input_artifact_id": "source-001"},
        }
    )
    diagnostic = detector.observe_tool_call(third)
    assert diagnostic is not None
    assert diagnostic.diagnostic_type == "identical_tool_call"


def test_loop_detector_repeated_verification_failure() -> None:
    detector = LoopDetector(maximum_identical_actions=1)
    result = VerificationResult(
        verification_id="verify-1",
        status="failed",
        failure_types=("invalid_interval",),
    )
    assert detector.observe_verification(result) is None
    diagnostic = detector.observe_verification(
        result.model_copy(update={"verification_id": "verify-2"})
    )
    assert diagnostic is not None
    assert diagnostic.diagnostic_type == "repeated_verification_failure"


def test_datetime_is_timezone_aware_fixture_sanity() -> None:
    assert datetime.now(UTC).utcoffset() is not None
