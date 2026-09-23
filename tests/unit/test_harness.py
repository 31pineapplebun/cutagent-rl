"""The reduced harness retains existing safety and state-replay behavior."""

from dataclasses import replace
from pathlib import Path

import pytest

from cutagent.agent.harness import HarnessOutcomeBuilder, create_harness_registry, harness_config
from cutagent.agent.harness_retrieval import local_artifact_path
from cutagent.agent.m4b_protocols import M4BPolicyModelRequest, M4BPolicyModelResult
from cutagent.agent.m4b_runtime import M4BAgentRuntime
from cutagent.agent.m4b_state_reducer import M4BStateReducer
from cutagent.schemas.agent import FinishDecision
from cutagent.schemas.task_input import DurationConstraint, TaskInput
from tests.integration.test_m4b_agent_runtime import HandoffPolicy
from tests.tool_fixtures import generate_tool_media


class CitationRepairPolicy(HandoffPolicy):
    """Test-only reproduction of the observed model's bad-then-corrected citation."""

    def __init__(self, *, correct: bool) -> None:
        super().__init__()
        self.correct = correct
        self.finish_count = 0

    def infer(self, request: M4BPolicyModelRequest) -> M4BPolicyModelResult:
        result = super().infer(request)
        if isinstance(result.decision, FinishDecision):
            self.finish_count += 1
            if self.finish_count == 1 or not self.correct:
                return replace(
                    result,
                    decision=result.decision.model_copy(
                        update={"evidence_ids": ("unknown-evidence",)}
                    ),
                )
        return result


def test_prepared_artifacts_must_be_local(tmp_path: Path) -> None:
    assert local_artifact_path(tmp_path.as_uri()) == tmp_path.resolve()
    for uri in ("https://example.org/video.mp4", "file://remote-host/share/video.mp4"):
        with pytest.raises(ValueError, match="local files"):
            local_artifact_path(uri)


def test_same_budgets_and_small_tool_surface(tmp_path: Path) -> None:
    baseline = harness_config("handoff_only").model_dump(exclude={"protocol_variant"})
    enhanced = harness_config("compact_recovery").model_dump(exclude={"protocol_variant"})
    assert baseline == enhanced
    registry = create_harness_registry(tmp_path)
    assert {tool.name for tool in registry.manifest().tools} == {
        "inspect_media",
        "trim_video",
        "validate_media",
    }


def test_reduced_registry_real_trim_and_replay(tmp_path: Path) -> None:
    registry = create_harness_registry(tmp_path / "tools")
    source = registry.artifact_store.import_file(
        generate_tool_media(tmp_path / "source.mp4"), media_type="video/mp4"
    )
    task = TaskInput(
        task_id="harness-test",
        video_ref=source,
        instruction="Trim 500-2000 ms and validate.",
        user_constraints=(DurationConstraint(min_ms=1400, max_ms=1600),),
    )
    trajectory = M4BAgentRuntime(
        registry=registry, policy_model=HandoffPolicy(), artifact_root=tmp_path / "agent"
    ).run(task, config=harness_config("compact_recovery"), run_id="harness-test-run")
    assert trajectory.terminal_reason == "SUCCESS"
    assert trajectory.final_output_artifact is not None
    assert (
        M4BStateReducer.replay(trajectory.initial_state, trajectory.events)
        == trajectory.final_state
    )


@pytest.mark.parametrize("correct", [True, False])
def test_finish_citation_can_be_repaired_but_never_bypassed(tmp_path: Path, correct: bool) -> None:
    registry = create_harness_registry(tmp_path / "tools")
    source = registry.artifact_store.import_file(
        generate_tool_media(tmp_path / "source.mp4"), media_type="video/mp4"
    )
    trajectory = M4BAgentRuntime(
        registry=registry,
        policy_model=CitationRepairPolicy(correct=correct),
        artifact_root=tmp_path / "agent",
        outcome_builder=HarnessOutcomeBuilder(),
    ).run(
        TaskInput(
            task_id="citation-task",
            video_ref=source,
            instruction="Trim 500-2000 ms and validate.",
            user_constraints=(DurationConstraint(min_ms=1400, max_ms=1600),),
        ),
        config=harness_config("compact_recovery"),
        run_id="citation-run",
    )
    assert trajectory.terminal_reason == ("SUCCESS" if correct else "BUDGET_EXHAUSTED")
    assert len(trajectory.tool_records) == 2
    failures = [v for v in trajectory.final_state.verification_results if v.status == "failed"]
    assert failures
    assert "completion_evidence_exists" in failures[0].failure_types
    assert (
        M4BStateReducer.replay(trajectory.initial_state, trajectory.events)
        == trajectory.final_state
    )
    outcome = HarnessOutcomeBuilder().build(
        trajectory.initial_state,
        summary_index=1,
        active_node_id=None,
        latest_record=None,
        latest_verification=failures[0],
    )
    assert not outcome.completion_ready  # No validated media: finish stays forbidden.
