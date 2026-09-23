"""Private M4B validation split and recovery metric contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from cutagent_evaluation.m2b_dataset import HeldOutSceneGold, HeldOutVideoGold
from cutagent_evaluation.m4b_agent import (
    M4BTrajectoryEvaluation,
    _legacy_patch_outcomes,
    build_m4b_validation_set,
    summarize_m4b_evaluations,
)

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.agent import ReplanDecision
from cutagent.schemas.event import PlanNode, PlanPatch
from cutagent.schemas.media import TimeRange


def _records(path: Path) -> tuple[tuple[HeldOutVideoGold, ...], dict[str, ArtifactRef]]:
    records: list[HeldOutVideoGold] = []
    references: dict[str, ArtifactRef] = {}
    for offset in range(2):
        source_number = 7 + offset
        group = f"m2b-heldout-source-{source_number:02d}"
        record = HeldOutVideoGold(
            source_group_id=group,
            seed=700 + offset,
            source_sha256=f"{source_number:064x}",
            scenes=tuple(
                HeldOutSceneGold(
                    scene_index=scene_index,
                    nominal_time_range=TimeRange(
                        start_ms=scene_index * 3000,
                        end_ms=(scene_index + 1) * 3000,
                    ),
                    primary_entity=f"heldout object {source_number}",
                    companion_entity=f"companion {scene_index}",
                    action=("stationary", "appear", "move_right", "disappear")[scene_index],
                    ocr_text=f"M2B{source_number:02d} S{scene_index + 1}",
                    transcript=f"Heldout {source_number} scene {scene_index + 1} marker.",
                )
                for scene_index in range(4)
            ),
        )
        records.append(record)
        references[group] = ArtifactRef.from_path(
            path,
            artifact_id=f"video-{source_number}",
            media_type="application/json",
        )
    return tuple(records), references


def test_m4b_validation_has_40_disjoint_private_cases() -> None:
    records, references = _records(Path(__file__))
    m4a_groups = {f"m2b-heldout-source-{index:02d}" for index in range(1, 7)}
    cases = build_m4b_validation_set(
        records,
        references,
        excluded_source_groups=m4a_groups,
    )
    assert len(cases) == 40
    assert {case.gold.source_group_id for case in cases}.isdisjoint(m4a_groups)
    assert sum(case.gold.designed_recovery_opportunity for case in cases) >= 15
    public = "\n".join(case.task_input.model_dump_json() for case in cases).casefold()
    for forbidden in ("source_group_id", '"split"', "gold_id", "relevant_time_ranges"):
        assert forbidden not in public
    assert {case.gold.category for case in cases} == {
        "search_trim",
        "search_trim_validate",
        "retrieve_concat",
        "subtitle",
        "speed",
        "reframe",
        "multi_constraint",
        "hard_negative",
        "impossible",
    }


def test_m4b_validation_rejects_m4a_source_overlap() -> None:
    records, references = _records(Path(__file__))
    with pytest.raises(ValueError, match="overlap"):
        build_m4b_validation_set(
            records,
            references,
            excluded_source_groups={records[0].source_group_id},
        )


def _evaluation(*, task_id: str, success: bool) -> M4BTrajectoryEvaluation:
    return M4BTrajectoryEvaluation(
        task_id=task_id,
        run_id=f"run-{task_id}",
        variant="compact_recovery",
        category="search_trim",
        task_success=success,
        editing_task_success=success,
        correct_impossible_refusal=False,
        hard_constraints_satisfied=success,
        correct_final_artifact=success,
        structured_output_validity=1.0,
        model_output_failure=False,
        invalid_tool_call_rate=0.0,
        loop_or_stagnation=False,
        repeated_editor_rate=0.0,
        budget_exhausted=False,
        premature_finish=False,
        premature_refusal=False,
        agent_steps=5,
        tool_calls=3,
        search_calls=1,
        recovery_opportunity=True,
        recovery_attempted=True,
        recovery_decision_validity=1.0,
        recovery_succeeded=success,
        conditional_recovery_success=success,
        recovery_additional_steps=2,
        recovery_additional_latency_ms=100,
        replan_recovery_structured_validity=1.0,
        patch_acceptance_rate=1.0,
        primary_failure=None if success else "planning_error",
    )


def test_m4b_summary_reports_conditional_recovery() -> None:
    summary = summarize_m4b_evaluations(
        (_evaluation(task_id="case-1", success=True), _evaluation(task_id="case-2", success=False))
    )
    assert summary.task_success_rate == 0.5
    assert summary.recovery_opportunity_count == 2
    assert summary.recovery_attempt_rate == 1.0
    assert summary.recovery_success_rate == 0.5
    assert summary.conditional_recovery_success == 0.5
    assert summary.failure_origin_fraction["policy_model"] == 1.0


def test_legacy_patch_acceptance_uses_emitted_plan_revisions() -> None:
    accepted_patch = PlanPatch(
        revision=2,
        reason="retry local node",
        steps=(PlanNode(node_id="trim", subgoal="trim scene", status="ready"),),
    )
    rejected_patch = PlanPatch(
        revision=3,
        reason="unsafe rewrite",
        steps=(PlanNode(node_id="other", subgoal="rewrite history", status="ready"),),
    )
    decisions = [
        ReplanDecision(
            reason="retry",
            affected_plan_nodes=("trim",),
            requested_patch=accepted_patch,
        ),
        ReplanDecision(
            reason="rewrite",
            affected_plan_nodes=("other",),
            requested_patch=rejected_patch,
        ),
    ]
    accepted, rejected = _legacy_patch_outcomes(decisions, {2})
    assert accepted == 1
    assert rejected == ("runtime rejected invalid targeted legacy replan",)
