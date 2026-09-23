"""Offline acceptance rejects wrong content, and preserves the 12/6/2 denominators."""

from collections import Counter
from pathlib import Path

import pytest
from cutagent_evaluation.harness_regression import development_cases, score_run
from cutagent_evaluation.m4b5_recovery import (
    DeterministicFailureInjectingRegistry,
    FailureInjectionConfig,
)

from cutagent.agent.harness import HarnessOutcomeBuilder, create_harness_registry, harness_config
from cutagent.agent.m4b_protocols import M4BPolicyModelRequest
from cutagent.agent.m4b_runtime import M4BAgentRuntime
from cutagent.models.harness_policy import HarnessPolicyBackend
from cutagent.models.qwen_policy_m4b import Qwen3VLPolicyBackendM4B
from cutagent.schemas.media import TimeRange
from cutagent.schemas.task_input import DurationConstraint, TaskInput
from scripts.harness_regression import freeze, run, write_json
from tests.integration.test_m4b_agent_runtime import HandoffPolicy
from tests.tool_fixtures import generate_tool_media


def test_predeclared_cases_and_no_category_in_public_ids() -> None:
    cases = development_cases()
    assert Counter(c.category for c in cases) == {"normal": 12, "injected": 6, "impossible": 2}
    assert len({c.task_id for c in cases}) == 20
    assert sum(c.requires_retrieval for c in cases) == 4
    assert all(c.task_id.startswith("harness-task-") for c in cases)


def test_citation_instruction_shared_and_parser_unchanged(tmp_path: Path) -> None:
    registry = create_harness_registry(tmp_path)
    backend = HarnessPolicyBackend(model_cache=tmp_path)
    for variant in ("handoff_only", "compact_recovery"):
        request = M4BPolicyModelRequest(
            operation="decide",
            protocol_variant=variant,
            serialized_context="{}",
            tool_manifest=registry.manifest(),
            output_directory=tmp_path,
            maximum_new_tokens=768,
            maximum_repairs=2,
            prompt_template_version="harness-policy-v1",
            seed=20260923,
        )
        assert "1-3 UNIQUE" in backend._prompt(request)
    assert HarnessPolicyBackend._parse is Qwen3VLPolicyBackendM4B._parse


def test_media_and_interval_scoring_with_real_injected_tool_recovery(tmp_path: Path) -> None:
    registry = create_harness_registry(tmp_path / "tools")
    source = registry.artifact_store.import_file(
        generate_tool_media(tmp_path / "source.mp4"), media_type="video/mp4"
    )
    case = development_cases()[13]  # Injected request for 500-2000 ms.
    injected = DeterministicFailureInjectingRegistry(
        registry,
        FailureInjectionConfig(
            injection_id="test-timeout",
            task_id=case.task_id,
            failure_type="tool_timeout",
            trigger_mode="pre_execute_failure",
            trigger_tool_names=("trim_video",),
            expected_recovery_operations=("retry_current_node",),
        ),
    )
    trajectory = M4BAgentRuntime(
        registry=injected,
        policy_model=HandoffPolicy(),
        artifact_root=tmp_path / "agent",
        outcome_builder=HarnessOutcomeBuilder(),
    ).run(
        TaskInput(
            task_id=case.task_id,
            video_ref=source,
            instruction=case.instruction,
            user_constraints=(DurationConstraint(min_ms=1400, max_ms=1600),),
        ),
        config=harness_config("compact_recovery"),
        run_id="test-regression",
    )
    assert injected.private_trigger().triggered
    result = score_run(case, trajectory, registry.artifact_store, tmp_path / "score")
    assert result["editing_success"]
    assert result["illegal_tool_calls"] == 0
    assert result["tool_call_attempts"] == 3
    wrong = case.model_copy(update={"expected_interval": TimeRange(start_ms=2000, end_ms=3500)})
    assert not score_run(wrong, trajectory, registry.artifact_store, tmp_path / "wrong")[
        "editing_success"
    ]
    impossible = development_cases()[-1]
    refused = score_run(impossible, trajectory, registry.artifact_store, tmp_path / "impossible")
    assert not refused["editing_success"]
    assert not refused["correct_refusal"]


def test_freeze_and_rerun_guards_before_model_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="new directory"):
        freeze(tmp_path, tmp_path / "missing")
    write_json(tmp_path / "freeze.json", {"manifest_sha256": "wrong-hash"})
    write_json(tmp_path / "private_manifest.json", {})
    with pytest.raises(ValueError, match="manifest changed"):
        run(tmp_path, tmp_path / "no-models")
