"""Explicitly whitelisted M4B policy view with working-media handoff."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import JsonValue

from cutagent.agent.policy_context import (
    PolicyBudgetView,
    PolicyContextBuilder,
    PolicyObservationView,
    PolicyTaskView,
    PolicyTerminalView,
    PolicyVerificationView,
)
from cutagent.core.errors import PolicyVisibilityError
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.event import PlanNode
from cutagent.schemas.m4b_agent import (
    M4BAgentState,
    ObservableMediaMetadata,
    StepOutcomeSummary,
    WorkingMediaArtifact,
    WorkingValidationStatus,
)
from cutagent.schemas.state import AgentState


class PolicyWorkingMediaView(SchemaModel):
    artifact_id: Identifier
    producing_tool: Identifier | None = None
    media: ObservableMediaMetadata | None = None
    validation_status: WorkingValidationStatus


class PolicyWorkingArtifactView(SchemaModel):
    original_input_artifact_id: Identifier
    current_working_media: PolicyWorkingMediaView
    latest_generated_artifact_id: Identifier | None = None
    final_candidate_artifact_id: Identifier | None = None


class M4BPolicyContext(SchemaModel):
    context_version: NonEmptyStr = "m4b-policy-context-v1"
    task: PolicyTaskView
    state_version: int
    plan_revision: int
    plan_steps: tuple[PlanNode, ...]
    working_artifacts: PolicyWorkingArtifactView
    step_outcome: StepOutcomeSummary
    budget: PolicyBudgetView
    recent_observations: tuple[PolicyObservationView, ...]
    recent_verifications: tuple[PolicyVerificationView, ...]
    terminal: PolicyTerminalView | None


def _working_media(item: WorkingMediaArtifact) -> PolicyWorkingMediaView:
    return PolicyWorkingMediaView(
        artifact_id=item.artifact_id,
        producing_tool=item.producing_tool,
        media=item.metadata,
        validation_status=item.validation_status,
    )


class M4BPolicyContextBuilder:
    """Project M4B state field-by-field; raw state is never serialized."""

    version = "m4b-policy-context-builder-v1"

    def __init__(self, *, history_limit: int = 3) -> None:
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        self._base_builder = PolicyContextBuilder(history_limit=history_limit)

    def build(
        self,
        state: M4BAgentState,
        *,
        step_outcome: StepOutcomeSummary,
    ) -> M4BPolicyContext:
        original = state.working_artifacts.original_input_artifact
        if original.artifact_id != state.task_input.video_ref.artifact_id:
            raise PolicyVisibilityError("M4B original input differs from TaskInput")
        surrogate = AgentState(
            task_input=state.task_input,
            execution_budget=state.execution_budget,
            plan_revision=state.plan_revision,
            plan_steps=state.plan_steps,
            tool_observations=state.tool_observations,
            verification_results=state.verification_results,
            terminal_event=state.terminal_event,
            processed_event_ids=state.processed_event_ids,
            last_sequence_no=state.last_sequence_no,
            state_version=state.state_version,
        )
        base = self._base_builder.build(surrogate)
        working = state.working_artifacts
        return M4BPolicyContext(
            task=base.task,
            state_version=state.state_version,
            plan_revision=state.plan_revision,
            plan_steps=state.plan_steps,
            working_artifacts=PolicyWorkingArtifactView(
                original_input_artifact_id=original.artifact_id,
                current_working_media=_working_media(working.current_working_artifact),
                latest_generated_artifact_id=(
                    working.latest_generated_artifact.artifact_id
                    if working.latest_generated_artifact is not None
                    else None
                ),
                final_candidate_artifact_id=(
                    working.final_candidate_artifact.artifact_id
                    if working.final_candidate_artifact is not None
                    else None
                ),
            ),
            step_outcome=step_outcome,
            budget=base.budget,
            recent_observations=base.recent_observations,
            recent_verifications=base.recent_verifications,
            terminal=base.terminal,
        )


class M4BPolicyViewSerializer:
    """Serialize only the M4B whitelist and reject path/private sentinels."""

    _forbidden_keys = frozenset(
        {
            "benchmarkgold",
            "benchmark_gold",
            "split",
            "source_group_id",
            "path",
            "file_path",
            "filesystem_path",
            "uri",
            "sha256",
            "hash",
            "run_id",
            "run_metadata",
            "evaluator_metadata",
            "ground_truth",
        }
    )
    _absolute_path = re.compile(r"(?:^[A-Za-z]:[\\/]|^/[^/\s])")

    @classmethod
    def to_dict(cls, context: M4BPolicyContext) -> dict[str, Any]:
        payload = context.model_dump(mode="json")
        cls._validate_payload(payload)
        return payload

    @classmethod
    def to_json(cls, context: M4BPolicyContext) -> str:
        return json.dumps(
            cls.to_dict(context),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def _validate_payload(cls, value: JsonValue, *, key_path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.casefold() in cls._forbidden_keys:
                    raise PolicyVisibilityError(
                        f"forbidden M4B policy field: {'.'.join((*key_path, key))}"
                    )
                cls._validate_payload(child, key_path=(*key_path, key))
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                cls._validate_payload(child, key_path=(*key_path, str(index)))
            return
        if isinstance(value, str) and (
            value.startswith("file://") or cls._absolute_path.search(value)
        ):
            raise PolicyVisibilityError("M4B policy value contains a filesystem path")
