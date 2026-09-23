"""Replaceable M4A policy/planner interfaces and inference records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.agent import (
    AgentBaseline,
    PlanGraph,
    PolicyDecision,
    PolicyInferenceStats,
    PolicyOperation,
    PolicyViewMode,
)
from cutagent.schemas.tools import ToolManifest


@dataclass(frozen=True, slots=True)
class PolicyVisualInput:
    artifact_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class PolicyModelRequest:
    operation: PolicyOperation
    baseline: AgentBaseline
    policy_view_mode: PolicyViewMode
    serialized_context: str
    tool_manifest: ToolManifest
    visual_inputs: tuple[PolicyVisualInput, ...]
    output_directory: Path
    maximum_new_tokens: int
    maximum_repairs: int
    prompt_template_version: str
    seed: int


@dataclass(frozen=True, slots=True)
class PolicyModelResult:
    operation: PolicyOperation
    decision: PolicyDecision | None
    proposed_plan: PlanGraph | None
    raw_output_artifact: ArtifactRef
    stats: PolicyInferenceStats


class AgentPolicyModel(Protocol):
    @property
    def backend_version(self) -> str: ...

    def infer(self, request: PolicyModelRequest) -> PolicyModelResult: ...
