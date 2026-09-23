"""Versioned policy interface for M4B handoff and compact recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.agent import PlanGraph, PolicyDecision
from cutagent.schemas.m4b_agent import (
    M4BPolicyInferenceStats,
    M4BPolicyOperation,
    M4BProtocolVariant,
    RecoveryDecision,
)
from cutagent.schemas.tools import ToolManifest


@dataclass(frozen=True, slots=True)
class M4BPolicyModelRequest:
    operation: M4BPolicyOperation
    protocol_variant: M4BProtocolVariant
    serialized_context: str
    tool_manifest: ToolManifest
    output_directory: Path
    maximum_new_tokens: int
    maximum_repairs: int
    prompt_template_version: str
    seed: int


@dataclass(frozen=True, slots=True)
class M4BPolicyModelResult:
    operation: M4BPolicyOperation
    decision: PolicyDecision | RecoveryDecision | None
    proposed_plan: PlanGraph | None
    raw_output_artifact: ArtifactRef
    stats: M4BPolicyInferenceStats


class M4BPolicyModel(Protocol):
    @property
    def backend_version(self) -> str: ...

    def infer(self, request: M4BPolicyModelRequest) -> M4BPolicyModelResult: ...
