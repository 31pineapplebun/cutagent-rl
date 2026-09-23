"""M4B runtime/private-data and constrained-recovery leakage boundaries."""

from __future__ import annotations

import inspect

from cutagent.agent.m4b_planning import build_local_recovery_patch
from cutagent.agent.m4b_policy_context import M4BPolicyContextBuilder
from cutagent.agent.m4b_runtime import M4BAgentRuntime
from cutagent.verification.m4b_online import StepOutcomeBuilder


def test_m4b_runtime_interfaces_cannot_accept_private_gold() -> None:
    signatures = "\n".join(
        str(inspect.signature(item))
        for item in (
            M4BAgentRuntime.run,
            M4BPolicyContextBuilder.build,
            StepOutcomeBuilder.build,
            build_local_recovery_patch,
        )
    ).casefold()
    assert "benchmarkgold" not in signatures
    assert "taskannotation" not in signatures
    assert "source_group_id" not in signatures
    assert "split" not in signatures


def test_m4b_runtime_has_only_tool_registry_execution_dependency() -> None:
    source = inspect.getsource(M4BAgentRuntime)
    assert "registry.execute" in source
    assert "subprocess" not in source
    assert "shell=true" not in source.casefold()
    assert "ffmpeg" not in source.casefold()
