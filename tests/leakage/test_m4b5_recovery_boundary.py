"""M4B.5 environment-private injection and runtime-policy isolation."""

from __future__ import annotations

import inspect

from cutagent_evaluation.m4b5_recovery import FailureInjectionConfig

from cutagent.agent.m4b_policy_context import M4BPolicyContextBuilder
from cutagent.agent.m4b_runtime import M4BAgentRuntime


def test_runtime_has_no_private_injection_dependency() -> None:
    source = "\n".join(
        (
            inspect.getsource(M4BAgentRuntime),
            inspect.getsource(M4BPolicyContextBuilder),
        )
    ).casefold()
    assert "failureinjectionconfig" not in source
    assert "expected_recovery_operations" not in source
    assert "environment_private" not in source


def test_injection_contract_is_evaluator_private() -> None:
    assert FailureInjectionConfig.__module__.startswith("cutagent_evaluation.")
