"""Deterministic M4A repeated-action and stagnation diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, cast

from cutagent.schemas.agent import RuntimeDiagnostic
from cutagent.schemas.event import PlanPatch, ToolObservation, VerificationResult
from cutagent.schemas.tools import ToolCall


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LoopDetector:
    version = "m4a-loop-detector-v1"

    def __init__(self, *, maximum_identical_actions: int) -> None:
        if maximum_identical_actions <= 0:
            raise ValueError("maximum_identical_actions must be positive")
        self.maximum_identical_actions = maximum_identical_actions
        self._actions: Counter[str] = Counter()
        self._search_results: Counter[str] = Counter()
        self._verification_failures: Counter[str] = Counter()
        self._last_plan: str | None = None
        self._observations: Counter[str] = Counter()

    def observe_tool_call(self, call: ToolCall) -> RuntimeDiagnostic | None:
        payload = call.model_dump(mode="json")
        payload.pop("tool_call_id", None)
        fingerprint = _fingerprint(payload)
        self._actions[fingerprint] += 1
        count = self._actions[fingerprint]
        if count > self.maximum_identical_actions:
            return RuntimeDiagnostic(
                diagnostic_type="identical_tool_call",
                summary="identical normalized ToolCall exceeded the configured repetition limit",
                fingerprint=fingerprint,
                occurrence_count=count,
            )
        return None

    def observe_observation(self, observation: ToolObservation) -> tuple[RuntimeDiagnostic, ...]:
        diagnostics: list[RuntimeDiagnostic] = []
        if observation.tool_name == "search_video" and observation.status == "success":
            response = observation.details.get("response")
            candidates: list[Any] = []
            if isinstance(response, dict) and isinstance(response.get("candidates"), list):
                candidates = cast(list[Any], response["candidates"])
            identities = [
                [item.get("video_id"), item.get("scene_id")]
                for item in candidates
                if isinstance(item, dict)
            ]
            fingerprint = _fingerprint(identities)
            self._search_results[fingerprint] += 1
            count = self._search_results[fingerprint]
            if count > self.maximum_identical_actions:
                diagnostics.append(
                    RuntimeDiagnostic(
                        diagnostic_type="identical_search_result",
                        summary=(
                            "effectively identical search results repeated without new evidence"
                        ),
                        fingerprint=fingerprint,
                        occurrence_count=count,
                    )
                )
        observation_fingerprint = _fingerprint(
            {
                "tool": observation.tool_name,
                "status": observation.status,
                "summary": observation.public_summary,
                "artifacts": [item.artifact_id for item in observation.artifacts],
            }
        )
        self._observations[observation_fingerprint] += 1
        count = self._observations[observation_fingerprint]
        if count > self.maximum_identical_actions:
            diagnostics.append(
                RuntimeDiagnostic(
                    diagnostic_type="no_information_gain",
                    summary="repeated observation added no new public artifact or result",
                    fingerprint=observation_fingerprint,
                    occurrence_count=count,
                )
            )
        return tuple(diagnostics)

    def observe_patch(self, patch: PlanPatch) -> RuntimeDiagnostic | None:
        fingerprint = _fingerprint(
            [
                {
                    **node.model_dump(mode="json"),
                    "status": node.status,
                }
                for node in patch.steps
            ]
        )
        if fingerprint == self._last_plan:
            return RuntimeDiagnostic(
                diagnostic_type="unchanged_replan",
                summary="replan produced no executable graph change",
                fingerprint=fingerprint,
                occurrence_count=2,
            )
        self._last_plan = fingerprint
        return None

    def observe_verification(self, verification: VerificationResult) -> RuntimeDiagnostic | None:
        if verification.status != "failed":
            return None
        fingerprint = _fingerprint(
            {
                "failure_types": verification.failure_types,
                "failed_checks": [
                    item.check_name
                    for item in verification.checks
                    if item.conclusive and not item.passed
                ],
            }
        )
        self._verification_failures[fingerprint] += 1
        count = self._verification_failures[fingerprint]
        if count > self.maximum_identical_actions:
            return RuntimeDiagnostic(
                diagnostic_type="repeated_verification_failure",
                summary="verification failure repeated without new evidence",
                fingerprint=fingerprint,
                occurrence_count=count,
            )
        return None
