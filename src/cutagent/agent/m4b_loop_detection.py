"""M4B loop guards with working-artifact-aware repeated-editor detection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, cast

from cutagent.schemas.event import PlanPatch, ToolObservation, VerificationResult
from cutagent.schemas.m4b_agent import M4BRuntimeDiagnostic
from cutagent.schemas.tools import ToolCall

_EDIT_TOOLS = frozenset(
    {
        "trim_video",
        "concat_videos",
        "change_speed",
        "add_subtitles",
        "reframe_video",
        "normalize_audio",
    }
)


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class M4BLoopDetector:
    version = "m4b-loop-detector-v1"

    def __init__(self, *, maximum_identical_actions: int) -> None:
        if maximum_identical_actions <= 0:
            raise ValueError("maximum_identical_actions must be positive")
        self.maximum_identical_actions = maximum_identical_actions
        self._actions: Counter[str] = Counter()
        self._editor_calls: Counter[str] = Counter()
        self._editor_last_information: dict[str, int] = {}
        self._editor_no_gain: Counter[str] = Counter()
        self._search_results: Counter[str] = Counter()
        self._verification_failures: Counter[str] = Counter()
        self._observations: Counter[str] = Counter()
        self._last_plan: str | None = None
        self.repeated_editor_occurrences = 0

    @staticmethod
    def _call_payload(call: ToolCall) -> dict[str, Any]:
        payload = call.model_dump(mode="json")
        payload.pop("tool_call_id", None)
        return payload

    def observe_tool_call(self, call: ToolCall) -> M4BRuntimeDiagnostic | None:
        fingerprint = _fingerprint(self._call_payload(call))
        self._actions[fingerprint] += 1
        count = self._actions[fingerprint]
        if count > self.maximum_identical_actions:
            return M4BRuntimeDiagnostic(
                diagnostic_type="identical_tool_call",
                summary="identical normalized ToolCall exceeded the configured repetition limit",
                fingerprint=fingerprint,
                occurrence_count=count,
            )
        return None

    def observe_editor_call(
        self,
        call: ToolCall,
        *,
        current_working_artifact_id: str,
        information_version: int,
    ) -> M4BRuntimeDiagnostic | None:
        if call.tool_name not in _EDIT_TOOLS:
            return None
        payload = self._call_payload(call)
        arguments = payload.get("arguments")
        effective_sources: object = current_working_artifact_id
        if isinstance(arguments, dict):
            effective_sources = arguments.get(
                "input_artifact_ids",
                arguments.get("input_artifact_id", current_working_artifact_id),
            )
        fingerprint = _fingerprint(
            {
                "tool_family": call.tool_name,
                "effective_sources": effective_sources,
                "normalized_arguments": arguments,
            }
        )
        self._editor_calls[fingerprint] += 1
        call_count = self._editor_calls[fingerprint]
        if call_count > 1:
            self.repeated_editor_occurrences += 1
        previous_information = self._editor_last_information.get(fingerprint)
        if previous_information is not None and information_version <= previous_information:
            self._editor_no_gain[fingerprint] += 1
        else:
            self._editor_no_gain[fingerprint] = 0
        self._editor_last_information[fingerprint] = information_version
        no_gain_count = self._editor_no_gain[fingerprint]
        if no_gain_count >= self.maximum_identical_actions:
            return M4BRuntimeDiagnostic(
                diagnostic_type="repeated_editor",
                summary=(
                    "equivalent editor repeated on the same effective source without "
                    "new public information or verification gain"
                ),
                fingerprint=fingerprint,
                occurrence_count=call_count,
            )
        return None

    def observe_observation(self, observation: ToolObservation) -> tuple[M4BRuntimeDiagnostic, ...]:
        diagnostics: list[M4BRuntimeDiagnostic] = []
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
                    M4BRuntimeDiagnostic(
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
                M4BRuntimeDiagnostic(
                    diagnostic_type="no_information_gain",
                    summary="repeated observation added no new public artifact or result",
                    fingerprint=observation_fingerprint,
                    occurrence_count=count,
                )
            )
        return tuple(diagnostics)

    def observe_patch(self, patch: PlanPatch) -> M4BRuntimeDiagnostic | None:
        fingerprint = _fingerprint([node.model_dump(mode="json") for node in patch.steps])
        if fingerprint == self._last_plan:
            return M4BRuntimeDiagnostic(
                diagnostic_type="unchanged_replan",
                summary="replan produced no executable graph change",
                fingerprint=fingerprint,
                occurrence_count=2,
            )
        self._last_plan = fingerprint
        return None

    def observe_verification(self, verification: VerificationResult) -> M4BRuntimeDiagnostic | None:
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
            return M4BRuntimeDiagnostic(
                diagnostic_type="repeated_verification_failure",
                summary="verification failure repeated without new evidence",
                fingerprint=fingerprint,
                occurrence_count=count,
            )
        return None
