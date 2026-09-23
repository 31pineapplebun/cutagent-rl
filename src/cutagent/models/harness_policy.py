"""Bound completion citations without changing the frozen parser or model weights."""

from cutagent.agent.m4b_protocols import M4BPolicyModelRequest
from cutagent.models.qwen_policy_m4b import Qwen3VLPolicyBackendM4B


class HarnessPolicyBackend(Qwen3VLPolicyBackendM4B):
    @property
    def backend_version(self) -> str:
        return f"harness-bounded-citations-v1+{super().backend_version}"

    def _prompt(
        self,
        request: M4BPolicyModelRequest,
        *,
        previous: str | None = None,
        error: str | None = None,
    ) -> str:
        prompt = super()._prompt(request, previous=previous, error=error)
        if request.operation == "decide":
            prompt += (
                "\nFinish is allowed ONLY when step_outcome.completion_ready is true. "
                "When false, execute the next required plan node or explicitly cannot_complete. "
                "A trim tool's internal validation does NOT replace the separate validate_media "
                "call on current_working_media. If a validation node is still ready, call "
                "validate_media, not finish.\n"
                "For finish, evidence_ids must contain only 1-3 UNIQUE existing artifact "
                "or retrieval evidence IDs. Citing the validated output artifact ID is valid. "
                "Never cite a step-outcome/summary ID. Never copy an exhaustive evidence list "
                "or repeat IDs. Close the JSON object after these few citations."
            )
        return prompt
