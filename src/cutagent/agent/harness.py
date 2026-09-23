"""Small assembly of existing runtime/tools; no training or offline labels."""

from pathlib import Path

from cutagent.schemas.event import VerificationResult
from cutagent.schemas.m4b_agent import (
    M4BAgentState,
    M4BProtocolVariant,
    M4BRuntimeConfig,
    StepOutcomeSummary,
)
from cutagent.schemas.tools import ToolExecutionRecord
from cutagent.tools.artifacts import ArtifactStore
from cutagent.tools.cache import ToolCache
from cutagent.tools.editing import TrimVideoTool
from cutagent.tools.executor import FFmpegExecutor
from cutagent.tools.readonly import InspectMediaTool, ValidateMediaTool
from cutagent.tools.registry import ToolRegistry
from cutagent.tools.trace import ToolTraceRecorder
from cutagent.tools.validation import MediaValidator
from cutagent.verification.m4b_online import StepOutcomeBuilder


class HarnessOutcomeBuilder(StepOutcomeBuilder):
    """A rejected finish remains feedback, not a permanent ban on corrected finishes.

    All media/plan prerequisites are recomputed from observable state. The existing
    completion verifier still validates each new FinishDecision, including citations.
    Historical M4B behavior is deliberately unchanged outside this new entry.
    """

    version = "harness-step-outcome-v1"

    def build(
        self,
        state: M4BAgentState,
        *,
        summary_index: int,
        active_node_id: str | None,
        latest_record: ToolExecutionRecord | None,
        latest_verification: VerificationResult | None,
    ) -> StepOutcomeSummary:
        outcome = super().build(
            state,
            summary_index=summary_index,
            active_node_id=active_node_id,
            latest_record=latest_record,
            latest_verification=latest_verification,
        )
        if (
            latest_verification is not None
            and latest_verification.verification_id.startswith("verify-finish-")
            and latest_verification.failure_types == ("completion_evidence_exists",)
        ):
            prerequisites = super().build(
                state,
                summary_index=summary_index,
                active_node_id=active_node_id,
                latest_record=latest_record,
                latest_verification=None,
            )
            if prerequisites.completion_ready:
                # The citation was a property of the previous decision, not of the
                # media. Keep it visible without misclassifying it as a persistent
                # media prerequisite. The next finish is still strictly checked.
                return prerequisites.model_copy(
                    update={
                        "online_verification_status": "failed",
                        "unverified_items": (
                            *prerequisites.unverified_items,
                            "Previous finish rejected: completion evidence IDs were unknown; "
                            "cite an existing artifact or retrieval evidence ID on retry.",
                        ),
                    }
                )
        return outcome


def harness_config(variant: M4BProtocolVariant) -> M4BRuntimeConfig:
    """Equal budgets; the existing recovery protocol is the only variant difference."""
    return M4BRuntimeConfig(
        protocol_variant=variant, seed=20260923, prompt_template_version="harness-policy-v1"
    )


def create_harness_registry(root: Path) -> ToolRegistry:
    """Three audited media tools. Search is explicitly registered when available."""
    store = ArtifactStore(root / "artifacts")
    executor = FFmpegExecutor("ffmpeg")
    validator = MediaValidator(artifact_store=store, executor=executor)
    registry = ToolRegistry(artifact_store=store, trace_recorder=ToolTraceRecorder(root / "traces"))
    registry.register(InspectMediaTool(artifact_store=store, validator=validator))
    registry.register(
        TrimVideoTool(
            artifact_store=store,
            executor=executor,
            validator=validator,
            cache=ToolCache(root / "cache", store),
        )
    )
    registry.register(ValidateMediaTool(artifact_store=store, validator=validator))
    return registry
