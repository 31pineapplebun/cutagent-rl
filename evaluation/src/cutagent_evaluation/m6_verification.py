"""M6-only observable metadata compatibility fix for executed oracle traces."""

from __future__ import annotations

from typing import Literal

from cutagent.schemas.m4b_agent import (
    M4BAgentState,
    ObservableMediaMetadata,
    WorkingArtifactValidationEvent,
)
from cutagent.schemas.tools import ToolExecutionRecord
from cutagent.verification.m4b_online import M4BOnlineVerifier


class M6OracleOnlineVerifier(M4BOnlineVerifier):
    """Read nested validate_media metadata without changing frozen M4B behavior."""

    version = "m6-oracle-online-verifier-v1"

    @staticmethod
    def working_validation_event(
        state: M4BAgentState,
        record: ToolExecutionRecord,
    ) -> WorkingArtifactValidationEvent | None:
        if record.trace.tool_name != "validate_media" or len(record.trace.parent_artifacts) != 1:
            return None
        artifact_id = record.trace.parent_artifacts[0].artifact_id
        if artifact_id != state.working_artifacts.current_working_artifact.artifact_id:
            return None
        status: Literal["passed", "failed", "inconclusive"] = (
            "passed"
            if record.observation.status == "success"
            else "inconclusive"
            if record.observation.status == "timeout"
            else "failed"
        )
        details = record.observation.details
        video = details.get("video")
        video = video if isinstance(video, dict) else {}
        audio = details.get("audio")
        duration = details.get("duration_ms")
        width = video.get("width", details.get("width"))
        height = video.get("height", details.get("height"))
        decoded = details.get("fully_decoded")
        metadata = None
        if record.observation.status == "success":
            raw_has_audio = details.get("has_audio")
            has_audio: bool | None = (
                bool(audio)
                if isinstance(audio, list)
                else raw_has_audio
                if isinstance(raw_has_audio, bool)
                else None
            )
            metadata = ObservableMediaMetadata(
                duration_ms=(
                    duration
                    if isinstance(duration, int) and not isinstance(duration, bool)
                    else None
                ),
                width=width if isinstance(width, int) and not isinstance(width, bool) else None,
                height=(
                    height if isinstance(height, int) and not isinstance(height, bool) else None
                ),
                has_audio=has_audio,
                fully_decoded=decoded if isinstance(decoded, bool) else None,
            )
        return WorkingArtifactValidationEvent(
            artifact_id=artifact_id,
            source_tool_call_id=record.observation.call_id,
            status=status,
            observable_metadata=metadata,
        )
