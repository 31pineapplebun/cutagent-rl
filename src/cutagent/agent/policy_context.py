"""Explicitly whitelisted, model-visible views of public Agent state."""

import json
import re
from typing import Any, Literal

from pydantic import JsonValue

from cutagent.core.errors import PolicyVisibilityError
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.event import (
    PlanNode,
    TerminalStatus,
    ToolStatus,
    VerificationStatus,
)
from cutagent.schemas.perception import (
    EvidenceRef,
    OCRSpan,
    TemporalEvent,
    TranscriptSpan,
    VideoWorldState,
    VisualAction,
    VisualEntity,
)
from cutagent.schemas.retrieval import RetrievalEvidenceType, RetrievalResponse
from cutagent.schemas.state import AgentState
from cutagent.schemas.task_input import ObservableConstraint, OutputRequest


class PolicyArtifactView(SchemaModel):
    artifact_id: Identifier
    media_type: NonEmptyStr


class PolicyTaskView(SchemaModel):
    task_id: Identifier
    video: PolicyArtifactView
    instruction: NonEmptyStr
    user_constraints: tuple[ObservableConstraint, ...]
    requested_output: OutputRequest | None


class PolicyBudgetView(SchemaModel):
    remaining_steps: int
    remaining_tool_calls: int
    remaining_search_calls: int
    remaining_edit_calls: int
    remaining_structured_output_repairs: int
    remaining_model_tokens: int | None
    remaining_wall_time_ms: int


class PolicyRetrievedEvidenceView(SchemaModel):
    evidence_type: RetrievalEvidenceType
    artifact_id: Identifier
    segment_id: Identifier | None
    observed_ms: int | None
    matched_text: NonEmptyStr | None


class PolicyRetrievedSceneView(SchemaModel):
    rank: int
    video_id: Identifier
    scene_id: Identifier
    start_ms: int
    end_ms: int
    evidence_types: tuple[RetrievalEvidenceType, ...]
    evidence: tuple[PolicyRetrievedEvidenceView, ...]


class PolicyMediaObservationView(SchemaModel):
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool | None = None
    fully_decoded: bool | None = None


class PolicyObservationView(SchemaModel):
    tool_name: Identifier
    status: ToolStatus
    public_summary: NonEmptyStr
    artifact_ids: tuple[Identifier, ...]
    retrieved_scenes: tuple[PolicyRetrievedSceneView, ...] = ()
    media: PolicyMediaObservationView | None = None


class PolicyVerificationView(SchemaModel):
    status: VerificationStatus
    check_summaries: tuple[NonEmptyStr, ...]
    failure_types: tuple[Identifier, ...]


class PolicyTerminalView(SchemaModel):
    status: TerminalStatus
    reason: NonEmptyStr


class PolicySceneWorldView(SchemaModel):
    segment_id: Identifier
    start_ms: int
    end_ms: int
    keyframe_evidence: tuple[EvidenceRef, ...]
    transcript_spans: tuple[TranscriptSpan, ...]
    ocr_spans: tuple[OCRSpan, ...]
    entities: tuple[VisualEntity, ...]
    actions: tuple[VisualAction, ...]
    temporal_events: tuple[TemporalEvent, ...]
    scene_summary: NonEmptyStr | None


class PolicyVideoWorldView(SchemaModel):
    video_id: Identifier
    duration_ms: int
    scenes: tuple[PolicySceneWorldView, ...]
    global_summary: NonEmptyStr | None


class PolicyVisualEvidenceView(SchemaModel):
    artifact_id: Identifier
    segment_id: Identifier | None
    observed_ms: int | None
    evidence_kind: NonEmptyStr


class PolicyContext(SchemaModel):
    context_version: Literal["m4a-policy-context-v1"] = "m4a-policy-context-v1"
    task: PolicyTaskView
    state_version: int
    plan_revision: int
    plan_steps: tuple[PlanNode, ...]
    budget: PolicyBudgetView
    recent_observations: tuple[PolicyObservationView, ...]
    recent_verifications: tuple[PolicyVerificationView, ...]
    terminal: PolicyTerminalView | None
    video_world: PolicyVideoWorldView | None = None
    visual_evidence: tuple[PolicyVisualEvidenceView, ...] = ()


class PolicyContextBuilder:
    """Build a policy context from public state using field-by-field copying."""

    def __init__(self, *, history_limit: int = 8, maximum_retrieved_scenes: int = 5) -> None:
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        if maximum_retrieved_scenes <= 0:
            raise ValueError("maximum_retrieved_scenes must be positive")
        self._history_limit = history_limit
        self._maximum_retrieved_scenes = maximum_retrieved_scenes

    def _project_retrieval(
        self, tool_name: str, status: ToolStatus, details: dict[str, JsonValue]
    ) -> tuple[PolicyRetrievedSceneView, ...]:
        if tool_name != "search_video" or status != "success":
            return ()
        raw = details.get("response")
        if not isinstance(raw, dict):
            raise PolicyVisibilityError("successful search observation omitted its response")
        try:
            response = RetrievalResponse.model_validate(raw)
        except ValueError as error:
            raise PolicyVisibilityError(
                "search observation has an invalid public response"
            ) from error
        return tuple(
            PolicyRetrievedSceneView(
                rank=candidate.rank,
                video_id=candidate.video_id,
                scene_id=candidate.scene_id,
                start_ms=candidate.time_range.start_ms,
                end_ms=candidate.time_range.end_ms,
                evidence_types=candidate.evidence_types,
                evidence=tuple(
                    PolicyRetrievedEvidenceView(
                        evidence_type=item.evidence_type,
                        artifact_id=item.evidence_ref.artifact_id,
                        segment_id=item.evidence_ref.segment_id,
                        observed_ms=item.evidence_ref.observed_ms,
                        matched_text=item.matched_text,
                    )
                    for item in candidate.evidence
                ),
            )
            for candidate in response.candidates[: self._maximum_retrieved_scenes]
        )

    @staticmethod
    def _project_media(
        tool_name: str, status: ToolStatus, details: dict[str, JsonValue]
    ) -> PolicyMediaObservationView | None:
        if status != "success" or tool_name == "search_video":
            return None
        duration = details.get("duration_ms")
        width = details.get("width")
        height = details.get("height")
        has_audio = details.get("has_audio")
        video = details.get("video")
        if isinstance(video, dict):
            width = video.get("width", width)
            height = video.get("height", height)
        audio = details.get("audio")
        if isinstance(audio, list):
            has_audio = bool(audio)
        decoded = details.get("fully_decoded")
        if not any(
            (
                isinstance(duration, int),
                isinstance(width, int),
                isinstance(height, int),
                isinstance(has_audio, bool),
                isinstance(decoded, bool),
            )
        ):
            return None
        return PolicyMediaObservationView(
            duration_ms=duration if isinstance(duration, int) else None,
            width=width if isinstance(width, int) else None,
            height=height if isinstance(height, int) else None,
            has_audio=has_audio if isinstance(has_audio, bool) else None,
            fully_decoded=decoded if isinstance(decoded, bool) else None,
        )

    def build(
        self,
        state: AgentState,
        *,
        world_state: VideoWorldState | None = None,
        include_visual_evidence: bool = False,
        maximum_visual_evidence: int = 4,
    ) -> PolicyContext:
        if maximum_visual_evidence < 0:
            raise ValueError("maximum_visual_evidence cannot be negative")
        task = state.task_input
        observations = tuple(
            PolicyObservationView(
                tool_name=observation.tool_name,
                status=observation.status,
                public_summary=observation.public_summary,
                artifact_ids=tuple(artifact.artifact_id for artifact in observation.artifacts),
                retrieved_scenes=self._project_retrieval(
                    observation.tool_name, observation.status, observation.details
                ),
                media=self._project_media(
                    observation.tool_name, observation.status, observation.details
                ),
            )
            for observation in state.tool_observations[-self._history_limit :]
        )
        verifications = tuple(
            PolicyVerificationView(
                status=result.status,
                check_summaries=tuple(check.summary for check in result.checks),
                failure_types=result.failure_types,
            )
            for result in state.verification_results[-self._history_limit :]
        )
        terminal = None
        if state.terminal_event is not None:
            terminal = PolicyTerminalView(
                status=state.terminal_event.status,
                reason=state.terminal_event.reason,
            )
        if world_state is not None:
            source_ids = {
                evidence.artifact_id
                for evidence in world_state.evidence_catalog
                if evidence.evidence_kind == "source_media"
            }
            if task.video_ref.artifact_id not in source_ids:
                raise PolicyVisibilityError("world state does not belong to the task video")
        world_view = None
        if world_state is not None:
            world_view = PolicyVideoWorldView(
                video_id=world_state.video_id,
                duration_ms=world_state.duration_ms,
                scenes=tuple(
                    PolicySceneWorldView(
                        segment_id=scene.segment_id,
                        start_ms=scene.time_range.start_ms,
                        end_ms=scene.time_range.end_ms,
                        keyframe_evidence=scene.keyframe_evidence,
                        transcript_spans=scene.transcript_spans,
                        ocr_spans=scene.ocr_spans,
                        entities=scene.entities,
                        actions=scene.actions,
                        temporal_events=scene.temporal_events,
                        scene_summary=scene.scene_summary,
                    )
                    for scene in world_state.scenes
                ),
                global_summary=world_state.global_summary,
            )
        visual_evidence: list[PolicyVisualEvidenceView] = []
        if include_visual_evidence:
            seen: set[str] = set()
            for observation in reversed(observations):
                for scene in observation.retrieved_scenes:
                    for evidence in scene.evidence:
                        if evidence.evidence_type != "keyframe":
                            continue
                        if evidence.artifact_id in seen:
                            continue
                        seen.add(evidence.artifact_id)
                        visual_evidence.append(
                            PolicyVisualEvidenceView(
                                artifact_id=evidence.artifact_id,
                                segment_id=evidence.segment_id,
                                observed_ms=evidence.observed_ms,
                                evidence_kind="retrieved_keyframe",
                            )
                        )
                        if len(visual_evidence) >= maximum_visual_evidence:
                            break
                    if len(visual_evidence) >= maximum_visual_evidence:
                        break
                if len(visual_evidence) >= maximum_visual_evidence:
                    break
        return PolicyContext(
            task=PolicyTaskView(
                task_id=task.task_id,
                video=PolicyArtifactView(
                    artifact_id=task.video_ref.artifact_id,
                    media_type=task.video_ref.media_type,
                ),
                instruction=task.instruction,
                user_constraints=task.user_constraints,
                requested_output=task.requested_output,
            ),
            state_version=state.state_version,
            plan_revision=state.plan_revision,
            plan_steps=state.plan_steps,
            budget=PolicyBudgetView(
                remaining_steps=state.execution_budget.remaining_steps,
                remaining_tool_calls=state.execution_budget.remaining_tool_calls,
                remaining_search_calls=state.execution_budget.remaining_search_calls,
                remaining_edit_calls=state.execution_budget.remaining_edit_calls,
                remaining_structured_output_repairs=(
                    state.execution_budget.remaining_structured_output_repairs
                ),
                remaining_model_tokens=state.execution_budget.remaining_model_tokens,
                remaining_wall_time_ms=state.execution_budget.remaining_wall_time_ms,
            ),
            recent_observations=observations,
            recent_verifications=verifications,
            terminal=terminal,
            video_world=world_view,
            visual_evidence=tuple(visual_evidence),
        )


class PolicyViewSerializer:
    """Serialize only the PolicyContext schema and enforce leakage sentinels."""

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
    def to_dict(cls, context: PolicyContext) -> dict[str, Any]:
        payload = context.model_dump(mode="json")
        cls._validate_payload(payload)
        return payload

    @classmethod
    def to_json(cls, context: PolicyContext) -> str:
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
                normalized = key.casefold()
                if normalized in cls._forbidden_keys:
                    dotted = ".".join((*key_path, key))
                    raise PolicyVisibilityError(f"forbidden policy field: {dotted}")
                cls._validate_payload(child, key_path=(*key_path, key))
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                cls._validate_payload(child, key_path=(*key_path, str(index)))
            return
        if isinstance(value, str) and (
            value.startswith("file://") or cls._absolute_path.search(value)
        ):
            raise PolicyVisibilityError("policy value contains a filesystem path")
