"""Public, answer-free schemas for independent human trajectory review."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel

ReviewMediaRole = Literal["source", "candidate_output"]
ObservableToolStatus = Literal["success", "error", "invalid", "timeout"]


class HumanReviewMediaRef(SchemaModel):
    """Portable media link relative to one static case directory."""

    role: ReviewMediaRole
    relative_path: NonEmptyStr
    media_type: NonEmptyStr

    @field_validator("relative_path")
    @classmethod
    def relative_bundle_path_only(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        parts = normalized.split("/")
        if normalized.startswith("/") or ":" in parts[0]:
            raise ValueError("review media path must be relative")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("review media path contains an unsafe component")
        return normalized


class HumanReviewMediaMetadata(SchemaModel):
    """Media properties observable through ordinary playback or probing."""

    duration_ms: int = Field(ge=0)
    has_video: bool
    has_audio: bool
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def dimensions_require_video(self) -> HumanReviewMediaMetadata:
        if not self.has_video and (self.width is not None or self.height is not None):
            raise ValueError("dimensions require a video stream")
        if self.has_video and (self.width is None or self.height is None):
            raise ValueError("video metadata requires width and height")
        return self


class HumanReviewToolObservation(SchemaModel):
    """Compact public outcome of one tool execution."""

    sequence: int = Field(ge=1)
    tool_name: Identifier
    status: ObservableToolStatus
    public_summary: NonEmptyStr
    error_code: Identifier | None = None


class HumanReviewTrajectorySummary(SchemaModel):
    """Minimal public trace needed to assign the frozen failure taxonomy."""

    tool_observations: tuple[HumanReviewToolObservation, ...] = ()
    verification_statuses: tuple[NonEmptyStr, ...] = ()
    structured_decision_failure_count: int = Field(ge=0)
    final_output_present: bool

    @field_validator("verification_statuses")
    @classmethod
    def bounded_verification_statuses(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 64:
            raise ValueError("too many verification statuses for a concise review summary")
        return value


class HumanReviewCase(SchemaModel):
    """Exactly the observable presentation shown to an independent human rater.

    ``extra='forbid'`` is inherited from :class:`SchemaModel`, so evaluator-only
    fields cannot be smuggled into this contract.
    """

    case_id: Identifier
    display_index: int = Field(ge=1, le=50)
    instruction: NonEmptyStr
    source_media_ref: HumanReviewMediaRef
    candidate_output_ref: HumanReviewMediaRef | None = None
    source_media_metadata: HumanReviewMediaMetadata
    candidate_output_metadata: HumanReviewMediaMetadata | None = None
    public_trajectory_summary: HumanReviewTrajectorySummary
    terminal_behavior: NonEmptyStr
    review_instructions: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_media_roles_and_output_pair(self) -> HumanReviewCase:
        if self.source_media_ref.role != "source":
            raise ValueError("source_media_ref must use the source role")
        if self.candidate_output_ref is not None:
            if self.candidate_output_ref.role != "candidate_output":
                raise ValueError("candidate_output_ref must use the candidate_output role")
            if self.candidate_output_metadata is None:
                raise ValueError("candidate output metadata is required with output media")
        elif self.candidate_output_metadata is not None:
            raise ValueError("candidate output metadata requires output media")
        if self.public_trajectory_summary.final_output_present != (
            self.candidate_output_ref is not None
        ):
            raise ValueError("trajectory output flag must match candidate output presentation")
        return self
