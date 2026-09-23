"""Only task information observable by a deployed Agent."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel


class DurationConstraint(SchemaModel):
    kind: Literal["duration"] = "duration"
    min_ms: int = Field(ge=0)
    max_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "DurationConstraint":
        if self.min_ms > self.max_ms:
            raise ValueError("min_ms cannot exceed max_ms")
        return self


class AspectRatioConstraint(SchemaModel):
    kind: Literal["aspect_ratio"] = "aspect_ratio"
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class RequiredContentConstraint(SchemaModel):
    kind: Literal["required_content"] = "required_content"
    description: NonEmptyStr


class ForbiddenContentConstraint(SchemaModel):
    kind: Literal["forbidden_content"] = "forbidden_content"
    description: NonEmptyStr


ObservableConstraint = Annotated[
    DurationConstraint
    | AspectRatioConstraint
    | RequiredContentConstraint
    | ForbiddenContentConstraint,
    Field(discriminator="kind"),
]


class OutputRequest(SchemaModel):
    container: Literal["mp4", "mov", "mkv"] | None = None
    video_codec: NonEmptyStr | None = None


class TaskInput(SchemaModel):
    """Public task contract with no evaluator-only fields."""

    task_id: Identifier
    video_ref: ArtifactRef
    instruction: NonEmptyStr
    user_constraints: tuple[ObservableConstraint, ...] = ()
    requested_output: OutputRequest | None = None
