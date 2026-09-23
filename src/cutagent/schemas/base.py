"""Shared Pydantic v2 base model and scalar types."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

SchemaVersion = Literal["1.0"]
SCHEMA_VERSION: SchemaVersion = "1.0"
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
    ),
]


class SchemaModel(BaseModel):
    """Immutable, strict base for versioned wire contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion = SCHEMA_VERSION
