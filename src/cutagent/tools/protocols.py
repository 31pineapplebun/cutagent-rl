"""Narrow runtime protocol for statically registered M3A tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import JsonValue

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.base import SchemaModel
from cutagent.schemas.tools import ToolExecutionContext, ToolSpec, ToolValidationCheck


@dataclass(frozen=True, slots=True)
class ToolResult:
    public_summary: str
    details: dict[str, JsonValue]
    parent_artifacts: tuple[ArtifactRef, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    output_artifact: ArtifactRef | None = None
    validation_results: tuple[ToolValidationCheck, ...] = ()
    ffmpeg_return_code: int | None = None
    cache_hit: bool = False
    cache_key: str | None = None


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    @property
    def arguments_type(self) -> type[SchemaModel]: ...

    def execute(
        self,
        arguments: SchemaModel,
        context: ToolExecutionContext,
        *,
        tool_call_id: str,
    ) -> ToolResult: ...
