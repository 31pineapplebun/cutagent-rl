"""Static M3A tool registry, policy enforcement, execution, and tracing."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from pydantic import JsonValue, ValidationError

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.base import SchemaModel
from cutagent.schemas.event import ToolObservation, ToolStatus
from cutagent.schemas.tools import (
    TOOL_CALL_ADAPTER,
    ToolCall,
    ToolCapability,
    ToolExecutionContext,
    ToolExecutionRecord,
    ToolManifest,
    ToolTrace,
)
from cutagent.tools.artifacts import ArtifactStore, trace_artifact
from cutagent.tools.errors import ToolFailure
from cutagent.tools.protocols import Tool
from cutagent.tools.trace import ToolTraceRecorder

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_FORBIDDEN_DETAIL_KEYS = {
    "path",
    "paths",
    "uri",
    "sha256",
    "source_group_id",
    "split",
    "benchmark_gold",
    "evaluator_metadata",
}

CAPABILITIES: tuple[ToolCapability, ...] = (
    ToolCapability(
        capability="retrieval.read",
        description="Read a frozen public retrieval index.",
        read_only=True,
    ),
    ToolCapability(
        capability="media.inspect",
        description="Read approved media metadata and integrity.",
        read_only=True,
    ),
    ToolCapability(
        capability="media.decode",
        description="Decode approved media for post-execution validation.",
        read_only=True,
    ),
    ToolCapability(
        capability="media.write",
        description="Create a new content-addressed video artifact.",
        read_only=False,
    ),
    ToolCapability(
        capability="audio.write",
        description="Transform audio in a new content-addressed media artifact.",
        read_only=False,
    ),
)


def _safe_identifier(value: object, fallback: str) -> str:
    if isinstance(value, str) and _IDENTIFIER.fullmatch(value):
        return value
    return fallback


def _stable_invalid_call_id(raw: object) -> str:
    try:
        serialized = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = type(raw).__name__
    return f"invalid-{hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]}"


def _public_details_are_safe(value: JsonValue, *, key: str | None = None) -> bool:
    if key is not None and key.casefold() in _FORBIDDEN_DETAIL_KEYS:
        return False
    if isinstance(value, str):
        lowered = value.casefold()
        if lowered.startswith("file://") or re.match(r"^[a-z]:[/\\]", lowered):
            return False
        return not value.startswith("/")
    if isinstance(value, list):
        return all(_public_details_are_safe(item) for item in value)
    if isinstance(value, dict):
        return all(
            _public_details_are_safe(item, key=str(item_key)) for item_key, item in value.items()
        )
    return True


def _artifact_ids(arguments: SchemaModel) -> tuple[str, ...]:
    payload = arguments.model_dump(mode="python")
    single = payload.get("input_artifact_id")
    multiple = payload.get("input_artifact_ids")
    if isinstance(single, str):
        return (single,)
    if isinstance(multiple, tuple):
        return cast(tuple[str, ...], multiple)
    return ()


class ToolRegistry:
    version = "m3a-tool-registry-v1"

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        trace_recorder: ToolTraceRecorder,
    ) -> None:
        self.artifact_store = artifact_store
        self.trace_recorder = trace_recorder
        self._tools: dict[str, Tool] = {}
        self._execution_grants: dict[str, set[str]] = {}

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"tool {name!r} is already registered")
        self._tools[name] = tool

    def manifest(self) -> ToolManifest:
        specs = tuple(self._tools[name].spec for name in sorted(self._tools))
        digest = hashlib.sha256(
            json.dumps(
                [item.model_dump(mode="json") for item in specs],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return ToolManifest(
            manifest_id=f"tool-manifest-{digest[:20]}",
            registry_version=self.version,
            tools=specs,
            capabilities=CAPABILITIES,
        )

    def _failure_record(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        tool_version: str,
        context: ToolExecutionContext,
        normalized_arguments: dict[str, JsonValue],
        parents: tuple[ArtifactRef, ...],
        started_at: datetime,
        started_ns: int,
        failure: ToolFailure,
    ) -> ToolExecutionRecord:
        status: ToolStatus = (
            "timeout"
            if failure.category == "timeout"
            else "invalid"
            if failure.category
            in {
                "invalid_call",
                "unknown_tool",
                "capability_denied",
                "artifact_not_allowed",
                "artifact_not_found",
                "filesystem_violation",
                "invalid_interval",
                "incompatible_media",
                "invalid_subtitle",
            }
            else "error"
        )
        trace_id = f"trace-{uuid4().hex}"
        ended_at = datetime.now(UTC)
        trace = ToolTrace(
            trace_id=trace_id,
            execution_id=context.execution_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_version=tool_version,
            normalized_arguments=normalized_arguments,
            parent_artifacts=tuple(trace_artifact(item) for item in parents),
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=(time.perf_counter_ns() - started_ns) // 1_000_000,
            ffmpeg_return_code=failure.ffmpeg_return_code,
            validation_results=failure.validation_results,
            status=status,
            error_category=failure.category,
            cache_hit=False,
        )
        self.trace_recorder.write(trace)
        observation = ToolObservation(
            call_id=tool_call_id,
            tool_name=tool_name,
            status=status,
            public_summary=failure.public_summary,
            details={"trace_id": trace_id, "error_category": failure.category},
            error_code=failure.category,
        )
        return ToolExecutionRecord(observation=observation, trace=trace)

    def execute(
        self,
        call: ToolCall | Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionRecord:
        raw: object = call
        if isinstance(call, SchemaModel):
            raw = call.model_dump(mode="python")
        raw_mapping = raw if isinstance(raw, Mapping) else {}
        fallback_id = _stable_invalid_call_id(raw)
        tool_call_id = _safe_identifier(raw_mapping.get("tool_call_id"), fallback_id)
        tool_name = _safe_identifier(raw_mapping.get("tool_name"), "invalid_tool")
        started_at = datetime.now(UTC)
        started_ns = time.perf_counter_ns()
        grants = self._execution_grants.setdefault(context.execution_id, set())
        grants.update(context.allowed_artifact_ids)
        effective_context = context.model_copy(
            update={"allowed_artifact_ids": tuple(sorted(grants))}
        )

        tool = self._tools.get(tool_name)
        if tool is None:
            return self._failure_record(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_version="unregistered",
                context=effective_context,
                normalized_arguments={},
                parents=(),
                started_at=started_at,
                started_ns=started_ns,
                failure=ToolFailure("unknown_tool", "requested tool is not registered"),
            )
        try:
            parsed = TOOL_CALL_ADAPTER.validate_python(raw)
        except ValidationError:
            return self._failure_record(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_version=tool.spec.version,
                context=effective_context,
                normalized_arguments={},
                parents=(),
                started_at=started_at,
                started_ns=started_ns,
                failure=ToolFailure("invalid_call", "tool call failed schema validation"),
            )
        if not isinstance(parsed.arguments, tool.arguments_type):
            return self._failure_record(
                tool_call_id=parsed.tool_call_id,
                tool_name=parsed.tool_name,
                tool_version=tool.spec.version,
                context=effective_context,
                normalized_arguments={},
                parents=(),
                started_at=started_at,
                started_ns=started_ns,
                failure=ToolFailure("invalid_call", "tool arguments do not match tool spec"),
            )
        normalized = cast(dict[str, JsonValue], parsed.arguments.model_dump(mode="json"))
        parents: tuple[ArtifactRef, ...] = ()
        try:
            missing_capabilities = set(tool.spec.capabilities) - set(
                effective_context.allowed_capabilities
            )
            if missing_capabilities:
                raise ToolFailure(
                    "capability_denied",
                    "execution policy does not grant required tool capabilities",
                )
            parents = tuple(
                self.artifact_store.resolve_allowed(item, effective_context)[0]
                for item in _artifact_ids(parsed.arguments)
            )
            result = tool.execute(
                parsed.arguments,
                effective_context,
                tool_call_id=parsed.tool_call_id,
            )
            if (
                result.output_artifact is not None
                and result.output_artifact.size_bytes > effective_context.maximum_output_bytes
            ):
                raise ToolFailure(
                    "output_too_large",
                    "tool output exceeded the configured size limit",
                )
            if not _public_details_are_safe(cast(JsonValue, result.details)):
                raise ToolFailure(
                    "internal_error",
                    "tool attempted to expose non-public execution metadata",
                )
        except ToolFailure as failure:
            return self._failure_record(
                tool_call_id=parsed.tool_call_id,
                tool_name=parsed.tool_name,
                tool_version=tool.spec.version,
                context=effective_context,
                normalized_arguments=normalized,
                parents=parents,
                started_at=started_at,
                started_ns=started_ns,
                failure=failure,
            )
        except Exception:  # defensive runtime boundary
            return self._failure_record(
                tool_call_id=parsed.tool_call_id,
                tool_name=parsed.tool_name,
                tool_version=tool.spec.version,
                context=effective_context,
                normalized_arguments=normalized,
                parents=parents,
                started_at=started_at,
                started_ns=started_ns,
                failure=ToolFailure("internal_error", "tool execution failed internally"),
            )

        trace_id = f"trace-{uuid4().hex}"
        ended_at = datetime.now(UTC)
        trace = ToolTrace(
            trace_id=trace_id,
            execution_id=effective_context.execution_id,
            tool_call_id=parsed.tool_call_id,
            tool_name=parsed.tool_name,
            tool_version=tool.spec.version,
            normalized_arguments=normalized,
            parent_artifacts=tuple(trace_artifact(item) for item in result.parent_artifacts),
            output_artifact=(
                trace_artifact(result.output_artifact)
                if result.output_artifact is not None
                else None
            ),
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=(time.perf_counter_ns() - started_ns) // 1_000_000,
            ffmpeg_return_code=result.ffmpeg_return_code,
            validation_results=result.validation_results,
            status="success",
            cache_hit=result.cache_hit,
            cache_key=result.cache_key,
        )
        self.trace_recorder.write(trace)
        details: dict[str, JsonValue] = {
            **result.details,
            "trace_id": trace_id,
            "cache_hit": result.cache_hit,
        }
        observation = ToolObservation(
            call_id=parsed.tool_call_id,
            tool_name=parsed.tool_name,
            status="success",
            public_summary=result.public_summary,
            details=details,
            artifacts=result.artifacts,
        )
        if result.output_artifact is not None:
            grants.add(result.output_artifact.artifact_id)
        return ToolExecutionRecord(observation=observation, trace=trace)
