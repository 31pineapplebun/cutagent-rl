"""Reproducible run metadata and deterministic configuration hashing."""

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator

from cutagent.core.artifacts import ArtifactRef
from cutagent.core.versions import CodeVersion, detect_code_version
from cutagent.schemas.base import Identifier, SchemaModel


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    raise TypeError(f"configuration value {type(value).__name__} is not JSON serializable")


def canonical_config_json(config: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


class RunManifest(SchemaModel):
    run_id: Identifier
    created_at: datetime
    code_version: CodeVersion
    seed: int = Field(ge=0, le=2**63 - 1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_manifest_ref: ArtifactRef
    model_versions: dict[str, str] = Field(default_factory=dict)
    dataset_versions: dict[str, str] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @field_validator("model_versions", "dataset_versions")
    @classmethod
    def validate_versions(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not version.strip() for key, version in value.items()):
            raise ValueError("version metadata keys and values must be non-empty")
        return value


class RunContext(SchemaModel):
    """Immutable inputs needed to create a run manifest."""

    run_id: Identifier
    created_at: datetime
    repository_root: Path
    code_version: CodeVersion
    seed: int = Field(ge=0, le=2**63 - 1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_manifest_ref: ArtifactRef

    @classmethod
    def create(
        cls,
        *,
        repository_root: Path,
        seed: int,
        config: Mapping[str, Any],
        environment_manifest_ref: ArtifactRef,
        run_id: str | None = None,
        created_at: datetime | None = None,
    ) -> "RunContext":
        return cls(
            run_id=run_id or f"run-{uuid4().hex}",
            created_at=created_at or datetime.now(UTC),
            repository_root=repository_root.resolve(),
            code_version=detect_code_version(repository_root),
            seed=seed,
            config_sha256=config_sha256(config),
            environment_manifest_ref=environment_manifest_ref,
        )

    def create_manifest(
        self,
        *,
        model_versions: Mapping[str, str] | None = None,
        dataset_versions: Mapping[str, str] | None = None,
    ) -> RunManifest:
        return RunManifest(
            run_id=self.run_id,
            created_at=self.created_at,
            code_version=self.code_version,
            seed=self.seed,
            config_sha256=self.config_sha256,
            environment_manifest_ref=self.environment_manifest_ref,
            model_versions=dict(model_versions or {}),
            dataset_versions=dict(dataset_versions or {}),
        )
