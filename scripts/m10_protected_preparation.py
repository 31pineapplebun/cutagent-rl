"""Typed control-plane authorization for protected public-media preparation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cutagent.schemas.base import Identifier, SchemaModel

DEVELOPMENT_PREPARATION_SPLITS = ("dev", "validation")
PROTECTED_PREPARATION_SPLITS = ("locked_test", "adversarial_test")
ALL_PREPARATION_SPLITS = DEVELOPMENT_PREPARATION_SPLITS + PROTECTED_PREPARATION_SPLITS
AUTHORIZATION_VERSION: Literal["m10c-protected-public-preparation-v1"] = (
    "m10c-protected-public-preparation-v1"
)
AUTHORIZATION_LIFETIME = timedelta(hours=24)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


class ProtectedPreparationAuthorization(SchemaModel):
    """Non-policy-visible capability record for one protected public split."""

    authorization_version: Literal["m10c-protected-public-preparation-v1"] = AUTHORIZATION_VERSION
    issuer: Literal["scripts.finalize_after_human_gate"] = "scripts.finalize_after_human_gate"
    purpose: Literal["protected_public_media_preparation_only"] = (
        "protected_public_media_preparation_only"
    )
    benchmark_version: Literal["cutagentbench-v0.1"] = "cutagentbench-v0.1"
    benchmark_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_gate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protected_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_task_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorized_split: Literal["locked_test", "adversarial_test"]
    finalization_run_id: Identifier
    issued_at: datetime
    expires_at: datetime

    @field_validator("issued_at", "expires_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("authorization timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_lifetime(self) -> ProtectedPreparationAuthorization:
        lifetime = self.expires_at - self.issued_at
        if lifetime <= timedelta(0) or lifetime > AUTHORIZATION_LIFETIME:
            raise ValueError("authorization lifetime must be positive and at most 24 hours")
        return self


class ProtectedPublicPreparationAccessRecord(SchemaModel):
    """Audit record for public preparation, separate from Gold and model access."""

    access_type: Literal["protected_public_preparation"] = "protected_public_preparation"
    status: Literal["started", "completed"]
    benchmark_version: Literal["cutagentbench-v0.1"] = "cutagentbench-v0.1"
    benchmark_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["locked_test", "adversarial_test"]
    finalization_run_id: Identifier
    authorization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_task_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=0)
    prepared_video_count: int | None = Field(default=None, ge=0)
    started_at: datetime
    completed_at: datetime | None = None
    protected_gold_deserialized: Literal[False] = False
    policy_visible: Literal[False] = False

    @field_validator("started_at", "completed_at")
    @classmethod
    def require_optional_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("access timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_completion(self) -> ProtectedPublicPreparationAccessRecord:
        if self.status == "started" and self.completed_at is not None:
            raise ValueError("started preparation cannot have completed_at")
        if self.status == "completed" and self.completed_at is None:
            raise ValueError("completed preparation requires completed_at")
        return self


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _benchmark_marker(benchmark_root: Path) -> str:
    marker_path = benchmark_root / "benchmark_manifest.sha256"
    fields = marker_path.read_text(encoding="utf-8").split()
    if not fields or len(fields[0]) != 64:
        raise ValueError("benchmark manifest hash marker is malformed")
    return fields[0]


def _validate_control_plane(
    *,
    benchmark_root: Path,
    human_gate_path: Path,
    protected_plan_path: Path,
) -> tuple[str, str, str, dict[str, object]]:
    benchmark_hash = _benchmark_marker(benchmark_root)
    human_gate = _load_json_object(human_gate_path, label="human gate acceptance")
    if human_gate.get("status") != "PASSED":
        raise PermissionError("protected preparation requires an accepted human gate")
    human_hash = file_sha256(human_gate_path)
    plan = _load_json_object(protected_plan_path, label="protected evaluation plan")
    plan_hash = file_sha256(protected_plan_path)
    benchmark = plan.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("protected plan benchmark declaration is missing")
    if benchmark.get("version") != "cutagentbench-v0.1":
        raise ValueError("protected plan benchmark version mismatch")
    if benchmark.get("manifest_sha256") != benchmark_hash:
        raise ValueError("protected plan benchmark hash mismatch")
    if plan.get("human_gate_acceptance_sha256") != human_hash:
        raise ValueError("protected plan human-gate hash mismatch")
    if plan.get("declared_splits") != list(PROTECTED_PREPARATION_SPLITS):
        raise ValueError("protected plan split declaration mismatch")
    return benchmark_hash, human_hash, plan_hash, plan


def issue_protected_preparation_authorization(
    *,
    split: str,
    benchmark_root: Path,
    human_gate_path: Path,
    protected_plan_path: Path,
    finalization_run_id: str,
    issued_at: datetime | None = None,
) -> ProtectedPreparationAuthorization:
    if split not in PROTECTED_PREPARATION_SPLITS:
        raise ValueError("authorization may be issued only for declared protected splits")
    benchmark_hash, human_hash, plan_hash, _ = _validate_control_plane(
        benchmark_root=benchmark_root,
        human_gate_path=human_gate_path,
        protected_plan_path=protected_plan_path,
    )
    public_manifest = benchmark_root / "public" / f"{split}.json"
    timestamp = issued_at or datetime.now(UTC)
    return ProtectedPreparationAuthorization(
        benchmark_manifest_sha256=benchmark_hash,
        human_gate_sha256=human_hash,
        protected_plan_sha256=plan_hash,
        public_task_manifest_sha256=file_sha256(public_manifest),
        authorized_split=split,  # type: ignore[arg-type]
        finalization_run_id=finalization_run_id,
        issued_at=timestamp,
        expires_at=timestamp + AUTHORIZATION_LIFETIME,
    )


def validate_preparation_request(
    *,
    split: str,
    benchmark_root: Path,
    authorization_path: Path | None,
    human_gate_path: Path | None,
    protected_plan_path: Path | None,
    now: datetime | None = None,
) -> ProtectedPreparationAuthorization | None:
    if split in DEVELOPMENT_PREPARATION_SPLITS:
        if any(
            item is not None for item in (authorization_path, human_gate_path, protected_plan_path)
        ):
            raise ValueError("development preparation must not receive protected authorization")
        return None
    if split not in PROTECTED_PREPARATION_SPLITS:
        raise ValueError("split is not authorized for media preparation")
    if authorization_path is None or human_gate_path is None or protected_plan_path is None:
        raise PermissionError(
            "protected preparation requires finalizer authorization, human gate, and frozen plan"
        )
    authorization = ProtectedPreparationAuthorization.model_validate_json(
        authorization_path.read_text(encoding="utf-8")
    )
    timestamp = now or datetime.now(UTC)
    if timestamp < authorization.issued_at - timedelta(minutes=5):
        raise PermissionError("protected preparation authorization is not yet valid")
    if timestamp > authorization.expires_at:
        raise PermissionError("protected preparation authorization has expired")
    if authorization.authorized_split != split:
        raise PermissionError("protected preparation authorization split mismatch")
    benchmark_hash, human_hash, plan_hash, _ = _validate_control_plane(
        benchmark_root=benchmark_root,
        human_gate_path=human_gate_path,
        protected_plan_path=protected_plan_path,
    )
    expected = {
        "benchmark_manifest_sha256": benchmark_hash,
        "human_gate_sha256": human_hash,
        "protected_plan_sha256": plan_hash,
        "public_task_manifest_sha256": file_sha256(benchmark_root / "public" / f"{split}.json"),
    }
    for field, value in expected.items():
        if getattr(authorization, field) != value:
            raise PermissionError(f"protected preparation authorization {field} mismatch")
    return authorization


def authorization_sha256(authorization: ProtectedPreparationAuthorization) -> str:
    return canonical_sha256(authorization.model_dump(mode="json"))
