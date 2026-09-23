"""Immutable references to reproducible artifacts."""

import hashlib
from pathlib import Path

from pydantic import Field, field_validator

from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel

_HASH_CHUNK_SIZE_BYTES = 1024 * 1024


def _sha256_file(path: Path) -> tuple[str, int]:
    """Hash a file incrementally and return the digest and opened-file size."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE_BYTES):
            digest.update(chunk)
        size_bytes = handle.tell()
    return digest.hexdigest(), size_bytes


class ArtifactRef(SchemaModel):
    """Internal artifact reference; policy views must redact URI and hash."""

    artifact_id: Identifier
    uri: NonEmptyStr
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: NonEmptyStr
    size_bytes: int = Field(ge=0)

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("artifact URI cannot contain NUL")
        return value

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        artifact_id: str,
        media_type: str,
    ) -> "ArtifactRef":
        resolved = path.resolve(strict=True)
        digest, size_bytes = _sha256_file(resolved)
        return cls(
            artifact_id=artifact_id,
            uri=resolved.as_uri(),
            sha256=digest,
            media_type=media_type,
            size_bytes=size_bytes,
        )
