"""Small content-addressed cache with canonical, version-aware keys."""

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from cutagent.core.errors import CacheError
from cutagent.core.run_context import canonical_config_json


class CacheKeyBuilder:
    """Build keys from source identity, operation, config, and tool versions."""

    @staticmethod
    def build(
        *,
        source_sha256: str,
        operation: str,
        config: Mapping[str, Any],
        tool_versions: Mapping[str, str],
    ) -> str:
        payload = {
            "source_sha256": source_sha256,
            "operation": operation,
            "config": dict(config),
            "tool_versions": dict(tool_versions),
        }
        return hashlib.sha256(canonical_config_json(payload).encode("utf-8")).hexdigest()


class ContentAddressedCache:
    """Filesystem cache whose entries are isolated by operation and key."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def entry_dir(self, operation: str, key: str) -> Path:
        if not operation or not operation.replace("_", "").isalnum():
            raise CacheError("cache operation must contain only alphanumerics and underscores")
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise CacheError("cache key must be a lowercase SHA-256 digest")
        path = self.root / operation / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, operation: str, key: str, filename: str) -> Path:
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise CacheError("cache filename must be a single safe path component")
        return self.entry_dir(operation, key) / filename

    def read_json(self, operation: str, key: str, filename: str) -> JsonValue | None:
        path = self.path(operation, key, filename)
        if not path.is_file():
            return None
        try:
            value: JsonValue = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CacheError(f"invalid cache JSON at {path}") from error
        return value

    def write_json(self, operation: str, key: str, filename: str, value: JsonValue) -> Path:
        path = self.path(operation, key, filename)
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if path.exists():
            if path.read_bytes() != serialized:
                raise CacheError(
                    f"immutable cache entry already exists with different bytes: {path}"
                )
            return path
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path
