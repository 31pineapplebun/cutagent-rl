"""Versioned cache for deterministic M3A editing operations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.tools import ToolSpec
from cutagent.tools.artifacts import ArtifactStore
from cutagent.tools.errors import ToolFailure


def build_tool_cache_key(
    *,
    spec: ToolSpec,
    normalized_arguments: dict[str, Any],
    parents: tuple[ArtifactRef, ...],
    ffmpeg_version: str,
    ffmpeg_contract_version: str,
) -> str:
    payload = {
        "tool_name": spec.name,
        "tool_version": spec.version,
        "arguments": normalized_arguments,
        "parents": [
            {
                "artifact_id": item.artifact_id,
                "sha256": item.sha256,
                "media_type": item.media_type,
            }
            for item in parents
        ],
        "ffmpeg_version": ffmpeg_version,
        "ffmpeg_contract_version": ffmpeg_contract_version,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class ToolCache:
    version = "m3a-tool-cache-v1"

    def __init__(self, root: Path, artifact_store: ArtifactStore) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_store = artifact_store

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("tool cache key must be a lowercase SHA-256 digest")
        path = (self.root / key[:2] / f"{key}.json").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("tool cache path escaped cache root")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get(self, key: str) -> tuple[ArtifactRef, Path] | None:
        path = self._path(key)
        if not path.is_file() or path.is_symlink():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            artifact_id = payload["artifact_id"]
            expected_sha256 = payload["sha256"]
        except (KeyError, TypeError, json.JSONDecodeError):
            return None
        if not isinstance(artifact_id, str) or not isinstance(expected_sha256, str):
            return None
        try:
            reference, object_path = self.artifact_store.get(artifact_id)
        except ToolFailure:
            return None
        if reference.sha256 != expected_sha256:
            return None
        return reference, object_path

    def put(self, key: str, reference: ArtifactRef) -> None:
        path = self._path(key)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "cache_version": self.version,
                    "artifact_id": reference.artifact_id,
                    "sha256": reference.sha256,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
