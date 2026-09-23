"""Controlled, content-addressed artifact storage for M3A tools."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.tools import ToolExecutionContext, TraceArtifact
from cutagent.tools.errors import ToolFailure

_MEDIA_EXTENSIONS = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "application/json": ".json",
    "text/plain": ".txt",
    "application/x-subrip": ".srt",
}


def trace_artifact(reference: ArtifactRef) -> TraceArtifact:
    return TraceArtifact(
        artifact_id=reference.artifact_id,
        sha256=reference.sha256,
        media_type=reference.media_type,
        size_bytes=reference.size_bytes,
    )


class ArtifactStore:
    """Maps opaque artifact IDs to files under one resolved storage root."""

    version = "m3a-artifact-store-v1"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.objects_root = self.root / "objects"
        self.manifests_root = self.root / "manifests"
        self.work_root = self.root / "work"
        for path in (self.objects_root, self.manifests_root, self.work_root):
            path.mkdir(parents=True, exist_ok=True)
        self._assert_controlled(self.root)

    def _assert_controlled(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ToolFailure("filesystem_violation", "artifact path escaped the tool workspace")
        return resolved

    @staticmethod
    def _extension(media_type: str) -> str:
        extension = _MEDIA_EXTENSIONS.get(media_type)
        if extension is None:
            raise ToolFailure("filesystem_violation", "artifact media type is not allowed")
        return extension

    def _object_path(self, sha256: str, media_type: str) -> Path:
        path = self.objects_root / sha256[:2] / f"{sha256}{self._extension(media_type)}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return self._assert_controlled(path)

    def _manifest_path(self, artifact_id: str) -> Path:
        return self._assert_controlled(self.manifests_root / f"{artifact_id}.json")

    def _write_manifest(self, reference: ArtifactRef, object_path: Path) -> None:
        relative = object_path.relative_to(self.root).as_posix()
        payload = {
            "schema_version": "1.0",
            "artifact": reference.model_dump(mode="json"),
            "object_relative_path": relative,
        }
        path = self._manifest_path(reference.artifact_id)
        if path.exists():
            try:
                existing_payload = json.loads(path.read_text(encoding="utf-8"))
                existing = ArtifactRef.model_validate(existing_payload["artifact"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ToolFailure(
                    "corrupt_media",
                    "existing artifact manifest is invalid",
                ) from exc
            if existing != reference:
                raise ToolFailure(
                    "filesystem_violation",
                    "artifact identifier is already bound to different content",
                )
            return
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _store_file(
        self,
        source: Path,
        *,
        media_type: str,
        artifact_id: str,
        move: bool,
    ) -> ArtifactRef:
        if source.is_symlink():
            raise ToolFailure("filesystem_violation", "symbolic-link artifacts are forbidden")
        source_resolved = source.resolve(strict=True)
        if not source_resolved.is_file():
            raise ToolFailure("artifact_not_found", "artifact source is not a regular file")
        provisional = ArtifactRef.from_path(
            source_resolved,
            artifact_id=artifact_id,
            media_type=media_type,
        )
        destination = self._object_path(provisional.sha256, media_type)
        if destination.is_symlink():
            raise ToolFailure("filesystem_violation", "artifact object cannot be a symbolic link")
        if not destination.exists():
            if move:
                self._assert_controlled(source_resolved)
                os.replace(source_resolved, destination)
            else:
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                shutil.copyfile(source_resolved, temporary)
                os.replace(temporary, destination)
        elif move and source_resolved != destination:
            self._assert_controlled(source_resolved).unlink()
        reference = ArtifactRef.from_path(
            destination,
            artifact_id=artifact_id,
            media_type=media_type,
        )
        self._write_manifest(reference, destination)
        return reference

    def import_file(
        self,
        source: Path,
        *,
        media_type: str,
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        """Trusted host-side import; ToolCall never contains this source path."""

        if source.is_symlink():
            raise ToolFailure("filesystem_violation", "symbolic-link imports are forbidden")
        provisional = ArtifactRef.from_path(
            source.resolve(strict=True),
            artifact_id=artifact_id or "import-provisional",
            media_type=media_type,
        )
        stable_id = artifact_id or f"input-{provisional.sha256[:20]}"
        return self._store_file(
            source,
            media_type=media_type,
            artifact_id=stable_id,
            move=False,
        )

    def commit_output(
        self,
        staging_path: Path,
        *,
        media_type: str,
        artifact_prefix: str,
    ) -> ArtifactRef:
        staging = self._assert_controlled(staging_path)
        provisional = ArtifactRef.from_path(
            staging,
            artifact_id=f"{artifact_prefix}-provisional",
            media_type=media_type,
        )
        return self._store_file(
            staging,
            media_type=media_type,
            artifact_id=f"{artifact_prefix}-{provisional.sha256[:20]}",
            move=True,
        )

    def put_json(self, payload: Any, *, artifact_prefix: str) -> ArtifactRef:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        digest_path = self.work_root / f"{artifact_prefix}-{os.getpid()}.json"
        digest_path = self._assert_controlled(digest_path)
        digest_path.write_bytes(serialized)
        return self.commit_output(
            digest_path,
            media_type="application/json",
            artifact_prefix=artifact_prefix,
        )

    def get(self, artifact_id: str) -> tuple[ArtifactRef, Path]:
        manifest_path = self._manifest_path(artifact_id)
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ToolFailure("artifact_not_found", "artifact identifier is not registered")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            reference = ArtifactRef.model_validate(payload["artifact"])
            relative = Path(payload["object_relative_path"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ToolFailure("corrupt_media", "artifact manifest is invalid") from exc
        if relative.is_absolute() or ".." in relative.parts:
            raise ToolFailure("filesystem_violation", "artifact manifest path is unsafe")
        path = self._assert_controlled(self.root / relative)
        if path.is_symlink():
            raise ToolFailure("filesystem_violation", "artifact object is a symbolic link")
        if not path.is_file():
            raise ToolFailure("artifact_not_found", "artifact object is missing")
        observed = ArtifactRef.from_path(
            path,
            artifact_id=reference.artifact_id,
            media_type=reference.media_type,
        )
        if observed != reference:
            raise ToolFailure("corrupt_media", "artifact content no longer matches its manifest")
        return reference, path

    def resolve_allowed(
        self, artifact_id: str, context: ToolExecutionContext
    ) -> tuple[ArtifactRef, Path]:
        if artifact_id not in context.allowed_artifact_ids:
            raise ToolFailure(
                "artifact_not_allowed",
                "artifact is not authorized for this execution",
            )
        return self.get(artifact_id)

    def create_work_dir(self, context: ToolExecutionContext, tool_call_id: str) -> Path:
        path = self.work_root / context.allowed_output_root_id / context.execution_id / tool_call_id
        controlled = self._assert_controlled(path)
        if controlled.exists() and controlled.is_symlink():
            raise ToolFailure("filesystem_violation", "tool work directory cannot be a symlink")
        controlled.mkdir(parents=True, exist_ok=True)
        return controlled
