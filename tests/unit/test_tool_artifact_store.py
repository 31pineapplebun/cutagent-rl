"""Filesystem and immutability tests for the controlled M3A artifact store."""

from pathlib import Path

import pytest

from cutagent.schemas.tools import ToolExecutionContext
from cutagent.tools.artifacts import ArtifactStore
from cutagent.tools.errors import ToolFailure


def _context(*artifact_ids: str, execution_id: str = "exec-001") -> ToolExecutionContext:
    return ToolExecutionContext(
        execution_id=execution_id,
        allowed_output_root_id="tests",
        allowed_artifact_ids=artifact_ids,
        allowed_capabilities=("media.inspect",),
    )


def test_artifact_authorization_is_execution_scoped(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"not-media-but-addressable")
    store = ArtifactStore(tmp_path / "store")
    reference = store.import_file(source, media_type="video/mp4")
    assert (
        store.resolve_allowed(reference.artifact_id, _context(reference.artifact_id))[0]
        == reference
    )
    with pytest.raises(ToolFailure, match="not authorized"):
        store.resolve_allowed(reference.artifact_id, _context(execution_id="other-exec"))


def test_manifest_traversal_and_absolute_paths_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fixture")
    store = ArtifactStore(tmp_path / "store")
    reference = store.import_file(source, media_type="video/mp4")
    manifest = store.manifests_root / f"{reference.artifact_id}.json"
    payload = manifest.read_text(encoding="utf-8").replace(
        f"objects/{reference.sha256[:2]}/{reference.sha256}.mp4",
        "../../outside.mp4",
    )
    manifest.write_text(payload, encoding="utf-8")
    with pytest.raises(ToolFailure, match="unsafe"):
        store.get(reference.artifact_id)


def test_artifact_identifier_cannot_be_rebound(tmp_path: Path) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    store = ArtifactStore(tmp_path / "store")
    store.import_file(first, media_type="video/mp4", artifact_id="stable-id")
    with pytest.raises(ToolFailure, match="different content"):
        store.import_file(second, media_type="video/mp4", artifact_id="stable-id")


def test_symlink_import_is_rejected_when_platform_supports_it(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    link = tmp_path / "link.mp4"
    source.write_bytes(b"fixture")
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ToolFailure, match="symbolic-link"):
        ArtifactStore(tmp_path / "store").import_file(link, media_type="video/mp4")
