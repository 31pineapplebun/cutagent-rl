"""Artifact hashing must remain correct without loading whole media files."""

import hashlib
from pathlib import Path

import pytest

from cutagent.core.artifacts import ArtifactRef


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"small fixture",
        bytes(range(256)) * 8_193,
    ],
    ids=["empty", "small", "multiple-chunks"],
)
def test_from_path_streaming_digest_matches_hashlib(tmp_path: Path, payload: bytes) -> None:
    fixture = tmp_path / "fixture.bin"
    fixture.write_bytes(payload)

    reference = ArtifactRef.from_path(
        fixture,
        artifact_id="fixture",
        media_type="application/octet-stream",
    )

    assert reference.sha256 == hashlib.sha256(payload).hexdigest()
    assert reference.size_bytes == len(payload)


def test_from_path_does_not_call_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture.bin"
    fixture.write_bytes(b"stream me")

    def reject_whole_file_read(_path: Path) -> bytes:
        raise AssertionError("ArtifactRef.from_path must not read the whole file at once")

    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_read)

    reference = ArtifactRef.from_path(
        fixture,
        artifact_id="fixture",
        media_type="application/octet-stream",
    )

    assert reference.sha256 == hashlib.sha256(b"stream me").hexdigest()
