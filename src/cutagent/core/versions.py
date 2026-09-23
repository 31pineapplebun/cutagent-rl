"""Source version detection without assuming a committed repository."""

import hashlib
import subprocess
from pathlib import Path
from typing import Literal

from cutagent.schemas.base import NonEmptyStr, SchemaModel


class CodeVersion(SchemaModel):
    identifier: NonEmptyStr
    vcs: Literal["git", "workspace"]
    revision: NonEmptyStr | None = None
    dirty: bool


def _git(
    root: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _workspace_hash(root: Path) -> str:
    listing = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if listing.returncode == 0:
        relative_paths = sorted(path for path in listing.stdout.split("\x00") if path)
    else:
        relative_paths = sorted(
            str(path.relative_to(root)).replace("\\", "/")
            for path in root.rglob("*")
            if path.is_file() and ".git" not in path.parts
        )

    digest = hashlib.sha256()
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.is_file():
            continue
        digest.update(relative_path.replace("\\", "/").encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def detect_code_version(root: Path) -> CodeVersion:
    """Return a Git revision plus dirty content identity, or a workspace hash."""

    resolved = root.resolve()
    inside = _git(resolved, "rev-parse", "--is-inside-work-tree")
    workspace_hash = _workspace_hash(resolved)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return CodeVersion(
            identifier=f"workspace-sha256:{workspace_hash}",
            vcs="workspace",
            dirty=True,
        )

    head = _git(resolved, "rev-parse", "HEAD")
    status = _git(resolved, "status", "--porcelain", "--untracked-files=all")
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    if head.returncode == 0:
        revision = head.stdout.strip()
        identifier = f"git:{revision}"
        if dirty:
            identifier = f"{identifier}+workspace-sha256:{workspace_hash}"
        return CodeVersion(
            identifier=identifier,
            vcs="git",
            revision=revision,
            dirty=dirty,
        )

    return CodeVersion(
        identifier=f"git:unborn+workspace-sha256:{workspace_hash}",
        vcs="git",
        dirty=True,
    )
