"""Small immutable-artifact helpers shared by perception backends."""

import json
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from cutagent.core.artifacts import ArtifactRef


def write_json_artifact(
    path: Path,
    payload: JsonValue | dict[str, Any],
    *,
    artifact_prefix: str,
) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if path.exists() and path.read_bytes() != serialized:
        raise ValueError(f"immutable provenance artifact differs: {path}")
    if not path.exists():
        path.write_bytes(serialized)
    provisional = ArtifactRef.from_path(
        path,
        artifact_id=f"{artifact_prefix}-provisional",
        media_type="application/json",
    )
    return ArtifactRef(
        **{
            **provisional.model_dump(),
            "artifact_id": f"{artifact_prefix}-{provisional.sha256[:20]}",
        }
    )
