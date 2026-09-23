"""Immutable content-addressed persistence for public M4A trajectories."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.agent import AgentTrajectory


class TrajectoryStore:
    version = "m4a-trajectory-store-v1"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, trajectory: AgentTrajectory) -> ArtifactRef:
        serialized = (
            json.dumps(
                trajectory.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        digest = hashlib.sha256(serialized).hexdigest()
        path = self.root / f"{trajectory.trajectory_id}-{digest[:12]}.json"
        if not path.exists():
            temporary = path.with_suffix(".json.tmp")
            temporary.write_bytes(serialized)
            os.replace(temporary, path)
        elif path.read_bytes() != serialized:
            raise ValueError("content-addressed trajectory path contains different data")
        return ArtifactRef.from_path(
            path,
            artifact_id=f"trajectory-{digest[:20]}",
            media_type="application/json",
        )
