"""Content-addressed persistence for reconstructable M4B trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.m4b_agent import M4BAgentTrajectory


class M4BTrajectoryStore:
    version = "m4b-trajectory-store-v1"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, trajectory: M4BAgentTrajectory) -> ArtifactRef:
        payload = json.dumps(
            trajectory.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        path = self.root / f"{digest}.json"
        if not path.exists():
            path.write_text(payload + "\n", encoding="utf-8")
        return ArtifactRef.from_path(
            path,
            artifact_id=f"m4b-trajectory-{digest[:20]}",
            media_type="application/json",
        )
