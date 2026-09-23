"""Persistent trace recording without host filesystem paths."""

from __future__ import annotations

import json
import os
from pathlib import Path

from cutagent.schemas.tools import ToolTrace


class ToolTraceRecorder:
    version = "m3a-tool-trace-v1"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, trace: ToolTrace) -> Path:
        path = (
            self.root / trace.execution_id / f"{trace.tool_call_id}-{trace.trace_id}.json"
        ).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("tool trace path escaped trace root")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = ToolTrace.model_validate_json(path.read_text(encoding="utf-8"))
            if existing != trace:
                raise ValueError("tool trace identifier collides with different content")
            return path
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                trace.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path
