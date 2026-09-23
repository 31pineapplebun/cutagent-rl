"""Run three qualitative M4A tasks on existing explicitly licensed M1B.5 clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cutagent.agent.runtime import AgentRuntime
from cutagent.agent.trajectory import TrajectoryStore
from cutagent.models.qwen_policy import Qwen3VLPolicyBackend
from cutagent.schemas.agent import AgentRuntimeConfig
from cutagent.schemas.task_input import (
    AspectRatioConstraint,
    DurationConstraint,
    TaskInput,
)
from cutagent.tools.factory import create_media_tool_registry


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--licensed-root", type=Path, default=Path("artifacts/m1b_5/real_media"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m4a/licensed_real"))
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    licensed_root = args.licensed_root.resolve()
    manifest = json.loads((licensed_root / "provenance.json").read_text(encoding="utf-8"))
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise RuntimeError("licensed provenance manifest has no source records")
    clips: dict[str, tuple[Path, dict[str, Any]]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for clip in source.get("clips", []):
            if isinstance(clip, dict) and isinstance(clip.get("clip_id"), str):
                path = (licensed_root / str(clip["relative_path"])).resolve(strict=True)
                clips[str(clip["clip_id"])] = (path, source)
    requested = ("pedestrian-01", "traffic-01", "wikipedia20-01")
    if any(item not in clips for item in requested):
        raise RuntimeError("required licensed qualitative clips are missing")
    root = args.artifact_root.resolve()
    registry = create_media_tool_registry(
        root=root / "tool_runtime",
        ffmpeg_executable=args.ffmpeg,
        ffprobe_executable=args.ffprobe,
    )
    source_refs = {
        clip_id: registry.artifact_store.import_file(
            clips[clip_id][0],
            media_type="video/mp4",
            artifact_id=f"licensed-{clip_id}",
        )
        for clip_id in requested
    }
    tasks = (
        TaskInput(
            task_id="m4a-real-pedestrian-trim",
            video_ref=source_refs["pedestrian-01"],
            instruction=(
                "Trim the interval [0,2500) ms from this licensed clip, validate it, "
                "and return the validated output."
            ),
            user_constraints=(DurationConstraint(min_ms=2250, max_ms=2750),),
        ),
        TaskInput(
            task_id="m4a-real-traffic-reframe",
            video_ref=source_refs["traffic-01"],
            instruction=(
                "Reframe this complete licensed clip to portrait 216x384 using crop, "
                "validate it, and return the validated output."
            ),
            user_constraints=(AspectRatioConstraint(width=9, height=16),),
        ),
        TaskInput(
            task_id="m4a-real-speech-speed",
            video_ref=source_refs["wikipedia20-01"],
            instruction=(
                "Change this complete licensed clip to 1.5x speed, validate it, and "
                "return the validated output."
            ),
            user_constraints=(DurationConstraint(min_ms=2750, max_ms=3250),),
        ),
    )
    policy = Qwen3VLPolicyBackend(model_cache=args.model_cache.resolve())
    runtime = AgentRuntime(
        registry=registry,
        policy_model=policy,
        artifact_root=root / "runs",
    )
    store = TrajectoryStore(root / "trajectories")
    rows: list[dict[str, Any]] = []
    for task, clip_id in zip(tasks, requested, strict=True):
        trajectory = runtime.run(
            task,
            config=AgentRuntimeConfig(
                baseline="react",
                policy_view_mode="structured_state",
                max_steps=10,
                max_tool_calls=8,
                max_search_calls=1,
                max_edit_calls=6,
                max_wall_time_ms=600_000,
                max_model_tokens=48_000,
                policy_maximum_new_tokens=768,
            ),
            run_id=f"licensed-{clip_id}",
        )
        reference = store.write(trajectory)
        source = clips[clip_id][1]
        rows.append(
            {
                "task_id": task.task_id,
                "clip_id": clip_id,
                "source_id": source["source_id"],
                "source_page": source["source_page"],
                "license": source["license"],
                "source_sha256": source["sha256"],
                "terminal_reason": trajectory.terminal_reason,
                "tool_names": [item.trace.tool_name for item in trajectory.tool_records],
                "tool_statuses": [item.observation.status for item in trajectory.tool_records],
                "final_output_artifact_id": (
                    trajectory.final_output_artifact.artifact_id
                    if trajectory.final_output_artifact is not None
                    else None
                ),
                "trajectory_artifact_id": reference.artifact_id,
                "latency": trajectory.latency.model_dump(mode="json"),
            }
        )
        print(
            f"{task.task_id}: terminal={trajectory.terminal_reason} "
            f"tools={len(trajectory.tool_records)}",
            flush=True,
        )
    result = {
        "schema_version": "1.0",
        "qualitative_only": True,
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "seed": 20_260_823,
        "rows": rows,
    }
    _write_json(root / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
