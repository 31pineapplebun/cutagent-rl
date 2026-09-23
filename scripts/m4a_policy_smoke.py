"""Run one real local Qwen3-VL M4A Agent trajectory through ToolRegistry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from cutagent.agent.runtime import AgentRuntime
from cutagent.agent.trajectory import TrajectoryStore
from cutagent.models.qwen_policy import Qwen3VLPolicyBackend
from cutagent.schemas.agent import AgentRuntimeConfig
from cutagent.schemas.media import IngestionResult
from cutagent.schemas.task_input import DurationConstraint, TaskInput
from cutagent.tools.factory import create_media_tool_registry


def _artifact_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("M4A smoke requires a local file artifact")
    text = unquote(parsed.path)
    if len(text) >= 3 and text[0] == "/" and text[2] == ":":
        text = text[1:]
    return Path(text).resolve(strict=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2b-root", type=Path, default=Path("artifacts/m2b"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m4a/policy_smoke"))
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    ingestion_path = next(iter(sorted((args.m2b_root / "ingestion").glob("*.json"))), None)
    if ingestion_path is None:
        raise RuntimeError("no M2B ingestion result is available for the M4A policy smoke")
    ingestion = IngestionResult.model_validate_json(ingestion_path.read_text(encoding="utf-8"))
    root = args.artifact_root.resolve()
    registry = create_media_tool_registry(
        root=root / "tool_runtime",
        ffmpeg_executable=args.ffmpeg,
        ffprobe_executable=args.ffprobe,
    )
    source = registry.artifact_store.import_file(
        _artifact_path(ingestion.video.source.uri),
        media_type="video/mp4",
        artifact_id=ingestion.video.video_id,
    )
    task = TaskInput(
        task_id="m4a-real-policy-smoke",
        video_ref=source,
        instruction=(
            "Using only registered tools, call trim_video exactly once for [0,3000) ms. "
            "Then call validate_media on that new output exactly once and return that "
            "validated artifact. Do not trim an already trimmed artifact."
        ),
        user_constraints=(DurationConstraint(min_ms=2750, max_ms=3250),),
    )
    policy = Qwen3VLPolicyBackend(model_cache=args.model_cache.resolve())
    runtime = AgentRuntime(
        registry=registry,
        policy_model=policy,
        artifact_root=root / "run",
    )
    trajectory = runtime.run(
        task,
        config=AgentRuntimeConfig(
            baseline="hierarchical",
            policy_view_mode="structured_state",
            max_steps=10,
            max_tool_calls=6,
            max_search_calls=1,
            max_edit_calls=4,
            max_structured_output_repairs=8,
            max_model_tokens=64_000,
            max_wall_time_ms=300_000,
        ),
        run_id="m4a-real-policy-smoke-run",
    )
    trajectory_ref = TrajectoryStore(root / "trajectories").write(trajectory)
    smoke_pass = (
        bool(trajectory.policy_steps)
        and bool(trajectory.tool_records)
        and any(item.observation.status == "success" for item in trajectory.tool_records)
        and trajectory.final_state.terminal_event is not None
    )
    result = {
        "smoke_pass": smoke_pass,
        "model": trajectory.model.model_dump(mode="json"),
        "policy_backend_version": policy.backend_version,
        "model_load_time_ms": policy.load_time_ms,
        "seed": trajectory.runtime_config.seed,
        "terminal_reason": trajectory.terminal_reason,
        "tool_names": [item.trace.tool_name for item in trajectory.tool_records],
        "tool_statuses": [item.observation.status for item in trajectory.tool_records],
        "policy_step_count": len(trajectory.policy_steps),
        "structured_output_repairs": sum(
            item.stats.repair_count for item in trajectory.policy_steps
        ),
        "peak_allocated_bytes": max(
            (item.stats.peak_allocated_bytes or 0 for item in trajectory.policy_steps),
            default=0,
        ),
        "peak_reserved_bytes": max(
            (item.stats.peak_reserved_bytes or 0 for item in trajectory.policy_steps),
            default=0,
        ),
        "latency": trajectory.latency.model_dump(mode="json"),
        "trajectory_artifact_id": trajectory_ref.artifact_id,
        "final_output_artifact_id": (
            trajectory.final_output_artifact.artifact_id
            if trajectory.final_output_artifact is not None
            else None
        ),
    }
    _write_json(root / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if smoke_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
