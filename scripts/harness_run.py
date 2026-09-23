"""Run one real local-model trim task without training or benchmark dependencies."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from cutagent.agent.harness import HarnessOutcomeBuilder, create_harness_registry, harness_config
from cutagent.agent.m4b_runtime import M4BAgentRuntime
from cutagent.agent.m4b_state_reducer import M4BStateReducer
from cutagent.models.harness_policy import HarnessPolicyBackend
from cutagent.schemas.media import IngestionResult
from cutagent.schemas.perception import PerceptionResult
from cutagent.schemas.task_input import DurationConstraint, TaskInput


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--duration-ms", type=int, required=True)
    parser.add_argument("--ingestion", type=Path, help="Existing public ingestion JSON")
    parser.add_argument("--perception", type=Path, help="Matching public perception JSON")
    parser.add_argument(
        "--variant", choices=("handoff_only", "compact_recovery"), default="compact_recovery"
    )
    args = parser.parse_args()
    if args.duration_ms <= 0:
        parser.error("duration must be positive")
    if (args.ingestion is None) != (args.perception is None):
        parser.error("--ingestion and --perception must be supplied together")
    ingestion = (
        IngestionResult.model_validate_json(args.ingestion.read_text(encoding="utf-8"))
        if args.ingestion is not None
        else None
    )
    root = args.output.resolve()
    if root.exists():
        parser.error("output must be a new directory; existing evidence is never overwritten")
    registry = create_harness_registry(root / "tools")
    source = registry.artifact_store.import_file(
        args.input.resolve(strict=True),
        media_type="video/mp4",
        artifact_id=ingestion.video.video_id if ingestion else "input-video",
    )
    if ingestion is not None:
        from cutagent.agent.harness_retrieval import prepared_search_tool

        perception = PerceptionResult.model_validate_json(
            args.perception.read_text(encoding="utf-8")
        )
        if source.sha256 != ingestion.video.source.sha256:
            parser.error("input hash does not match the prepared ingestion")
        registry.register(
            prepared_search_tool(
                ingestion=ingestion,
                perception=perception,
                index_root=root / "index",
                model_cache=args.model_cache.resolve(strict=True),
            )
        )
    task = TaskInput(
        task_id="user-task",
        video_ref=source,
        instruction=args.instruction,
        user_constraints=(
            DurationConstraint(
                min_ms=max(0, args.duration_ms - 100), max_ms=args.duration_ms + 100
            ),
        ),
    )
    config = harness_config(args.variant)
    (root / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    policy = HarnessPolicyBackend(model_cache=args.model_cache.resolve(strict=True))
    runtime = M4BAgentRuntime(
        registry=registry,
        policy_model=policy,
        artifact_root=root / "agent",
        outcome_builder=HarnessOutcomeBuilder(),
    )
    trajectory = runtime.run(task, config=config, run_id="harness-user-run")
    (root / "trajectory.json").write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")
    replay = (
        M4BStateReducer.replay(trajectory.initial_state, trajectory.events)
        == trajectory.final_state
    )
    output = None
    if trajectory.final_output_artifact is not None:
        _, path = registry.artifact_store.get(trajectory.final_output_artifact.artifact_id)
        output = root / "output.mp4"
        shutil.copyfile(path, output)
    result = {
        "terminal_reason": trajectory.terminal_reason,
        "state_replay_identical": replay,
        "output": str(output) if output else None,
        "trajectory": str(root / "trajectory.json"),
        "semantic_correctness_claimed": False,
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if trajectory.terminal_reason == "SUCCESS" and replay else 2


if __name__ == "__main__":
    raise SystemExit(main())
