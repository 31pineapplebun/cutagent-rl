"""Validate M4A trajectory replay, cross-baseline task identity, and private-label isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cutagent.agent.state_reducer import StateReducer
from cutagent.schemas.agent import AgentTrajectory

VARIANTS = ("react_structured", "hierarchical_structured", "hierarchical_visual")
FORBIDDEN_PRIVATE_TOKENS = (
    "benchmarkgold",
    "benchmark_gold",
    "source_group_id",
    '"split"',
    "ground_truth",
    "evaluator_metadata",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m4a"))
    parser.add_argument("--expected-task-count", type=int, default=50)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    task_sets: dict[str, tuple[str, ...]] = {}
    digests: dict[str, str] = {}
    replayed = 0
    failure_artifacts = 0
    for variant in VARIANTS:
        paths = tuple(sorted((root / "progress" / variant).glob("*.json")))
        if len(paths) != args.expected_task_count:
            raise RuntimeError(
                f"{variant} has {len(paths)} trajectories; expected {args.expected_task_count}"
            )
        task_ids: list[str] = []
        digest = hashlib.sha256()
        for path in paths:
            raw = path.read_bytes()
            trajectory = AgentTrajectory.model_validate_json(raw)
            replay = StateReducer.replay(trajectory.initial_state, trajectory.events)
            if replay != trajectory.final_state:
                raise RuntimeError(f"trajectory replay differs: {trajectory.trajectory_id}")
            serialized = raw.decode("utf-8").casefold()
            for token in FORBIDDEN_PRIVATE_TOKENS:
                if token in serialized:
                    raise RuntimeError(
                        f"private token {token!r} leaked into {trajectory.trajectory_id}"
                    )
            if trajectory.model.revision != "ebb281ec70b05090aa6165b016eac8ec08e71b17":
                raise RuntimeError("trajectory used an unexpected policy model revision")
            task_ids.append(trajectory.task_input.task_id)
            digest.update(raw)
            replayed += 1
            failure_artifacts += len(trajectory.policy_failures)
        task_sets[variant] = tuple(task_ids)
        digests[variant] = digest.hexdigest()
    if len(set(task_sets.values())) != 1:
        raise RuntimeError("M4A variants did not run exactly the same ordered task IDs")
    result = {
        "schema_version": "1.0",
        "variant_count": len(VARIANTS),
        "task_count_per_variant": args.expected_task_count,
        "trajectory_count": replayed,
        "deterministic_event_replay_count": replayed,
        "private_label_leak_count": 0,
        "policy_failure_artifact_count": failure_artifacts,
        "ordered_task_ids_identical": True,
        "variant_trajectory_digests": digests,
    }
    output = root / "trajectory_validation.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
