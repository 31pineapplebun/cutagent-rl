"""Read-only audit of every frozen development-regression trajectory and score."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cutagent_evaluation.harness_regression import HarnessCase, score_run

from cutagent.agent.m4b_state_reducer import M4BStateReducer
from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.m4b_agent import M4BAgentTrajectory
from cutagent.tools.artifacts import ArtifactStore


class RelocatedEvidenceStore(ArtifactStore):
    """Read an archived store at a new path while checking IDs, bytes and sizes."""

    def get(self, artifact_id: str) -> tuple[ArtifactRef, Path]:
        manifest_path = self._manifest_path(artifact_id)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        original = ArtifactRef.model_validate(payload["artifact"])
        relative = Path(payload["object_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("archived artifact path is unsafe")
        path = self._assert_controlled(self.root / relative)
        observed = ArtifactRef.from_path(
            path, artifact_id=original.artifact_id, media_type=original.media_type
        )
        if (
            original.artifact_id != artifact_id
            or observed.sha256 != original.sha256
            or observed.size_bytes != original.size_bytes
        ):
            raise ValueError("archived artifact identity or bytes changed")
        return observed, path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path) -> dict[str, object]:
    frozen = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    assert frozen["manifest_sha256"] == _sha256(root / "private_manifest.json")
    manifest = json.loads((root / "private_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == "harness-development-regression-v1"
    cases = [HarnessCase.model_validate(row) for row in manifest["cases"]]
    assert len(cases) == 20 and len({case.task_id for case in cases}) == 20
    assert (root / "run_complete.json").is_file()
    assert len(list(root.glob("runs/*/*/trajectory.json"))) == 40
    assert len(list(root.glob("runs/*/*/score.json"))) == 40
    checked = 0
    triggers = 0
    for case in cases:
        for variant in ("handoff_only", "compact_recovery"):
            directory = root / "runs" / variant / case.task_id
            trajectory_path = directory / "trajectory.json"
            trajectory = M4BAgentTrajectory.model_validate_json(
                trajectory_path.read_text(encoding="utf-8")
            )
            stored = json.loads((directory / "score.json").read_text(encoding="utf-8"))
            assert stored["trajectory_sha256"] == _sha256(trajectory_path)
            assert trajectory.run_id == f"{variant}-{case.task_id}"
            assert trajectory.task_input.task_id == case.task_id
            assert trajectory.task_input.instruction == case.instruction
            assert trajectory.task_input.video_ref.sha256 == manifest["source"]["sha256"]
            assert trajectory.runtime_config.model_dump(mode="json") == manifest["configs"][variant]
            assert trajectory.model.model_dump(mode="json") == manifest["model"]
            assert (
                M4BStateReducer.replay(trajectory.initial_state, trajectory.events)
                == trajectory.final_state
            )
            actual = score_run(
                case,
                trajectory,
                RelocatedEvidenceStore(directory / "tools" / "artifacts"),
                directory / "independent_audit",
            )
            for key, value in actual.items():
                assert stored[key] == value, (
                    case.task_id,
                    variant,
                    key,
                    value,
                    stored[key],
                    actual,
                )
            if case.category == "injected":
                trigger = json.loads(
                    (directory / "private_trigger.json").read_text(encoding="utf-8")
                )
                assert stored["injection_triggered"] == trigger["triggered"]
                assert trigger["triggered"]
                assert trigger["public_tool_status"] == "timeout"
                assert trigger["trigger_source"] == "tool_observation"
                triggers += 1
            else:
                assert not stored["injection_triggered"]
            if trajectory.final_output_artifact is not None:
                assert (directory / "output.mp4").is_file()
                assert _sha256(directory / "output.mp4") == trajectory.final_output_artifact.sha256
            else:
                assert not (directory / "output.mp4").exists()
            checked += 1
    return {"trajectory_count": checked, "trigger_count": triggers, "all_scores_recomputed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
