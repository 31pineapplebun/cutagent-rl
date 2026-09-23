"""Private M4A task-set and evaluator contract tests."""

from __future__ import annotations

from pathlib import Path

from cutagent_evaluation.m2b_dataset import HeldOutSceneGold, HeldOutVideoGold
from cutagent_evaluation.m4a_agent import build_m4a_development_set

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.media import TimeRange


def test_m4a_development_set_has_50_public_private_isolated_cases() -> None:
    videos: list[HeldOutVideoGold] = []
    refs: dict[str, ArtifactRef] = {}
    for index in range(6):
        digest = f"{index + 1:064x}"
        group_id = f"m2b-heldout-source-{index + 1:02d}"
        actions = ("stationary", "appear", "move_right", "disappear")
        gold = HeldOutVideoGold(
            source_group_id=group_id,
            seed=100 + index,
            source_sha256=digest,
            scenes=tuple(
                HeldOutSceneGold(
                    scene_index=scene_index,
                    nominal_time_range=TimeRange(
                        start_ms=scene_index * 3000,
                        end_ms=(scene_index + 1) * 3000,
                    ),
                    primary_entity=f"object {index}",
                    companion_entity=f"companion {scene_index}",
                    action=actions[scene_index],
                    ocr_text=f"M2B{index + 1:02d} S{scene_index + 1}",
                    transcript=f"Video {index + 1} scene {scene_index + 1} marker.",
                )
                for scene_index in range(4)
            ),
        )
        videos.append(gold)
        refs[gold.source_group_id] = ArtifactRef.from_path(
            Path(__file__),
            artifact_id=f"video-{gold.source_sha256[:20]}",
            media_type="application/json",
        )
    cases = build_m4a_development_set(videos, refs)
    assert len(cases) == 50
    assert len({item.task_input.task_id for item in cases}) == 50
    public = "\n".join(item.task_input.model_dump_json() for item in cases).casefold()
    for forbidden in ("source_group_id", '"split"', "gold_id", "relevant_time_ranges"):
        assert forbidden not in public
    categories = {item.gold.category for item in cases}
    assert categories == {
        "search_trim",
        "search_trim_validate",
        "retrieve_concat",
        "subtitle",
        "speed",
        "reframe",
        "multi_constraint",
        "hard_negative",
        "impossible",
    }
