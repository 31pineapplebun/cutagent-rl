"""Private validation-set and recovery-funnel contracts for M4B.5."""

from __future__ import annotations

from cutagent_evaluation.m2b_dataset import HeldOutSceneGold, HeldOutVideoGold
from cutagent_evaluation.m4b5_recovery import build_m4b5_recovery_set

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.media import TimeRange


def _video(index: int) -> HeldOutVideoGold:
    actions = ("stationary", "appear", "move_right", "disappear")
    return HeldOutVideoGold(
        source_group_id=f"m2b-heldout-source-{index + 1:02d}",
        seed=index,
        source_sha256=f"{index + 1:064x}",
        scenes=tuple(
            HeldOutSceneGold(
                scene_index=scene_index,
                nominal_time_range=TimeRange(
                    start_ms=scene_index * 3000,
                    end_ms=(scene_index + 1) * 3000,
                ),
                primary_entity="red square",
                companion_entity="blue circle",
                action=actions[scene_index],  # type: ignore[arg-type]
                ocr_text=f"CASE {index} {scene_index}",
                transcript=f"case {index} scene {scene_index}",
            )
            for scene_index in range(4)
        ),
    )


def test_recovery_set_is_new_private_and_covers_all_failures() -> None:
    videos = tuple(_video(index) for index in range(6))
    refs = {
        video.source_group_id: ArtifactRef(
            artifact_id=f"video-{index}",
            uri=f"file:///synthetic/video-{index}.mp4",
            sha256=f"{index + 11:064x}",
            size_bytes=1,
            media_type="video/mp4",
        )
        for index, video in enumerate(videos)
    }
    cases = build_m4b5_recovery_set(
        videos,
        refs,
        excluded_source_groups={"m2b-heldout-source-07", "m2b-heldout-source-08"},
    )
    assert len(cases) == 34
    assert len({item.task_input.task_id for item in cases}) == 34
    assert {item.gold.injection.failure_type for item in cases} == {
        "search_no_results",
        "tool_timeout",
        "invalid_tool_arguments",
        "artifact_not_allowed",
        "post_execution_validation_failure",
        "corrupt_media",
        "incompatible_concat_inputs",
        "invalid_subtitle_timing",
        "output_size_limit",
        "repeated_editor_stagnation",
    }
    public = "\n".join(item.task_input.model_dump_json() for item in cases).casefold()
    assert "failure_type" not in public
    assert "expected_recovery_operations" not in public
    assert "source_group_id" not in public
    assert '"split"' not in public
    impossible = [item for item in cases if item.gold.task_gold.category == "impossible"]
    assert len(impossible) == 4
    assert all(
        item.gold.injection.expected_recovery_operations == ("cannot_recover",)
        for item in impossible
    )
    solvable_empty_search = [
        item
        for item in cases
        if item.gold.injection.failure_type == "search_no_results"
        and item.gold.task_gold.category != "impossible"
    ]
    assert all(
        "cannot_recover" not in item.gold.injection.expected_recovery_operations
        for item in solvable_empty_search
    )
