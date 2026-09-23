"""Deterministic scene-boundary merge behavior."""

from cutagent.ingestion.scene_splitter import merge_scene_boundaries


def test_scene_ranges_cover_timeline_and_merge_short_neighbors() -> None:
    ranges = merge_scene_boundaries(
        [100, 1000, 1100, 2000, 2950],
        duration_ms=3000,
        minimum_scene_duration_ms=500,
    )
    assert [(value.start_ms, value.end_ms) for value in ranges] == [
        (0, 1000),
        (1000, 2000),
        (2000, 3000),
    ]


def test_scene_boundary_order_and_duplicates_do_not_change_result() -> None:
    first = merge_scene_boundaries(
        [2000, 1000, 1000], duration_ms=3000, minimum_scene_duration_ms=200
    )
    second = merge_scene_boundaries([1000, 2000], duration_ms=3000, minimum_scene_duration_ms=200)
    assert first == second
