"""Content-addressed M1A cache key contracts."""

from cutagent.ingestion.cache import CacheKeyBuilder


def _key(operation: str, config: dict[str, object]) -> str:
    return CacheKeyBuilder.build(
        source_sha256="a" * 64,
        operation=operation,
        config=config,
        tool_versions={"ffmpeg": "ffmpeg-test-v1"},
    )


def test_same_source_config_and_tool_version_has_same_key() -> None:
    first = _key("scene_split", {"threshold": 0.3, "minimum_ms": 500})
    second = _key("scene_split", {"minimum_ms": 500, "threshold": 0.3})
    assert first == second


def test_scene_threshold_changes_scene_cache_key() -> None:
    assert _key("scene_split", {"threshold": 0.3}) != _key("scene_split", {"threshold": 0.4})


def test_keyframe_strategy_changes_keyframe_cache_key() -> None:
    assert _key("keyframe_extract", {"strategy": "midpoint"}) != _key(
        "keyframe_extract", {"strategy": "uniform", "count": 2}
    )


def test_tool_version_changes_cache_key() -> None:
    first = CacheKeyBuilder.build(
        source_sha256="a" * 64,
        operation="ffprobe",
        config={},
        tool_versions={"ffprobe": "v1"},
    )
    second = CacheKeyBuilder.build(
        source_sha256="a" * 64,
        operation="ffprobe",
        config={},
        tool_versions={"ffprobe": "v2"},
    )
    assert first != second
