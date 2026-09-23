"""Persistent component cache, determinism, and invalidation tests."""

from pathlib import Path

import numpy as np

from cutagent.core.artifacts import ArtifactRef
from cutagent.retrieval.index import KeyframeAsset, RetrievalIndexBuilder
from cutagent.schemas.retrieval import RetrievalConfig
from tests.retrieval_fixtures import FakeTextEncoder, FakeVisualEncoder, sample_world_states


def _assets(tmp_path: Path) -> tuple[KeyframeAsset, ...]:
    output = []
    for artifact_id in ("frame-red", "frame-blue"):
        path = tmp_path / f"{artifact_id}.jpg"
        path.write_bytes(artifact_id.encode("utf-8"))
        output.append(
            KeyframeAsset(
                artifact=ArtifactRef.from_path(
                    path, artifact_id=artifact_id, media_type="image/jpeg"
                ),
                path=path,
            )
        )
    return tuple(output)


def test_index_build_is_deterministic_and_second_build_hits_cache(tmp_path: Path) -> None:
    builder = RetrievalIndexBuilder(tmp_path / "index")
    text = FakeTextEncoder()
    visual = FakeVisualEncoder()
    first = builder.build(
        world_states=sample_world_states(),
        keyframe_assets=_assets(tmp_path),
        config=RetrievalConfig(),
        text_encoder=text,
        visual_encoder=visual,
    )
    assert text.calls == 1
    assert visual.image_calls == 1
    cached_text = FakeTextEncoder()
    cached_visual = FakeVisualEncoder()
    second = builder.build(
        world_states=sample_world_states(),
        keyframe_assets=_assets(tmp_path),
        config=RetrievalConfig(),
        text_encoder=cached_text,
        visual_encoder=cached_visual,
    )
    assert cached_text.calls == 0
    assert cached_visual.image_calls == 0
    assert second.cache_hits == {"documents": True, "dense_text": True, "visual": True}
    assert first.documents == second.documents
    assert first.manifest.index_id == second.manifest.index_id
    assert first.dense_embeddings is not None
    assert second.dense_embeddings is not None
    assert first.visual_embeddings is not None
    assert second.visual_embeddings is not None
    assert np.array_equal(first.dense_embeddings, second.dense_embeddings)
    assert np.array_equal(first.visual_embeddings, second.visual_embeddings)


def test_text_config_change_invalidates_only_dense_component(tmp_path: Path) -> None:
    builder = RetrievalIndexBuilder(tmp_path / "index")
    assets = _assets(tmp_path)
    builder.build(
        world_states=sample_world_states(),
        keyframe_assets=assets,
        config=RetrievalConfig(),
        text_encoder=FakeTextEncoder(),
        visual_encoder=FakeVisualEncoder(),
    )
    changed_text = FakeTextEncoder()
    unchanged_visual = FakeVisualEncoder()
    result = builder.build(
        world_states=sample_world_states(),
        keyframe_assets=assets,
        config=RetrievalConfig(text_encoder_max_tokens=256),
        text_encoder=changed_text,
        visual_encoder=unchanged_visual,
    )
    assert result.cache_hits["documents"] is True
    assert result.cache_hits["dense_text"] is False
    assert result.cache_hits["visual"] is True
    assert changed_text.calls == 1
    assert unchanged_visual.image_calls == 0


def test_duplicate_keyframe_artifact_is_embedded_once(tmp_path: Path) -> None:
    states = sample_world_states()
    repeated_reference = (
        states[0].scenes[0].keyframe_evidence[0].model_copy(update={"observed_ms": 600})
    )
    repeated_scene = (
        states[0]
        .scenes[0]
        .model_copy(
            update={
                "keyframe_evidence": (
                    states[0].scenes[0].keyframe_evidence[0],
                    repeated_reference,
                )
            }
        )
    )
    repeated_world = states[0].model_copy(update={"scenes": (repeated_scene, states[0].scenes[1])})
    visual = FakeVisualEncoder()
    memory = RetrievalIndexBuilder(tmp_path / "index").build(
        world_states=(repeated_world,),
        keyframe_assets=_assets(tmp_path),
        config=RetrievalConfig(),
        text_encoder=FakeTextEncoder(),
        visual_encoder=visual,
    )
    assert visual.image_calls == 1
    assert memory.visual_embeddings is not None
    assert memory.visual_embeddings.shape[0] == 3
    assert np.array_equal(memory.visual_embeddings[0], memory.visual_embeddings[1])
