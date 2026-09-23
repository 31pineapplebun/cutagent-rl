"""Deterministic local index construction with component-level cache invalidation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cutagent.core.artifacts import ArtifactRef
from cutagent.core.errors import CacheError
from cutagent.core.run_context import canonical_config_json
from cutagent.ingestion.cache import CacheKeyBuilder, ContentAddressedCache
from cutagent.retrieval.documents import SceneDocument, SceneDocumentBuilder
from cutagent.retrieval.protocols import TextEncoder, VisualEncoder
from cutagent.retrieval.sparse import BM25Index
from cutagent.retrieval.store import RetrievalMemory, VisualVectorRow
from cutagent.schemas.perception import EvidenceRef, VideoWorldState
from cutagent.schemas.retrieval import RetrievalConfig, RetrievalIndexManifest


@dataclass(frozen=True, slots=True)
class KeyframeAsset:
    artifact: ArtifactRef
    path: Path


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_config_json({"value": value}).encode("utf-8")).hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _write_npy_immutable(path: Path, array: np.ndarray[Any, Any]) -> None:
    if path.is_file():
        with path.open("rb") as handle:
            existing = np.load(handle, allow_pickle=False)
        if not np.array_equal(existing, array):
            raise CacheError(f"immutable NumPy cache entry differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_visual_rows(value: Sequence[object]) -> tuple[VisualVectorRow, ...]:
    rows: list[VisualVectorRow] = []
    for item in value:
        if not isinstance(item, dict):
            raise CacheError("invalid visual row metadata")
        document_index = item.get("document_index")
        if not isinstance(document_index, int):
            raise CacheError("invalid visual document index")
        rows.append(
            VisualVectorRow(
                document_index=document_index,
                evidence_ref=EvidenceRef.model_validate(item.get("evidence_ref")),
            )
        )
    return tuple(rows)


class RetrievalIndexBuilder:
    """Build and reload small local indices without a vector database."""

    def __init__(self, root: Path) -> None:
        self.cache = ContentAddressedCache(root)

    def build(
        self,
        *,
        world_states: tuple[VideoWorldState, ...],
        keyframe_assets: tuple[KeyframeAsset, ...],
        config: RetrievalConfig,
        text_encoder: TextEncoder | None,
        visual_encoder: VisualEncoder | None,
    ) -> RetrievalMemory:
        started = time.perf_counter()
        if not world_states:
            raise ValueError("at least one world state is required")
        if config.document_template_version != SceneDocumentBuilder.template_version:
            raise ValueError("retrieval config requests an unsupported document template")
        world_hashes = tuple(
            sorted(_sha256_json(world.model_dump(mode="json")) for world in world_states)
        )
        aggregate_world_hash = _sha256_json(world_hashes)
        versions = {
            "numpy": np.__version__,
            "transformers": _package_version("transformers"),
            "torch": _package_version("torch"),
        }

        document_key = CacheKeyBuilder.build(
            source_sha256=aggregate_world_hash,
            operation="scene_documents",
            config={"template_version": config.document_template_version},
            tool_versions={"cutagent": SceneDocumentBuilder.template_version},
        )
        cached_documents = self.cache.read_json("scene_documents", document_key, "documents.json")
        document_hit = cached_documents is not None
        if cached_documents is None:
            documents = SceneDocumentBuilder().build(world_states)
            self.cache.write_json(
                "scene_documents",
                document_key,
                "documents.json",
                [document.model_dump(mode="json") for document in documents],
            )
        else:
            if not isinstance(cached_documents, list):
                raise CacheError("scene document cache must contain a JSON list")
            documents = tuple(SceneDocument.model_validate(item) for item in cached_documents)
        document_path = self.cache.path("scene_documents", document_key, "documents.json")
        document_artifact = ArtifactRef.from_path(
            document_path,
            artifact_id=f"m2a-documents-{document_key[:16]}",
            media_type="application/json",
        )

        sparse_indices = {
            field: BM25Index.build(tuple(document.text_for(field) for document in documents))
            for field in config.sparse_document_fields
        }
        sparse_key = CacheKeyBuilder.build(
            source_sha256=document_artifact.sha256,
            operation="bm25_index",
            config={"fields": config.sparse_document_fields, "tokenizer": "m2a-cjk-v1"},
            tool_versions={"cutagent": "m2a-bm25-v1"},
        )

        dense_embeddings: np.ndarray[Any, np.dtype[np.float32]] | None = None
        dense_artifact: ArtifactRef | None = None
        dense_hit = False
        dense_key = CacheKeyBuilder.build(
            source_sha256=document_artifact.sha256,
            operation="dense_text_index",
            config={
                "field": config.dense_document_field,
                "model_id": config.text_encoder.model_id,
                "revision": config.text_encoder.revision,
                "pooling": config.text_encoder.pooling,
                "maximum_tokens": config.text_encoder_max_tokens,
            },
            tool_versions={"transformers": versions["transformers"], "numpy": np.__version__},
        )
        dense_path = self.cache.path("dense_text_index", dense_key, "embeddings.npy")
        if dense_path.is_file():
            dense_hit = True
            with dense_path.open("rb") as handle:
                dense_embeddings = np.load(handle, allow_pickle=False).astype(
                    np.float32, copy=False
                )
        elif text_encoder is not None:
            if (
                text_encoder.model_id != config.text_encoder.model_id
                or text_encoder.revision != config.text_encoder.revision
            ):
                raise ValueError("text encoder identity does not match retrieval config")
            dense_embeddings = text_encoder.encode(
                tuple(document.text_for(config.dense_document_field) for document in documents)
            )
            if dense_embeddings.shape != (len(documents), text_encoder.embedding_dimension):
                raise ValueError("text encoder returned an unexpected matrix shape")
            _write_npy_immutable(dense_path, dense_embeddings)
        if dense_embeddings is not None:
            dense_artifact = ArtifactRef.from_path(
                dense_path,
                artifact_id=f"m2a-dense-{dense_key[:16]}",
                media_type="application/x-npy",
            )

        visual_embeddings: np.ndarray[Any, np.dtype[np.float32]] | None = None
        visual_rows: tuple[VisualVectorRow, ...] = ()
        visual_artifact: ArtifactRef | None = None
        visual_hit = False
        asset_by_id = {asset.artifact.artifact_id: asset for asset in keyframe_assets}
        ordered_visual_inputs: list[tuple[int, EvidenceRef, KeyframeAsset]] = []
        for document_index, document in enumerate(documents):
            for reference in document.keyframe_evidence:
                asset = asset_by_id.get(reference.artifact_id)
                if asset is not None:
                    if not asset.path.is_file():
                        raise FileNotFoundError(asset.path)
                    actual = ArtifactRef.from_path(
                        asset.path,
                        artifact_id=asset.artifact.artifact_id,
                        media_type=asset.artifact.media_type,
                    )
                    if actual.sha256 != asset.artifact.sha256:
                        raise ValueError(f"keyframe hash mismatch: {asset.artifact.artifact_id}")
                    ordered_visual_inputs.append((document_index, reference, asset))
        visual_source_hash = _sha256_json(
            [
                {
                    "document": index,
                    "reference": reference.model_dump(mode="json"),
                    "sha256": asset.artifact.sha256,
                }
                for index, reference, asset in ordered_visual_inputs
            ]
        )
        visual_key = CacheKeyBuilder.build(
            source_sha256=visual_source_hash,
            operation="visual_index",
            config={
                "model_id": config.visual_encoder.model_id,
                "revision": config.visual_encoder.revision,
                "pooling": config.visual_encoder.scene_pooling,
            },
            tool_versions={
                "transformers": versions["transformers"],
                "numpy": np.__version__,
                "input_deduplication": "artifact_id_v1",
            },
        )
        visual_path = self.cache.path("visual_index", visual_key, "embeddings.npy")
        visual_meta = self.cache.read_json("visual_index", visual_key, "rows.json")
        if visual_path.is_file() and visual_meta is not None:
            if not isinstance(visual_meta, list):
                raise CacheError("visual row cache must contain a JSON list")
            visual_hit = True
            with visual_path.open("rb") as handle:
                visual_embeddings = np.load(handle, allow_pickle=False).astype(
                    np.float32, copy=False
                )
            visual_rows = _parse_visual_rows(visual_meta)
        elif visual_encoder is not None and ordered_visual_inputs:
            if (
                visual_encoder.model_id != config.visual_encoder.model_id
                or visual_encoder.revision != config.visual_encoder.revision
            ):
                raise ValueError("visual encoder identity does not match retrieval config")
            unique_assets = tuple(
                dict.fromkeys(asset.artifact.artifact_id for _, _, asset in ordered_visual_inputs)
            )
            unique_embeddings = visual_encoder.encode_images(
                tuple(asset_by_id[artifact_id].path for artifact_id in unique_assets)
            )
            if unique_embeddings.shape[0] != len(unique_assets):
                raise ValueError("visual encoder returned an unexpected row count")
            embedding_by_artifact = {
                artifact_id: unique_embeddings[index]
                for index, artifact_id in enumerate(unique_assets)
            }
            visual_embeddings = np.stack(
                [
                    embedding_by_artifact[asset.artifact.artifact_id]
                    for _, _, asset in ordered_visual_inputs
                ]
            ).astype(np.float32, copy=False)
            visual_rows = tuple(
                VisualVectorRow(document_index=index, evidence_ref=reference)
                for index, reference, _ in ordered_visual_inputs
            )
            _write_npy_immutable(visual_path, visual_embeddings)
            self.cache.write_json(
                "visual_index",
                visual_key,
                "rows.json",
                [
                    {
                        "document_index": row.document_index,
                        "evidence_ref": row.evidence_ref.model_dump(mode="json"),
                    }
                    for row in visual_rows
                ],
            )
        if visual_embeddings is not None:
            visual_artifact = ArtifactRef.from_path(
                visual_path,
                artifact_id=f"m2a-visual-{visual_key[:16]}",
                media_type="application/x-npy",
            )

        component_artifacts = {"documents": document_artifact, "sparse": document_artifact}
        if dense_artifact is not None:
            component_artifacts["dense_text"] = dense_artifact
        if visual_artifact is not None:
            component_artifacts["visual"] = visual_artifact
        config_sha256 = _sha256_json(config.model_dump(mode="json"))
        manifest = RetrievalIndexManifest(
            index_id=f"m2a-index-{_sha256_json((world_hashes, config_sha256))[:16]}",
            config_sha256=config_sha256,
            source_world_state_sha256=world_hashes,
            scene_count=len(documents),
            keyframe_count=len(visual_rows),
            document_template_version=config.document_template_version,
            component_cache_keys={
                "documents": document_key,
                "sparse": sparse_key,
                "dense_text": dense_key,
                "visual": visual_key,
            },
            component_artifacts=component_artifacts,
            encoder_models={
                "dense_text": f"{config.text_encoder.model_id}@{config.text_encoder.revision}",
                "visual": f"{config.visual_encoder.model_id}@{config.visual_encoder.revision}",
            },
            library_versions=versions,
            build_time_ms=round((time.perf_counter() - started) * 1000),
            metadata={
                "document_cache_hit": document_hit,
                "dense_cache_hit": dense_hit,
                "visual_cache_hit": visual_hit,
            },
        )
        return RetrievalMemory(
            manifest=manifest,
            documents=documents,
            sparse_indices=sparse_indices,
            dense_document_field=config.dense_document_field,
            dense_embeddings=dense_embeddings,
            visual_embeddings=visual_embeddings,
            visual_rows=visual_rows,
            cache_hits={
                "documents": document_hit,
                "dense_text": dense_hit,
                "visual": visual_hit,
            },
        )
