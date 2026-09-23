"""Small evidence-linked world states and deterministic fake M2A encoders."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from cutagent.retrieval.store import RetrievalMemory
from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import (
    EvidenceDescriptor,
    EvidenceRef,
    OCRSpan,
    ScenePerception,
    TranscriptSpan,
    VideoWorldState,
    VisualAction,
    VisualEntity,
)
from cutagent.schemas.retrieval import TextEncoderSpec, VisualEncoderSpec


def sample_world_states() -> tuple[VideoWorldState, ...]:
    audio = EvidenceRef(
        artifact_id="audio-red",
        evidence_kind="source_audio_pcm",
        time_range=TimeRange(start_ms=0, end_ms=1000),
    )
    frame_red = EvidenceRef(
        artifact_id="frame-red",
        evidence_kind="keyframe",
        segment_id="scene-red",
        observed_ms=500,
    )
    frame_blue = EvidenceRef(
        artifact_id="frame-blue",
        evidence_kind="keyframe",
        segment_id="scene-blue",
        observed_ms=1500,
    )
    vlm_red = EvidenceRef(
        artifact_id="vlm-red",
        evidence_kind="vlm_raw_output",
        segment_id="scene-red",
        time_range=TimeRange(start_ms=0, end_ms=1000),
    )
    vlm_blue = EvidenceRef(
        artifact_id="vlm-blue",
        evidence_kind="vlm_raw_output",
        segment_id="scene-blue",
        time_range=TimeRange(start_ms=1000, end_ms=2000),
    )
    catalog = (
        EvidenceDescriptor(**audio.model_dump(), media_type="audio/wav"),
        EvidenceDescriptor(**frame_red.model_dump(), media_type="image/jpeg"),
        EvidenceDescriptor(**frame_blue.model_dump(), media_type="image/jpeg"),
        EvidenceDescriptor(**vlm_red.model_dump(), media_type="application/json"),
        EvidenceDescriptor(**vlm_blue.model_dump(), media_type="application/json"),
    )
    red_scene = ScenePerception(
        segment_id="scene-red",
        time_range=TimeRange(start_ms=0, end_ms=1000),
        keyframe_evidence=(frame_red,),
        transcript_spans=(
            TranscriptSpan(
                span_id="speech-red",
                text="A red square moves right",
                time_range=TimeRange(start_ms=0, end_ms=900),
                language="en",
                evidence_refs=(audio,),
            ),
        ),
        ocr_spans=(
            OCRSpan(
                span_id="ocr-red",
                exact_text="SALE 42",
                segment_id="scene-red",
                observed_ms=500,
                evidence_refs=(frame_red,),
            ),
        ),
        entities=(VisualEntity(label="red square", evidence_refs=(vlm_red,)),),
        actions=(
            VisualAction(subject="red square", action="moves right", evidence_refs=(vlm_red,)),
        ),
        temporal_events=(),
        scene_summary="A red square crosses the frame.",
        summary_evidence_refs=(vlm_red,),
    )
    blue_scene = ScenePerception(
        segment_id="scene-blue",
        time_range=TimeRange(start_ms=1000, end_ms=2000),
        keyframe_evidence=(frame_blue,),
        transcript_spans=(),
        ocr_spans=(
            OCRSpan(
                span_id="ocr-blue",
                exact_text="出口",
                segment_id="scene-blue",
                observed_ms=1500,
                evidence_refs=(frame_blue,),
            ),
        ),
        entities=(VisualEntity(label="blue circle", evidence_refs=(vlm_blue,)),),
        actions=(
            VisualAction(subject="blue circle", action="stationary", evidence_refs=(vlm_blue,)),
        ),
        temporal_events=(),
        scene_summary="A blue circle remains still near the exit sign.",
        summary_evidence_refs=(vlm_blue,),
    )
    return (
        VideoWorldState(
            video_id="video-sample",
            duration_ms=2000,
            evidence_catalog=catalog,
            scenes=(red_scene, blue_scene),
            global_summary="Two geometric scenes.",
            global_summary_evidence_refs=(vlm_red, vlm_blue),
        ),
    )


class FakeTextEncoder:
    def __init__(self) -> None:
        self.calls = 0
        self.spec = TextEncoderSpec()

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def revision(self) -> str:
        return self.spec.revision

    @property
    def embedding_dimension(self) -> int:
        return self.spec.embedding_dimension

    def encode(self, texts: tuple[str, ...]) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        self.calls += 1
        matrix = np.zeros((len(texts), self.embedding_dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.casefold().split():
                column = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 1024
                matrix[row, column] += 1
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


class FakeVisualEncoder:
    def __init__(self) -> None:
        self.image_calls = 0
        self.query_calls = 0
        self.spec = VisualEncoderSpec()

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def revision(self) -> str:
        return self.spec.revision

    @staticmethod
    def _vector(value: str) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        lowered = value.casefold()
        vector = np.array(
            [
                float("red" in lowered or "红" in lowered),
                float("blue" in lowered or "蓝" in lowered),
                float("square" in lowered or "方" in lowered),
                float("circle" in lowered or "圆" in lowered),
            ],
            dtype=np.float32,
        )
        norm = np.linalg.norm(vector)
        return vector / norm if norm else np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    def encode_images(
        self, paths: tuple[Path, ...]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        self.image_calls += 1
        return np.stack([self._vector(path.stem) for path in paths])

    def encode_queries(
        self, texts: tuple[str, ...]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        self.query_calls += 1
        return np.stack([self._vector(text) for text in texts])


def build_fake_memory(
    root: Path,
) -> tuple[RetrievalMemory, FakeTextEncoder, FakeVisualEncoder]:
    """Build the real local index with tiny deterministic encoders."""

    from cutagent.core.artifacts import ArtifactRef
    from cutagent.retrieval.index import KeyframeAsset, RetrievalIndexBuilder
    from cutagent.schemas.retrieval import RetrievalConfig

    assets = []
    for artifact_id in ("frame-red", "frame-blue"):
        path = root / f"{artifact_id}.jpg"
        path.write_text(artifact_id, encoding="utf-8")
        assets.append(
            KeyframeAsset(
                ArtifactRef.from_path(path, artifact_id=artifact_id, media_type="image/jpeg"),
                path,
            )
        )
    text = FakeTextEncoder()
    visual = FakeVisualEncoder()
    memory = RetrievalIndexBuilder(root / "index").build(
        world_states=sample_world_states(),
        keyframe_assets=tuple(assets),
        config=RetrievalConfig(),
        text_encoder=text,
        visual_encoder=visual,
    )
    return memory, text, visual
