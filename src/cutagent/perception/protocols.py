"""Replaceable M1B backend protocols and their internal result records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.media import SceneSegment, VideoAsset
from cutagent.schemas.perception import (
    EvidenceRef,
    ModelPerformance,
    OCRSpan,
    PerceptionConfig,
    TranscriptSpan,
    VisualObservation,
)


@dataclass(frozen=True, slots=True)
class ASRRequest:
    audio_path: Path
    audio_artifact: ArtifactRef
    video: VideoAsset
    timeline_offset_ms: int
    output_directory: Path
    config: PerceptionConfig


@dataclass(frozen=True, slots=True)
class ASRBackendResult:
    spans: tuple[TranscriptSpan, ...]
    raw_output: ArtifactRef
    performance: ModelPerformance


@dataclass(frozen=True, slots=True)
class VisualInput:
    evidence_alias: str
    path: Path
    evidence_ref: EvidenceRef


@dataclass(frozen=True, slots=True)
class VLMRequest:
    video: VideoAsset
    scene: SceneSegment
    visual_inputs: tuple[VisualInput, ...]
    output_directory: Path
    config: PerceptionConfig


@dataclass(frozen=True, slots=True)
class VLMBackendResult:
    observation: VisualObservation
    raw_output: ArtifactRef
    performance: ModelPerformance


class ASRBackend(Protocol):
    @property
    def backend_version(self) -> str: ...

    def transcribe(self, request: ASRRequest) -> ASRBackendResult: ...


class VLMBackend(Protocol):
    @property
    def backend_version(self) -> str: ...

    def perceive(self, request: VLMRequest) -> VLMBackendResult: ...


class OCRBackend(Protocol):
    @property
    def backend_version(self) -> str: ...

    def extract(self, observations: tuple[VisualObservation, ...]) -> tuple[OCRSpan, ...]: ...
