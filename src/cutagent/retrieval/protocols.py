"""Replaceable retrieval and encoder protocols."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from cutagent.schemas.retrieval import (
    NativeVideoVerification,
    RetrievalCandidate,
    RetrievalQuery,
    RetrievalResponse,
)

if TYPE_CHECKING:
    from cutagent.retrieval.native_video import SceneClipAsset
    from cutagent.retrieval.store import RetrievalMemory

FloatMatrix = npt.NDArray[np.float32]


class TextEncoder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def embedding_dimension(self) -> int: ...

    def encode(self, texts: tuple[str, ...]) -> FloatMatrix: ...


class VisualEncoder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def encode_images(self, paths: tuple[Path, ...]) -> FloatMatrix: ...

    def encode_queries(self, texts: tuple[str, ...]) -> FloatMatrix: ...


class MultimodalRetriever(Protocol):
    def search(self, query: RetrievalQuery, memory_ref: RetrievalMemory) -> RetrievalResponse: ...


class MotionEvidenceVerifier(Protocol):
    def verify(
        self,
        *,
        query: RetrievalQuery,
        candidate: RetrievalCandidate,
        scene_clip: SceneClipAsset,
    ) -> NativeVideoVerification: ...
