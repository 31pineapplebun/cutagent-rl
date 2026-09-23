"""Pinned local Transformers encoder implementations for the M2A baselines."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from cutagent.retrieval.protocols import FloatMatrix
from cutagent.schemas.retrieval import TextEncoderSpec, VisualEncoderSpec


def _torch_and_transformers() -> tuple[Any, Any, Any, Any]:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as error:  # pragma: no cover - optional real-model runtime
        raise RuntimeError("install CutAgent with the retrieval extra") from error
    return (
        torch,
        (transformers.AutoModel, transformers.AutoTokenizer),
        transformers.AutoModel,
        transformers.AutoProcessor,
    )


def pooled_feature_tensor(value: Any) -> Any:
    """Normalize the Transformers 5 pooled-output API to its feature tensor."""

    pooled = getattr(value, "pooler_output", None)
    return pooled if pooled is not None else value


class BGETextEncoder:
    """BGE-M3 dense CLS encoder, pinned and loaded locally in BF16."""

    def __init__(
        self,
        *,
        model_cache: Path,
        batch_size: int = 16,
        maximum_tokens: int = 512,
        device: str = "cuda",
    ) -> None:
        torch, auto, _, _ = _torch_and_transformers()
        auto_model, auto_tokenizer = auto
        self.spec = TextEncoderSpec()
        self.batch_size = batch_size
        self.maximum_tokens = maximum_tokens
        self.device = device
        self._torch = torch
        self._tokenizer = auto_tokenizer.from_pretrained(
            self.spec.model_id,
            revision=self.spec.revision,
            cache_dir=model_cache,
            local_files_only=True,
        )
        self._model = auto_model.from_pretrained(
            self.spec.model_id,
            revision=self.spec.revision,
            cache_dir=model_cache,
            local_files_only=True,
            dtype=torch.bfloat16,
        ).to(device)
        self._model.eval()

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def revision(self) -> str:
        return self.spec.revision

    @property
    def embedding_dimension(self) -> int:
        return self.spec.embedding_dimension

    def encode(self, texts: tuple[str, ...]) -> FloatMatrix:
        if not texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)
        output: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                inputs = self._tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=self.maximum_tokens,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                hidden = self._model(**inputs).last_hidden_state[:, 0]
                normalized = self._torch.nn.functional.normalize(hidden.float(), dim=-1)
                output.append(normalized.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(output, axis=0)


class Siglip2VisualEncoder:
    """Shared SigLIP2 image/text space with max-per-keyframe scene scoring."""

    def __init__(
        self,
        *,
        model_cache: Path,
        batch_size: int = 16,
        device: str = "cuda",
    ) -> None:
        torch, _, auto_model, auto_processor = _torch_and_transformers()
        self.spec = VisualEncoderSpec()
        self.batch_size = batch_size
        self.device = device
        self._torch = torch
        self._processor = auto_processor.from_pretrained(
            self.spec.model_id,
            revision=self.spec.revision,
            cache_dir=model_cache,
            local_files_only=True,
        )
        # The pinned repository currently declares ``model_type=siglip``;
        # AutoModel is therefore the only shape-safe official loader.
        self._model = auto_model.from_pretrained(
            self.spec.model_id,
            revision=self.spec.revision,
            cache_dir=model_cache,
            local_files_only=True,
            dtype=torch.bfloat16,
        ).to(device)
        self._model.eval()

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def revision(self) -> str:
        return self.spec.revision

    def encode_images(self, paths: tuple[Path, ...]) -> FloatMatrix:
        if not paths:
            return np.empty((0, 768), dtype=np.float32)
        try:
            image_module = importlib.import_module("PIL.Image")
        except ImportError as error:  # pragma: no cover - optional real-model runtime
            raise RuntimeError("install CutAgent with the retrieval extra") from error

        output: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        with self._torch.inference_mode():
            for start in range(0, len(paths), self.batch_size):
                images = []
                for path in paths[start : start + self.batch_size]:
                    with image_module.open(path) as image:
                        images.append(image.convert("RGB"))
                inputs = self._processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device, dtype=self._torch.bfloat16)
                features = pooled_feature_tensor(
                    self._model.get_image_features(pixel_values=pixel_values)
                )
                normalized = self._torch.nn.functional.normalize(features.float(), dim=-1)
                output.append(normalized.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(output, axis=0)

    def encode_queries(self, texts: tuple[str, ...]) -> FloatMatrix:
        if not texts:
            return np.empty((0, 768), dtype=np.float32)
        output: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                inputs = self._processor(
                    text=list(texts[start : start + self.batch_size]),
                    padding="max_length",
                    max_length=64,
                    return_tensors="pt",
                )
                text_inputs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                    if key in {"input_ids", "attention_mask"}
                }
                features = pooled_feature_tensor(self._model.get_text_features(**text_inputs))
                normalized = self._torch.nn.functional.normalize(features.float(), dim=-1)
                output.append(normalized.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(output, axis=0)
