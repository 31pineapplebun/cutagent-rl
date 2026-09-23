"""Qwen3-VL policy backend composing frozen M6 SFT and M7 DPO adapters."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import time
from pathlib import Path
from typing import Any

from cutagent.models.qwen_policy_m4b import MODEL_ID, MODEL_REVISION
from cutagent.models.qwen_policy_m6 import Qwen3VLSFTPolicyBackend


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Qwen3VLDPOPolicyBackend(Qwen3VLSFTPolicyBackend):
    """Activate the additive SFT+DPO LoRA stack for trained decision operations."""

    def __init__(
        self,
        *,
        model_cache: Any,
        sft_adapter_checkpoint: Path,
        dpo_adapter_checkpoint: Path,
    ) -> None:
        super().__init__(
            model_cache=model_cache,
            adapter_checkpoint=sft_adapter_checkpoint,
        )
        self._dpo_adapter_checkpoint = dpo_adapter_checkpoint.resolve(strict=True)
        weights = self._dpo_adapter_checkpoint / "adapter_model.safetensors"
        if not weights.is_file():
            raise ValueError("M7 adapter checkpoint omitted adapter_model.safetensors")
        self.dpo_adapter_sha256 = _sha256(weights)

    @property
    def backend_version(self) -> str:
        return (
            "cutagent-qwen3-vl-policy-m7-dpo-v1+"
            f"sft-{self.adapter_sha256[:16]}+"
            f"dpo-{self.dpo_adapter_sha256[:16]}+"
            f"torch-{importlib.metadata.version('torch')}+"
            f"transformers-{importlib.metadata.version('transformers')}+"
            f"peft-{importlib.metadata.version('peft')}"
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        started = time.perf_counter_ns()
        torch: Any = importlib.import_module("torch")
        transformers: Any = importlib.import_module("transformers")
        peft: Any = importlib.import_module("peft")
        process_vision_info: Any = importlib.import_module("qwen_vl_utils").process_vision_info
        base = transformers.AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=self._model_cache,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True,
        )
        model = peft.PeftMixedModel.from_pretrained(
            base,
            self._adapter_checkpoint,
            adapter_name="sft",
            is_trainable=False,
        )
        model.load_adapter(
            self._dpo_adapter_checkpoint,
            adapter_name="dpo",
            is_trainable=False,
        )
        model.set_adapter(["sft", "dpo"])
        model.eval()
        self._model = model
        self._processor = transformers.AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=self._model_cache,
            local_files_only=True,
        )
        self._torch = torch
        self._process_vision_info = process_vision_info
        self.load_time_ms = (time.perf_counter_ns() - started) // 1_000_000
