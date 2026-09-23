"""Qwen3-VL M6 PEFT adapter backend without changing the frozen M4B baseline."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from cutagent.agent.m4b_protocols import M4BPolicyModelRequest
from cutagent.models.policy_prompt_m6 import render_m6_prompt
from cutagent.models.qwen_policy_m4b import (
    MODEL_ID,
    MODEL_REVISION,
    Qwen3VLPolicyBackendM4B,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Qwen3VLSFTPolicyBackend(Qwen3VLPolicyBackendM4B):
    """Load one immutable PEFT adapter and use it only for trained decisions."""

    def __init__(self, *, model_cache: Any, adapter_checkpoint: Path) -> None:
        super().__init__(model_cache=model_cache)
        self._adapter_checkpoint = adapter_checkpoint.resolve(strict=True)
        weights = self._adapter_checkpoint / "adapter_model.safetensors"
        if not weights.is_file():
            raise ValueError("M6 adapter checkpoint omitted adapter_model.safetensors")
        self.adapter_sha256 = _sha256(weights)

    @property
    def backend_version(self) -> str:
        return (
            "cutagent-qwen3-vl-policy-m6-sft-v1+"
            f"adapter-{self.adapter_sha256[:16]}+"
            f"torch-{importlib.metadata.version('torch')}+"
            f"transformers-{importlib.metadata.version('transformers')}+"
            f"peft-{importlib.metadata.version('peft')}"
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        import time

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
        self._model = peft.PeftModel.from_pretrained(
            base,
            self._adapter_checkpoint,
            is_trainable=False,
        )
        self._model.eval()
        self._processor = transformers.AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=self._model_cache,
            local_files_only=True,
        )
        self._torch = torch
        self._process_vision_info = process_vision_info
        self.load_time_ms = (time.perf_counter_ns() - started) // 1_000_000

    def _prompt(
        self,
        request: M4BPolicyModelRequest,
        *,
        previous: str | None = None,
        error: str | None = None,
    ) -> str:
        if request.operation == "plan":
            return super()._prompt(request, previous=previous, error=error)
        prompt = render_m6_prompt(
            operation=request.operation,
            serialized_context=request.serialized_context,
            tool_manifest=request.tool_manifest,
        )
        if previous is not None and error is not None:
            prompt += (
                "\nThe previous output was invalid. Return one corrected JSON object only.\n"
                f"Validation error: {error}"
            )
        return prompt

    def _generate(self, request: M4BPolicyModelRequest, prompt: str) -> tuple[str, int, int]:
        assert self._model is not None
        context = self._model.disable_adapter() if request.operation == "plan" else nullcontext()
        with context:
            return super()._generate(request, prompt)
