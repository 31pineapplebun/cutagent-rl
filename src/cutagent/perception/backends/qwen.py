"""Local Transformers Qwen3-VL structured visual-perception backend."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import time
from pathlib import Path
from typing import Any

from cutagent.core.errors import StructuredOutputError
from cutagent.perception.artifacts import write_json_artifact
from cutagent.perception.protocols import VLMBackendResult, VLMRequest
from cutagent.perception.structured_output import (
    parse_with_bounded_repair,
    to_visual_observation,
)
from cutagent.schemas.perception import ModelPerformance

PROMPT_TEMPLATE = """You are a conservative video perception system. Return exactly one JSON object,
without markdown, using this schema:
{
  "scene_summary": {"text": string, "evidence_ids": [string]} | null,
  "entities": [{"label": string, "attributes": {string:string}, "evidence_ids": [string]}],
  "actions": [{"subject": string, "action": string, "object": string|null,
    "evidence_ids": [string]}],
  "directly_visible_text": [{"exact_text": string, "normalized_text": string|null,
    "evidence_ids": [string]}],
  "inferred_semantic_text": [string],
  "temporal_events": [{"description": string, "start_ms": integer, "end_ms": integer,
    "uncertainty": string|null, "evidence_ids": [string]}],
  "uncertainties": [string]
}
Every claim must cite one or more supplied evidence IDs. Use only those exact IDs. Scene times are
half-open [start_ms,end_ms). Do not infer unreadable text as directly_visible_text; place semantic
inferences in inferred_semantic_text. Use empty arrays or null when unsupported. Do not invent
confidence values. Report concrete visible objects in entities even when static. Do not create a
temporal event merely to restate a frame timestamp; events require visible change or action.
"""

EXPLICIT_TEMPORAL_ADDENDUM = """
Compare the supplied visual evidence in chronological order before producing JSON. Distinguish
object motion from camera motion. Explicitly check whether each important entity is stationary,
moves left/right/up/down, appears, disappears, enters or exits the frame, approaches, or moves
away. For multiple moving objects, compare which motion starts first. Cite evidence from at least
two different times for a motion claim when such evidence is available. Use conservative event
ranges inside the scene; leave actions/events empty when change is not visually supported.
"""


class Qwen3VLVLMBackend:
    """Pinned BF16 eager backend with strict validation and bounded repair."""

    def __init__(self, *, model_cache: str) -> None:
        self._model_cache = model_cache
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._process_vision_info: Any | None = None

    @property
    def backend_version(self) -> str:
        return (
            "cutagent-qwen3-vl-eager-v1+"
            f"torch-{importlib.metadata.version('torch')}+"
            f"transformers-{importlib.metadata.version('transformers')}+"
            f"qwen-vl-utils-{importlib.metadata.version('qwen-vl-utils')}+"
            f"torchcodec-{importlib.metadata.version('torchcodec')}"
        )

    def _load(self, request: VLMRequest) -> None:
        if self._model is not None:
            return
        torch: Any = importlib.import_module("torch")
        transformers: Any = importlib.import_module("transformers")
        process_vision_info: Any = importlib.import_module("qwen_vl_utils").process_vision_info
        spec = request.config.qwen
        self._model = transformers.AutoModelForImageTextToText.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            cache_dir=self._model_cache,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self._processor = transformers.AutoProcessor.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            cache_dir=self._model_cache,
        )
        self._torch = torch
        self._process_vision_info = process_vision_info

    @staticmethod
    def _file_uri(path: Path) -> str:
        return path.resolve(strict=True).as_uri()

    def _messages(self, request: VLMRequest, prompt: str) -> list[dict[str, Any]]:
        evidence_lines = []
        last_index = len(request.visual_inputs) - 1
        for index, item in enumerate(request.visual_inputs):
            if item.evidence_ref.observed_ms is not None:
                timing = f"observed at {item.evidence_ref.observed_ms} ms"
            elif item.evidence_ref.time_range is not None:
                timing = (
                    f"covers [{item.evidence_ref.time_range.start_ms},"
                    f"{item.evidence_ref.time_range.end_ms}) ms"
                )
            else:
                timing = "no finer timestamp"
            if request.config.label_temporal_phases and request.config.visual_mode == "keyframes":
                if last_index <= 0 or index * 2 == last_index:
                    phase = "middle"
                elif index == 0:
                    phase = "before"
                elif index == last_index:
                    phase = "after"
                elif index * 2 < last_index:
                    phase = "between before and middle"
                else:
                    phase = "between middle and after"
                timing += f"; chronological phase={phase}"
            evidence_lines.append(f"- {item.evidence_alias}: {timing}")
        if request.config.temporal_prompt_style == "explicit_comparison":
            prompt += EXPLICIT_TEMPORAL_ADDENDUM
        temporal = (
            f"Scene {request.scene.segment_id} covers "
            f"[{request.scene.time_range.start_ms},{request.scene.time_range.end_ms}) ms.\n"
            "Available evidence:\n" + "\n".join(evidence_lines) + "\n" + prompt
        )
        aliases = [item.evidence_alias for item in request.visual_inputs]
        temporal += (
            "\nThe only permitted evidence_ids are: "
            + ", ".join(aliases)
            + ". Every claim object, including every directly_visible_text item, MUST contain "
            'an "evidence_ids" array. '
        )
        if len(aliases) == 1:
            temporal += f'For this input that array must be ["{aliases[0]}"].'
        content: list[dict[str, Any]] = []
        if request.config.visual_mode == "keyframes":
            content.extend(
                {
                    "type": "image",
                    "image": self._file_uri(item.path),
                    "max_pixels": request.config.maximum_image_pixels,
                }
                for item in request.visual_inputs
            )
        else:
            if len(request.visual_inputs) != 1:
                raise ValueError("native-video mode requires exactly one scene clip")
            content.append(
                {
                    "type": "video",
                    "video": self._file_uri(request.visual_inputs[0].path),
                    "fps": request.config.native_video_fps,
                    "max_pixels": request.config.maximum_video_frame_pixels,
                    "total_pixels": request.config.maximum_video_total_pixels,
                }
            )
        content.append({"type": "text", "text": temporal})
        return [{"role": "user", "content": content}]

    def _generate(self, request: VLMRequest, prompt: str) -> tuple[str, int, int]:
        assert self._model is not None
        assert self._processor is not None
        assert self._process_vision_info is not None
        assert self._torch is not None
        torch = self._torch
        messages = self._messages(request, prompt)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = self._process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_metadata = None
        if videos is not None:
            video_tensors, metadata = zip(*videos, strict=True)
            videos = list(video_tensors)
            video_metadata = list(metadata)
        visual_frames = (
            sum(int(video.shape[0]) for video in videos)
            if videos is not None
            else (0 if images is None else len(images))
        )
        inputs = self._processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        ).to(self._model.device)
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=request.config.maximum_new_tokens,
                do_sample=False,
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated, strict=True)
        ]
        output = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return output, int(inputs["input_ids"].shape[-1]), visual_frames

    def perceive(self, request: VLMRequest) -> VLMBackendResult:
        self._load(request)
        assert self._torch is not None
        torch = self._torch
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        initial, input_tokens, visual_frames = self._generate(request, PROMPT_TEMPLATE)

        def repair(previous: str, validation_error: str) -> str:
            repair_prompt = (
                PROMPT_TEMPLATE
                + "\nYour prior output was invalid. Return a corrected complete JSON object.\n"
                + f"Validation error: {validation_error}\nPrior output: {previous}"
            )
            repaired, _, _ = self._generate(request, repair_prompt)
            return repaired

        try:
            parsed, attempts = parse_with_bounded_repair(
                initial,
                maximum_repairs=request.config.maximum_repair_attempts,
                repair=repair,
            )
        except StructuredOutputError as error:
            torch.cuda.synchronize()
            failure_latency_ms = (time.perf_counter_ns() - started) // 1_000_000
            failure_artifact = write_json_artifact(
                request.output_directory / "raw_qwen_failure.json",
                {
                    "prompt_template_version": request.config.prompt_template_version,
                    "attempts": list(error.attempts),
                    "error": str(error),
                    "latency_ms": failure_latency_ms,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "input_tokens": input_tokens,
                    "frames": visual_frames,
                },
                artifact_prefix="vlm-raw-failure",
            )
            raise StructuredOutputError(
                f"{error}; raw_failure_artifact={failure_artifact.artifact_id}",
                attempts=error.attempts,
            ) from error
        evidence = {item.evidence_alias: item.evidence_ref for item in request.visual_inputs}
        digest = hashlib.sha256(
            (
                f"{request.video.source.sha256}:{request.scene.segment_id}:"
                f"{request.config.visual_mode}:{request.config.prompt_template_version}"
            ).encode()
        ).hexdigest()[:20]
        observation = to_visual_observation(
            parsed,
            observation_id=f"visual-{digest}",
            segment_id=request.scene.segment_id,
            mode=request.config.visual_mode,
            scene_range=request.scene.time_range,
            available_evidence=evidence,
            repair_count=len(attempts) - 1,
        )
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter_ns() - started) // 1_000_000
        raw_artifact = write_json_artifact(
            request.output_directory / "raw_qwen_output.json",
            {
                "prompt_template_version": request.config.prompt_template_version,
                "attempts": list(attempts),
                "validated": parsed.model_dump(mode="json"),
                "observation": observation.model_dump(mode="json"),
            },
            artifact_prefix="vlm-raw",
        )
        spec = request.config.qwen
        return VLMBackendResult(
            observation=observation,
            raw_output=raw_artifact,
            performance=ModelPerformance(
                operation=f"qwen_{request.config.visual_mode}",
                model_id=spec.model_id,
                model_revision=spec.revision,
                dtype=spec.dtype,
                device="cuda:0",
                latency_ms=latency_ms,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                frames=visual_frames,
                input_tokens=input_tokens,
                repair_count=len(attempts) - 1,
            ),
        )
