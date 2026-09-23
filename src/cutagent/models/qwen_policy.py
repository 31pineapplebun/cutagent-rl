"""Pinned local Qwen3-VL BF16 adapter for M4A planning and typed decisions."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from cutagent.agent.protocols import PolicyModelRequest, PolicyModelResult
from cutagent.core.errors import StructuredOutputError
from cutagent.perception.artifacts import write_json_artifact
from cutagent.perception.structured_output import parse_json_object
from cutagent.schemas.agent import (
    POLICY_DECISION_ADAPTER,
    PlanGraph,
    PolicyInferenceStats,
    ReplanDecision,
)

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

_DECISION_CONTRACT = """Return exactly one balanced JSON object without markdown.
Choose one decision_type. These are the exact nesting contracts:
1) tool example:
{"decision_type":"tool","tool_call":{"tool_name":"inspect_media",
"tool_call_id":"call-001","arguments":{"input_artifact_id":"opaque-001"}},
"rationale":"why","expected_observation":"what","success_condition":"condition"}
The rationale, expected_observation, and success_condition fields are TOP-LEVEL siblings of
tool_call, never fields inside tool_call. Close both arguments and tool_call before rationale.
2) replan:
{"decision_type":"replan","reason":"why","affected_plan_nodes":["node-1"],
"requested_patch":{"event_type":"plan_patch","revision":2,"reason":"why",
"steps":[complete PlanNode objects]}}
3) finish:
{"decision_type":"finish","output_artifact_id":"opaque-output",
"completion_summary":"what completed","evidence_ids":["opaque-evidence"]}
4) cannot_complete:
{"decision_type":"cannot_complete","reason":"why",
"missing_evidence_or_capability":["missing item"],"attempted_action_ids":["call-001"]}
Never emit shell commands, paths, hidden metadata, or unsupported tool names. Use artifact IDs and
integer millisecond half-open ranges exactly as shown in the context/tool schema. Never repeat an
identical successful tool call. After an editing tool succeeds, operate on its newest output
artifact. When independent validation is required and an edited output exists, call validate_media
on that output; after successful validation, finish with that same output artifact. Do not claim
finish unless a real output artifact exists and has been validated when validation is possible.
"""

_PLAN_CONTRACT = """Return exactly one JSON object without markdown using:
{"revision":1,"nodes":[{"node_id":opaque-id,"subgoal":string,
"dependencies":[node-id],"status":"pending"|"ready","expected_evidence":[string],
"preferred_capability":opaque-id|null,"completion_criteria":[string]}]}.
Use an acyclic graph with 2-8 general task-derived nodes. Do not include tool arguments or private
answers in the plan. Initial dependency-free nodes should be ready; other nodes pending.
"""


class Qwen3VLPolicyBackend:
    """Transformers eager reference backend with strict JSON and bounded repair."""

    def __init__(self, *, model_cache: Path) -> None:
        self._model_cache = model_cache
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._process_vision_info: Any | None = None
        self.load_time_ms = 0

    @property
    def backend_version(self) -> str:
        return (
            "cutagent-qwen3-vl-policy-eager-v1+"
            f"torch-{importlib.metadata.version('torch')}+"
            f"transformers-{importlib.metadata.version('transformers')}+"
            f"qwen-vl-utils-{importlib.metadata.version('qwen-vl-utils')}"
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        started = time.perf_counter_ns()
        torch: Any = importlib.import_module("torch")
        transformers: Any = importlib.import_module("transformers")
        process_vision_info: Any = importlib.import_module("qwen_vl_utils").process_vision_info
        self._model = transformers.AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=self._model_cache,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True,
        )
        self._processor = transformers.AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=self._model_cache,
            local_files_only=True,
        )
        self._torch = torch
        self._process_vision_info = process_vision_info
        self.load_time_ms = (time.perf_counter_ns() - started) // 1_000_000

    @staticmethod
    def _tool_contract(request: PolicyModelRequest) -> str:
        tools = [
            {
                "name": item.name,
                "version": item.version,
                "description": item.description,
                "arguments": item.argument_schema,
            }
            for item in request.tool_manifest.tools
        ]
        return json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _prompt(
        self,
        request: PolicyModelRequest,
        *,
        previous: str | None = None,
        error: str | None = None,
    ) -> str:
        role = {
            "plan": "global planner",
            "decide": "local video-tool policy",
            "replan": "targeted replanner",
        }[request.operation]
        contract = _PLAN_CONTRACT if request.operation == "plan" else _DECISION_CONTRACT
        operation_rule = ""
        if request.baseline == "react":
            operation_rule += (
                "This is the ReAct baseline with no PlanGraph. Never emit decision_type=replan; "
                "recover by selecting another typed tool or cannot_complete.\n"
            )
        else:
            operation_rule += (
                "This is the hierarchical baseline. Follow the current PlanGraph and use a "
                "targeted replan only when observable failure makes it necessary.\n"
            )
        if request.operation == "replan":
            operation_rule += (
                "This is a targeted replan call. Return decision_type=replan only. Preserve every "
                "succeeded and unaffected node exactly; patch only failed/blocked affected nodes.\n"
            )
        repair = ""
        if previous is not None and error is not None:
            repair = (
                "\nYour previous output was invalid. Do not repeat it byte-for-byte. "
                "Correct it completely, balance every JSON brace, and obey field nesting.\n"
                f"Validation error: {error}\nPrevious output: {previous}\n"
            )
        return (
            f"You are the CutAgent-RL {role}. Be conservative and evidence-grounded.\n"
            f"Prompt version: {request.prompt_template_version}\n"
            f"Deterministic seed: {request.seed}\n"
            f"{operation_rule}{contract}\n"
            f"Registered tools: {self._tool_contract(request)}\n"
            f"Whitelisted policy context: {request.serialized_context}\n"
            f"{repair}"
        )

    def _messages(self, request: PolicyModelRequest, prompt: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for item in request.visual_inputs:
            content.append(
                {
                    "type": "image",
                    "image": item.path.resolve(strict=True).as_uri(),
                    "max_pixels": 384 * 384,
                }
            )
        if request.visual_inputs:
            aliases = ", ".join(item.artifact_id for item in request.visual_inputs)
            prompt += f"\nImages are ordered and correspond to opaque evidence IDs: {aliases}."
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _generate(self, request: PolicyModelRequest, prompt: str) -> tuple[str, int, int]:
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None
        assert self._process_vision_info is not None
        messages = self._messages(request, prompt)
        self._torch.manual_seed(request.seed)
        self._torch.cuda.manual_seed_all(request.seed)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = self._process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        inputs = self._processor(
            text=text,
            images=images,
            videos=videos,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        ).to(self._model.device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=request.maximum_new_tokens,
                do_sample=False,
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated, strict=True)
        ]
        raw = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return raw, int(inputs["input_ids"].shape[-1]), int(trimmed[0].shape[-1])

    @staticmethod
    def _parse(request: PolicyModelRequest, raw: str) -> tuple[PlanGraph | None, Any | None]:
        payload = parse_json_object(raw)
        if request.operation == "plan":
            return PlanGraph.model_validate(payload), None
        decision = POLICY_DECISION_ADAPTER.validate_python(payload)
        if request.operation == "replan" and not isinstance(decision, ReplanDecision):
            raise ValueError("replan operation requires a ReplanDecision")
        return None, decision

    def infer(self, request: PolicyModelRequest) -> PolicyModelResult:
        self._load()
        assert self._torch is not None
        torch = self._torch
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        attempts: list[str] = []
        plan: PlanGraph | None = None
        decision: Any | None = None
        input_tokens = output_tokens = 0
        previous: str | None = None
        validation_error: str | None = None
        for attempt_index in range(request.maximum_repairs + 1):
            prompt = self._prompt(request, previous=previous, error=validation_error)
            raw, current_input, current_output = self._generate(request, prompt)
            attempts.append(raw)
            input_tokens += current_input
            output_tokens += current_output
            try:
                plan, decision = self._parse(request, raw)
                break
            except (StructuredOutputError, ValidationError, ValueError) as error:
                if attempt_index >= request.maximum_repairs:
                    failure = write_json_artifact(
                        request.output_directory / "raw_policy_failure.json",
                        {
                            "operation": request.operation,
                            "prompt_template_version": request.prompt_template_version,
                            "seed": request.seed,
                            "attempts": attempts,
                            "error": str(error),
                        },
                        artifact_prefix="policy-raw-failure",
                    )
                    raise StructuredOutputError(
                        f"policy output invalid; artifact={failure.artifact_id}",
                        attempts=tuple(attempts),
                    ) from error
                previous = raw
                validation_error = str(error)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter_ns() - started) // 1_000_000
        raw_artifact = write_json_artifact(
            request.output_directory / "raw_policy_output.json",
            {
                "operation": request.operation,
                "prompt_template_version": request.prompt_template_version,
                "seed": request.seed,
                "attempts": attempts,
                "validated": (
                    plan.model_dump(mode="json")
                    if plan is not None
                    else cast(Any, decision).model_dump(mode="json")
                ),
            },
            artifact_prefix="policy-raw",
        )
        return PolicyModelResult(
            operation=request.operation,
            decision=decision,
            proposed_plan=plan,
            raw_output_artifact=raw_artifact,
            stats=PolicyInferenceStats(
                operation=request.operation,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                repair_count=len(attempts) - 1,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            ),
        )
