"""Pinned local Qwen3-VL backend for the versioned M4B protocol."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import time
from typing import Any, cast

from pydantic import ValidationError

from cutagent.agent.m4b_protocols import M4BPolicyModelRequest, M4BPolicyModelResult
from cutagent.core.errors import StructuredOutputError
from cutagent.perception.artifacts import write_json_artifact
from cutagent.perception.structured_output import parse_json_object
from cutagent.schemas.agent import POLICY_DECISION_ADAPTER, PlanGraph, ReplanDecision
from cutagent.schemas.m4b_agent import (
    RECOVERY_DECISION_ADAPTER,
    M4BPolicyInferenceStats,
)

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

_PLAN_CONTRACT = """Return exactly one JSON object without markdown. The top level has integer
revision=1 and nodes as a JSON array of 2-6 task-specific local nodes. Every node has exactly:
node_id:string, subgoal:string, dependencies:array[string], status:"ready"|"pending",
expected_evidence:array[string], preferred_capability:null, and
completion_criteria:array[string]. All array fields must remain JSON arrays even with one item.
Use unique descriptive IDs. A root is ready; a dependent node is pending. Dependencies may only
reference earlier node IDs. preferred_capability MUST be JSON null in the initial plan; the runtime
tool manifest remains authoritative for capability selection. Include only operations actually
needed by the public task: one node per retrieval/edit/validation
operation. When editing is required, final validation depends on the last edit. Never add generic
placeholder, export, return, completion, or finish nodes. FinishDecision is outside the PlanGraph.
Never include tool arguments.
"""

_DECISION_CONTRACT = """Return exactly one balanced JSON object without markdown. Choose one:
1) tool:
{"decision_type":"tool","tool_call":{"tool_name":"validate_media",
"tool_call_id":"call-001","arguments":{"input_artifact_id":"opaque-id"}},
"rationale":"why","expected_observation":"what","success_condition":"condition"}
2) finish:
{"decision_type":"finish","output_artifact_id":"opaque-id",
"completion_summary":"completed work","evidence_ids":["opaque-evidence-id"]}
3) cannot_complete:
{"decision_type":"cannot_complete","reason":"why",
"missing_evidence_or_capability":["missing item"],"attempted_action_ids":["call-001"]}
Use only registered tool schemas and opaque IDs. Never emit shell commands, FFmpeg strings, paths,
hashes, or private data. Read working_artifacts and step_outcome directly. Do not reconstruct a new
artifact ID from history. If completion_ready is true, finish with final_candidate_artifact_id.
After an editor succeeds, use current_working_media for subsequent edits or validation. Do not
repeat an equivalent editor on an older source. If verification is inconclusive, inspect missing
evidence rather than treating it as failure.
"""

_FULL_REPLAN_CONTRACT = """Return exactly one balanced ReplanDecision JSON object:
{"decision_type":"replan","reason":"why","affected_plan_nodes":["failed-node"],
"requested_patch":{"event_type":"plan_patch","revision":2,"reason":"why",
"steps":[{"node_id":"node","subgoal":"subgoal","dependencies":[],"status":"ready",
"expected_evidence":["evidence"],"preferred_capability":"media.write",
"completion_criteria":["criterion"]}]}}.
Preserve succeeded and unrelated nodes exactly. This handoff-only ablation intentionally retains
the M4A complex replan contract.
"""

_RECOVERY_CONTRACT = """Return exactly one compact RecoveryDecision JSON object without markdown.
Choose one local operation; never return a PlanGraph, dependency edge, PlanPatch, or tool call:
1) {"recovery_type":"retry_current_node","node_id":"failed-node","reason":"why",
"preferred_capability":"media.inspect"}
2) {"recovery_type":"modify_current_node","node_id":"failed-node","reason":"why",
"revised_subgoal":"local revised subgoal","revised_completion_criteria":["criterion"],
"preferred_capability":"media.write"}
3) {"recovery_type":"insert_recovery_node","affected_node_id":"failed-node","reason":"why",
"recovery_subgoal":"obtain missing observable evidence","completion_criteria":["criterion"],
"preferred_capability":"retrieval.read"}
4) {"recovery_type":"skip_blocked_node","node_id":"failed-node","reason":"why"}
5) {"recovery_type":"cannot_recover","reason":"why",
"missing_evidence_or_capability":["missing item"]}
Target only the active failed/blocked node shown in step_outcome. The runtime owns graph mutation.
"""


class Qwen3VLPolicyBackendM4B:
    """Transformers eager BF16 backend with strict M4B structured output."""

    def __init__(self, *, model_cache: Any) -> None:
        self._model_cache = model_cache
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._process_vision_info: Any | None = None
        self.load_time_ms = 0

    @property
    def backend_version(self) -> str:
        return (
            "cutagent-qwen3-vl-policy-m4b-eager-v5+"
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
    def _tool_contract(request: M4BPolicyModelRequest) -> str:
        return json.dumps(
            [
                {
                    "name": item.name,
                    "version": item.version,
                    "description": item.description,
                    "arguments": item.argument_schema,
                }
                for item in request.tool_manifest.tools
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _prompt(
        self,
        request: M4BPolicyModelRequest,
        *,
        previous: str | None = None,
        error: str | None = None,
    ) -> str:
        contract = {
            "plan": _PLAN_CONTRACT,
            "decide": _DECISION_CONTRACT,
            "replan": _FULL_REPLAN_CONTRACT,
            "recover": _RECOVERY_CONTRACT,
        }[request.operation]
        variant_rule = (
            "Use the handoff-only protocol; replan retains the full M4A PlanPatch contract."
            if request.protocol_variant == "handoff_only"
            else "Use compact recovery; never generate a PlanPatch during recovery."
        )
        repair = ""
        if previous is not None and error is not None:
            repair = (
                "\nPrevious output was invalid. Return one corrected object, not commentary.\n"
                f"Validation error: {error}\n"
                "The previous bytes are intentionally omitted to prevent repetition.\n"
            )
        return (
            "You are the CutAgent-RL M4B structured-state video-tool policy. "
            "Be conservative and evidence-grounded.\n"
            f"Operation: {request.operation}\nProtocol: {variant_rule}\n"
            f"Prompt version: {request.prompt_template_version}\nSeed: {request.seed}\n"
            f"{contract}\nRegistered tools: {self._tool_contract(request)}\n"
            f"Whitelisted policy context: {request.serialized_context}\n{repair}"
        )

    def _generate(self, request: M4BPolicyModelRequest, prompt: str) -> tuple[str, int, int]:
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        self._torch.manual_seed(request.seed)
        self._torch.cuda.manual_seed_all(request.seed)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)
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
    def _parse_payload(raw: str) -> dict[str, Any]:
        """Accept one object or exact repeated copies; reject prose and divergent objects."""

        try:
            return parse_json_object(raw)
        except StructuredOutputError as original:
            text = raw.strip()
            decoder = json.JSONDecoder()
            try:
                first, offset = decoder.raw_decode(text)
            except json.JSONDecodeError:
                raise original from None
            if not isinstance(first, dict):
                raise original
            remainder = text[offset:].strip()
            while remainder:
                try:
                    repeated, offset = decoder.raw_decode(remainder)
                except json.JSONDecodeError:
                    raise original from None
                if repeated != first:
                    raise original
                remainder = remainder[offset:].strip()
            return cast(dict[str, Any], first)

    @staticmethod
    def _parse(request: M4BPolicyModelRequest, raw: str) -> tuple[PlanGraph | None, Any | None]:
        payload = Qwen3VLPolicyBackendM4B._parse_payload(raw)
        if request.operation == "plan":
            plan = PlanGraph.model_validate(payload)
            if any(node.preferred_capability is not None for node in plan.nodes):
                raise ValueError("initial M4B plan must leave preferred_capability null")
            if not 2 <= len(plan.nodes) <= 6:
                raise ValueError("M4B plan must contain 2-6 task-specific nodes")
            return plan, None
        if request.operation == "recover":
            return None, RECOVERY_DECISION_ADAPTER.validate_python(payload)
        decision = POLICY_DECISION_ADAPTER.validate_python(payload)
        if request.operation == "replan" and not isinstance(decision, ReplanDecision):
            raise ValueError("handoff-only replan requires ReplanDecision")
        if request.operation == "decide" and isinstance(decision, ReplanDecision):
            raise ValueError("ordinary decide cannot emit ReplanDecision")
        return None, decision

    def infer(self, request: M4BPolicyModelRequest) -> M4BPolicyModelResult:
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
            raw, current_input, current_output = self._generate(
                request,
                self._prompt(request, previous=previous, error=validation_error),
            )
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
                        artifact_prefix="m4b-policy-raw-failure",
                    )
                    raise StructuredOutputError(
                        f"M4B policy output invalid; artifact={failure.artifact_id}",
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
            artifact_prefix="m4b-policy-raw",
        )
        return M4BPolicyModelResult(
            operation=request.operation,
            decision=decision,
            proposed_plan=plan,
            raw_output_artifact=raw_artifact,
            stats=M4BPolicyInferenceStats(
                operation=request.operation,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                repair_count=len(attempts) - 1,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            ),
        )
