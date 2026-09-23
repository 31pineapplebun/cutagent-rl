"""Versioned M6 policy prompt used by dataset construction and adapter inference."""

from __future__ import annotations

import json
from typing import Literal

from cutagent.schemas.tools import ToolManifest

M6_PROMPT_VERSION = "m6-decision-sft-v1"

_CONTRACTS = {
    "decide": (
        "Return exactly one JSON PolicyDecision: ToolDecision, FinishDecision, or "
        "CannotCompleteDecision. Use only registered typed tools and opaque artifact IDs."
    ),
    "replan": (
        "Return exactly one constrained ReplanDecision JSON object. Preserve succeeded history."
    ),
    "recover": (
        "Return exactly one compact RecoveryDecision JSON object. Do not emit a PlanGraph, "
        "PlanPatch, tool call, shell command, or prose."
    ),
}


def render_m6_prompt(
    *,
    operation: Literal["decide", "replan", "recover"],
    serialized_context: str,
    tool_manifest: ToolManifest,
) -> str:
    tools = [
        {
            "name": item.name,
            "version": item.version,
            "description": item.description,
            "arguments": item.argument_schema,
        }
        for item in tool_manifest.tools
    ]
    return (
        "You are the CutAgent-RL structured-state video-tool policy. "
        "Use only observable evidence and be conservative.\n"
        f"Operation: {operation}\nPrompt version: {M6_PROMPT_VERSION}\n"
        f"Contract: {_CONTRACTS[operation]}\n"
        "Never output shell commands, FFmpeg strings, filesystem paths, hashes, private labels, "
        "or evaluator metadata. ToolObservation and VerificationResult are environment inputs, "
        "never outputs.\n"
        "Registered tools: "
        f"{json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
        f"Whitelisted policy context: {serialized_context}"
    )
