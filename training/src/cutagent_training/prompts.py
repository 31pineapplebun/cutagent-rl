"""Compatibility re-export for the runtime-owned M6 inference prompt."""

from __future__ import annotations

from cutagent.models.policy_prompt_m6 import M6_PROMPT_VERSION, render_m6_prompt

__all__ = ["M6_PROMPT_VERSION", "render_m6_prompt"]
