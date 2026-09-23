"""Structured failures for the M3A tool boundary."""

from __future__ import annotations

from cutagent.schemas.tools import ToolErrorCategory, ToolValidationCheck


class ToolFailure(Exception):
    """Expected tool failure that is converted into a ToolObservation."""

    def __init__(
        self,
        category: ToolErrorCategory,
        public_summary: str,
        *,
        validation_results: tuple[ToolValidationCheck, ...] = (),
        ffmpeg_return_code: int | None = None,
    ) -> None:
        super().__init__(public_summary)
        self.category = category
        self.public_summary = public_summary
        self.validation_results = validation_results
        self.ffmpeg_return_code = ffmpeg_return_code


class ToolTimeout(ToolFailure):
    def __init__(self, public_summary: str = "tool execution exceeded its timeout") -> None:
        super().__init__("timeout", public_summary)
