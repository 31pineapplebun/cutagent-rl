"""Primary local M1B model backends."""

from cutagent.perception.backends.qwen import Qwen3VLVLMBackend
from cutagent.perception.backends.whisper import WhisperLargeV3TurboBackend

__all__ = ["Qwen3VLVLMBackend", "WhisperLargeV3TurboBackend"]
