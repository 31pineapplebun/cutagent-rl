"""Typed M0 errors."""


class CutAgentError(Exception):
    """Base error for the runtime package."""


class StateTransitionError(CutAgentError):
    """Raised when an event cannot be applied to an Agent state."""


class DuplicateEventError(StateTransitionError):
    """Raised when an event identifier has already been reduced."""


class EventSequenceError(StateTransitionError):
    """Raised when an event sequence number is not the next number."""


class ParentStateVersionError(StateTransitionError):
    """Raised when an event was created for a different state version."""


class TaskMismatchError(StateTransitionError):
    """Raised when an event belongs to another task."""


class TerminalStateError(StateTransitionError):
    """Raised when an event is applied after termination."""


class InvalidEventError(StateTransitionError):
    """Raised when event content violates a reducer invariant."""


class PolicyVisibilityError(CutAgentError):
    """Raised when model-visible serialization would expose private data."""


class MediaProbeError(CutAgentError):
    """Raised when ffprobe fails or reports an unsupported media contract."""


class MediaProbeTimeoutError(MediaProbeError):
    """Raised when ffprobe exceeds an explicitly bounded execution time."""


class MediaProcessingError(CutAgentError):
    """Raised when deterministic media processing fails."""


class CacheError(CutAgentError):
    """Raised when a content-addressed cache entry is invalid."""


class PerceptionError(CutAgentError):
    """Raised when a perception backend cannot produce a supported result."""


class StructuredOutputError(PerceptionError):
    """Raised after bounded structured-output repair is exhausted."""

    def __init__(self, message: str, *, attempts: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.attempts = attempts


class EvidenceIntegrityError(PerceptionError):
    """Raised when a perception claim references unavailable evidence."""
