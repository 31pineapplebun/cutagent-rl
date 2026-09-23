"""Qwen-visible-text OCR baseline and replaceable protocol implementation."""

import hashlib

from cutagent.schemas.perception import OCRSpan, VisualObservation


def normalize_ocr_text(text: str) -> str:
    return " ".join(text.casefold().split())


class VLMVisibleTextOCRBackend:
    """Promote only explicitly direct visible-text claims into OCR spans."""

    @property
    def backend_version(self) -> str:
        return "vlm-direct-visible-text-v1"

    def extract(self, observations: tuple[VisualObservation, ...]) -> tuple[OCRSpan, ...]:
        spans: list[OCRSpan] = []
        for observation in observations:
            for index, text in enumerate(observation.directly_visible_text):
                observed_times = {
                    reference.observed_ms
                    for reference in text.evidence_refs
                    if reference.observed_ms is not None
                }
                observed_ms = next(iter(observed_times)) if len(observed_times) == 1 else None
                digest = hashlib.sha256(
                    (
                        f"{observation.observation_id}:{index}:{text.exact_text}:"
                        f"{','.join(ref.artifact_id for ref in text.evidence_refs)}"
                    ).encode()
                ).hexdigest()[:20]
                spans.append(
                    OCRSpan(
                        span_id=f"ocr-{digest}",
                        exact_text=text.exact_text,
                        normalized_text=text.normalized_text or normalize_ocr_text(text.exact_text),
                        segment_id=observation.segment_id,
                        observed_ms=observed_ms,
                        time_range=None if observed_ms is not None else observation.time_range,
                        confidence=None,
                        evidence_refs=text.evidence_refs,
                    )
                )
        return tuple(spans)
