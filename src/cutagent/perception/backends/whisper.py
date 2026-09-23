"""Local Transformers Whisper-large-v3-turbo ASR backend."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import time
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import numpy as np

from cutagent.core.artifacts import ArtifactRef
from cutagent.perception.artifacts import write_json_artifact
from cutagent.perception.protocols import ASRBackendResult, ASRRequest
from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import EvidenceRef, ModelPerformance, TranscriptSpan


def _seconds_to_ms(value: float) -> int:
    return int((Decimal(str(value)) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def normalize_whisper_chunks(
    chunks: list[Any],
    *,
    audio_artifact: ArtifactRef,
    duration_ms: int,
    timeline_offset_ms: int,
    language: str | None,
) -> tuple[TranscriptSpan, ...]:
    """Map Whisper's extracted-audio seconds onto M1A normalized milliseconds."""

    spans: list[TranscriptSpan] = []
    for index, item in enumerate(chunks):
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text", "")).split())
        timestamps = item.get("timestamp")
        if not text or not isinstance(timestamps, (tuple, list)) or len(timestamps) != 2:
            continue
        raw_start, raw_end = timestamps
        if raw_start is None or raw_end is None:
            continue
        start_ms = timeline_offset_ms + _seconds_to_ms(float(raw_start))
        end_ms = timeline_offset_ms + _seconds_to_ms(float(raw_end))
        start_ms = max(0, min(start_ms, duration_ms - 1))
        end_ms = max(start_ms + 1, min(end_ms, duration_ms))
        digest = hashlib.sha256(
            f"{audio_artifact.sha256}:{index}:{start_ms}:{end_ms}:{text}".encode()
        ).hexdigest()[:20]
        spans.append(
            TranscriptSpan(
                span_id=f"transcript-{digest}",
                text=text,
                time_range=TimeRange(start_ms=start_ms, end_ms=end_ms),
                language=language,
                confidence=None,
                evidence_refs=(
                    EvidenceRef(
                        artifact_id=audio_artifact.artifact_id,
                        evidence_kind="source_audio_pcm",
                        time_range=TimeRange(start_ms=start_ms, end_ms=end_ms),
                    ),
                ),
            )
        )
    return tuple(spans)


class WhisperLargeV3TurboBackend:
    """Pinned local eager-inference implementation; no external inference API."""

    def __init__(self, *, model_cache: str) -> None:
        self._model_cache = model_cache
        self._pipe: Any | None = None
        self._torch: Any | None = None

    @property
    def backend_version(self) -> str:
        return (
            "cutagent-whisper-eager-v1+"
            f"torch-{importlib.metadata.version('torch')}+"
            f"transformers-{importlib.metadata.version('transformers')}"
        )

    def _load(self, request: ASRRequest) -> None:
        if self._pipe is not None:
            return
        torch: Any = importlib.import_module("torch")
        transformers: Any = importlib.import_module("transformers")
        spec = request.config.whisper
        model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            cache_dir=self._model_cache,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to("cuda:0")
        processor = transformers.AutoProcessor.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            cache_dir=self._model_cache,
        )
        self._pipe = transformers.pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=torch.float16,
            device="cuda:0",
        )
        self._torch = torch

    @staticmethod
    def _waveform(path: str) -> np.ndarray:
        import subprocess

        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                path,
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
        )
        return np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe(self, request: ASRRequest) -> ASRBackendResult:
        self._load(request)
        assert self._pipe is not None
        assert self._torch is not None
        torch = self._torch
        waveform = self._waveform(str(request.audio_path))
        rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter_ns()
        if rms <= request.config.asr_silence_rms_threshold:
            raw: dict[str, Any] = {
                "text": "",
                "chunks": [],
                "silence_gate": True,
                "rms": rms,
            }
        else:
            generation: dict[str, Any] = {"task": "transcribe"}
            if request.config.asr_language is not None:
                generation["language"] = request.config.asr_language
            raw_result = self._pipe(
                {"array": waveform, "sampling_rate": 16000},
                return_timestamps=True,
                generate_kwargs=generation,
            )
            raw = dict(raw_result)
            raw["silence_gate"] = False
            raw["rms"] = rms
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter_ns() - started) // 1_000_000
        chunks = raw.get("chunks")
        normalized_chunks = chunks if isinstance(chunks, list) else []
        spans = normalize_whisper_chunks(
            normalized_chunks,
            audio_artifact=request.audio_artifact,
            duration_ms=request.video.duration_ms,
            timeline_offset_ms=request.timeline_offset_ms,
            language=request.config.asr_language,
        )
        raw["normalized_timeline_offset_ms"] = request.timeline_offset_ms
        raw["normalized_spans"] = [span.model_dump(mode="json") for span in spans]
        raw_artifact = write_json_artifact(
            request.output_directory / "raw_whisper_output.json",
            raw,
            artifact_prefix="asr-raw",
        )
        spec = request.config.whisper
        return ASRBackendResult(
            spans=spans,
            raw_output=raw_artifact,
            performance=ModelPerformance(
                operation="whisper_asr",
                model_id=spec.model_id,
                model_revision=spec.revision,
                dtype=spec.dtype,
                device="cuda:0",
                latency_ms=latency_ms,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                frames=0,
            ),
        )
