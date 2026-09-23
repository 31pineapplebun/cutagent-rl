"""Whisper timestamp mapping onto the M1A normalized timeline."""

from pathlib import Path

from cutagent.core.artifacts import ArtifactRef
from cutagent.perception.backends.whisper import normalize_whisper_chunks


def _audio_ref(tmp_path: Path) -> ArtifactRef:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"synthetic-audio")
    return ArtifactRef.from_path(path, artifact_id="audio-001", media_type="audio/wav")


def test_audio_offset_is_applied_without_assuming_shared_zero_pts(tmp_path: Path) -> None:
    spans = normalize_whisper_chunks(
        [{"text": " hello ", "timestamp": (0.1, 0.6)}],
        audio_artifact=_audio_ref(tmp_path),
        duration_ms=3000,
        timeline_offset_ms=250,
        language="en",
    )
    assert spans[0].time_range.start_ms == 350
    assert spans[0].time_range.end_ms == 850
    assert spans[0].text == "hello"


def test_negative_audio_offset_is_clipped_to_video_timeline(tmp_path: Path) -> None:
    spans = normalize_whisper_chunks(
        [{"text": "start", "timestamp": (0.0, 0.4)}],
        audio_artifact=_audio_ref(tmp_path),
        duration_ms=1000,
        timeline_offset_ms=-200,
        language=None,
    )
    assert spans[0].time_range.start_ms == 0
    assert spans[0].time_range.end_ms == 200
