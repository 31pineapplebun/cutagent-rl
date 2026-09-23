"""M1B integration tests use fake backends and real tiny M1A media."""

from pathlib import Path

from cutagent.ingestion.pipeline import VideoIngestionPipeline
from cutagent.perception.artifacts import write_json_artifact
from cutagent.perception.pipeline import MultimodalPerceptionPipeline
from cutagent.perception.protocols import (
    ASRBackendResult,
    ASRRequest,
    VLMBackendResult,
    VLMRequest,
)
from cutagent.schemas.media import (
    IngestionConfig,
    IngestionResult,
    KeyframeExtractionConfig,
    TimeRange,
)
from cutagent.schemas.perception import (
    DirectlyVisibleText,
    EvidenceRef,
    ModelPerformance,
    PerceptionConfig,
    TranscriptSpan,
    VisualEntity,
    VisualObservation,
)
from tests.media_fixtures import generate_multiscene_with_audio, require_ffmpeg


class FakeASRBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.offsets: list[int] = []

    @property
    def backend_version(self) -> str:
        return "fake-asr-v1"

    def transcribe(self, request: ASRRequest) -> ASRBackendResult:
        self.calls += 1
        self.offsets.append(request.timeline_offset_ms)
        start_ms = max(0, request.timeline_offset_ms)
        span_range = TimeRange(start_ms=start_ms, end_ms=start_ms + 500)
        span = TranscriptSpan(
            span_id="transcript-fake",
            text="synthetic speech",
            time_range=span_range,
            language="en",
            evidence_refs=(
                EvidenceRef(
                    artifact_id=request.audio_artifact.artifact_id,
                    evidence_kind="source_audio_pcm",
                    time_range=span_range,
                ),
            ),
        )
        raw = write_json_artifact(
            request.output_directory / "fake_asr.json",
            {"span": span.model_dump(mode="json")},
            artifact_prefix="asr-raw",
        )
        return ASRBackendResult(
            spans=(span,),
            raw_output=raw,
            performance=ModelPerformance(
                operation="fake_asr",
                model_id="fake-asr",
                model_revision="fake-revision",
                dtype="float32",
                device="cpu",
                latency_ms=1,
            ),
        )


class FakeVLMBackend:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def backend_version(self) -> str:
        return "fake-vlm-v1"

    def perceive(self, request: VLMRequest) -> VLMBackendResult:
        self.calls += 1
        evidence = request.visual_inputs[0].evidence_ref
        observation = VisualObservation(
            observation_id=f"visual-{request.scene.segment_id}-{request.config.visual_mode}",
            segment_id=request.scene.segment_id,
            mode=request.config.visual_mode,
            time_range=request.scene.time_range,
            scene_summary=f"Observed {request.scene.segment_id}.",
            summary_evidence_refs=(evidence,),
            entities=(VisualEntity(label="colored frame", evidence_refs=(evidence,)),),
            directly_visible_text=(
                DirectlyVisibleText(exact_text="CUTAGENT", evidence_refs=(evidence,)),
            ),
        )
        raw = write_json_artifact(
            request.output_directory / "fake_vlm.json",
            {"scene": request.scene.segment_id, "observation": observation.model_dump(mode="json")},
            artifact_prefix="vlm-raw",
        )
        return VLMBackendResult(
            observation=observation,
            raw_output=raw,
            performance=ModelPerformance(
                operation="fake_vlm",
                model_id="fake-vlm",
                model_revision="fake-revision",
                dtype="float32",
                device="cpu",
                latency_ms=1,
                frames=len(request.visual_inputs),
            ),
        )


def _ingest(tmp_path: Path) -> tuple[Path, IngestionResult]:
    ffmpeg, ffprobe = require_ffmpeg()
    source = generate_multiscene_with_audio(tmp_path / "perception.mp4")
    ingestion = VideoIngestionPipeline(
        cache_root=tmp_path / "ingestion-cache",
        ffmpeg_executable=ffmpeg,
        ffprobe_executable=ffprobe,
    ).ingest(
        source,
        config=IngestionConfig(
            scene_threshold=0.1,
            minimum_scene_duration_ms=300,
            keyframes=KeyframeExtractionConfig(strategy="midpoint"),
        ),
    )
    return source, ingestion


def test_m1a_to_m1b_world_state_and_heavy_cache(tmp_path: Path) -> None:
    source, ingestion = _ingest(tmp_path)
    fake_asr = FakeASRBackend()
    fake_vlm = FakeVLMBackend()
    pipeline = MultimodalPerceptionPipeline(
        cache_root=tmp_path / "perception-cache",
        asr_backend=fake_asr,
        vlm_backend=fake_vlm,
        ffmpeg_executable=require_ffmpeg()[0],
    )

    first = pipeline.run(source, ingestion=ingestion, config=PerceptionConfig())
    second = pipeline.run(source, ingestion=ingestion, config=PerceptionConfig())

    assert len(first.world_state.scenes) == 3
    assert len(first.transcript_spans) == 1
    assert len(first.ocr_spans) == 3
    assert len(first.visual_observations) == 3
    assert fake_asr.calls == 1
    assert fake_vlm.calls == 3
    assert all(not record.hit for record in first.cache_records)
    assert all(record.hit for record in second.cache_records)
    assert first.world_state == second.world_state
    assert first.transcript_spans == second.transcript_spans


def test_native_video_mode_creates_scene_clip_evidence(tmp_path: Path) -> None:
    source, ingestion = _ingest(tmp_path)
    pipeline = MultimodalPerceptionPipeline(
        cache_root=tmp_path / "native-cache",
        asr_backend=FakeASRBackend(),
        vlm_backend=FakeVLMBackend(),
        ffmpeg_executable=require_ffmpeg()[0],
    )
    result = pipeline.run(
        source,
        ingestion=ingestion,
        config=PerceptionConfig(visual_mode="native_video", asr_enabled=False),
    )
    assert sum(record.operation == "scene_clip" for record in result.cache_records) == 3
    assert all(
        observation.summary_evidence_refs[0].evidence_kind == "scene_clip"
        for observation in result.visual_observations
    )


def test_prompt_change_invalidates_only_vlm_cache_key(tmp_path: Path) -> None:
    source, ingestion = _ingest(tmp_path)
    fake_asr = FakeASRBackend()
    fake_vlm = FakeVLMBackend()
    pipeline = MultimodalPerceptionPipeline(
        cache_root=tmp_path / "prompt-cache",
        asr_backend=fake_asr,
        vlm_backend=fake_vlm,
        ffmpeg_executable=require_ffmpeg()[0],
    )
    pipeline.run(source, ingestion=ingestion, config=PerceptionConfig())
    changed = pipeline.run(
        source,
        ingestion=ingestion,
        config=PerceptionConfig(prompt_template_version="m1b-structured-v2"),
    )
    assert fake_asr.calls == 1
    assert fake_vlm.calls == 6
    assert next(record for record in changed.cache_records if record.operation == "asr").hit
    assert all(
        not record.hit for record in changed.cache_records if record.operation == "vlm_perception"
    )


def test_temporal_ablation_change_invalidates_only_vlm_cache_key(tmp_path: Path) -> None:
    source, ingestion = _ingest(tmp_path)
    fake_asr = FakeASRBackend()
    fake_vlm = FakeVLMBackend()
    pipeline = MultimodalPerceptionPipeline(
        cache_root=tmp_path / "temporal-prompt-cache",
        asr_backend=fake_asr,
        vlm_backend=fake_vlm,
        ffmpeg_executable=require_ffmpeg()[0],
    )
    pipeline.run(source, ingestion=ingestion, config=PerceptionConfig())
    changed = pipeline.run(
        source,
        ingestion=ingestion,
        config=PerceptionConfig(
            prompt_template_version="m1b5-explicit-temporal-v1",
            temporal_prompt_style="explicit_comparison",
            label_temporal_phases=True,
        ),
    )

    assert fake_asr.calls == 1
    assert fake_vlm.calls == 6
    assert next(record for record in changed.cache_records if record.operation == "asr").hit
    assert all(
        not record.hit for record in changed.cache_records if record.operation == "vlm_perception"
    )


def test_model_revision_change_invalidates_vlm_cache_key(tmp_path: Path) -> None:
    """Exercise cache identity independently of the production revision literal."""

    source, ingestion = _ingest(tmp_path)
    fake_asr = FakeASRBackend()
    fake_vlm = FakeVLMBackend()
    pipeline = MultimodalPerceptionPipeline(
        cache_root=tmp_path / "revision-cache",
        asr_backend=fake_asr,
        vlm_backend=fake_vlm,
        ffmpeg_executable=require_ffmpeg()[0],
    )
    base = PerceptionConfig()
    pipeline.run(source, ingestion=ingestion, config=base)
    # model_copy deliberately bypasses the frozen Literal for this cache-contract
    # test; validated runtime configs still accept only the pinned revision.
    future_qwen = base.qwen.model_copy(update={"revision": "future-revision"})
    changed_config = base.model_copy(update={"qwen": future_qwen})
    changed = pipeline.run(source, ingestion=ingestion, config=changed_config)

    assert fake_asr.calls == 1
    assert fake_vlm.calls == 6
    assert next(record for record in changed.cache_records if record.operation == "asr").hit
    assert all(
        not record.hit for record in changed.cache_records if record.operation == "vlm_perception"
    )


def test_evidence_timestamp_change_invalidates_affected_vlm_cache_key(tmp_path: Path) -> None:
    source, ingestion = _ingest(tmp_path)
    fake_asr = FakeASRBackend()
    fake_vlm = FakeVLMBackend()
    pipeline = MultimodalPerceptionPipeline(
        cache_root=tmp_path / "evidence-time-cache",
        asr_backend=fake_asr,
        vlm_backend=fake_vlm,
        ffmpeg_executable=require_ffmpeg()[0],
    )
    pipeline.run(source, ingestion=ingestion, config=PerceptionConfig())
    first = ingestion.keyframes[0]
    assert first.observed_timestamp_ms is not None
    shifted = first.model_copy(update={"observed_timestamp_ms": first.observed_timestamp_ms + 1})
    shifted_ingestion = ingestion.model_copy(
        update={"keyframes": (shifted, *ingestion.keyframes[1:])}
    )
    changed = pipeline.run(
        source,
        ingestion=shifted_ingestion,
        config=PerceptionConfig(),
    )
    vlm_records = [
        record for record in changed.cache_records if record.operation == "vlm_perception"
    ]

    assert fake_vlm.calls == 4
    assert sum(not record.hit for record in vlm_records) == 1
    assert sum(record.hit for record in vlm_records) == 2
