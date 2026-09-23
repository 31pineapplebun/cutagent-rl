"""On-demand local Qwen native-video verification for ambiguous motion candidates."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cutagent.core.artifacts import ArtifactRef
from cutagent.ingestion.cache import CacheKeyBuilder, ContentAddressedCache
from cutagent.ingestion.ffmpeg import FFmpegRunner
from cutagent.perception.artifacts import write_json_artifact
from cutagent.schemas.base import NonEmptyStr, SchemaModel
from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import EvidenceRef, QwenModelSpec
from cutagent.schemas.retrieval import (
    AdaptiveRetrievalConfig,
    NativeVerificationStatus,
    NativeVideoVerification,
    RetrievalCandidate,
    RetrievalQuery,
)

NATIVE_VERIFIER_PROMPT = """You verify whether a candidate video scene directly supports a user
retrieval query. Inspect motion and temporal change across the video, not just objects in one
frame. Return exactly one JSON object without markdown:
{"status":"supports|contradicts|uncertain","explanation":string,
 "unsupported_claims":[string]}
Use supports only when visible scene evidence supports the requested action/direction/order.
Use contradicts only when visible evidence clearly shows an incompatible action or direction.
Use uncertain when sampling, visibility, or the scene does not establish the claim. Do not invent
precise timestamps, speech, OCR, confidence, people, or objects. The output is verification
evidence, never benchmark ground truth.
"""


class NativeVerifierOutput(SchemaModel):
    status: NativeVerificationStatus
    explanation: NonEmptyStr
    unsupported_claims: tuple[NonEmptyStr, ...] = ()

    @classmethod
    def validate_status(cls, raw: str) -> NativeVerifierOutput:
        return cls.model_validate_json(raw)


@dataclass(frozen=True, slots=True)
class SceneClipAsset:
    video_id: str
    scene_id: str
    time_range: TimeRange
    artifact: ArtifactRef
    path: Path


def _artifact(path: Path, *, prefix: str, media_type: str) -> ArtifactRef:
    provisional = ArtifactRef.from_path(
        path,
        artifact_id=f"{prefix}-provisional",
        media_type=media_type,
    )
    return ArtifactRef(
        **{
            **provisional.model_dump(),
            "artifact_id": f"{prefix}-{provisional.sha256[:20]}",
        }
    )


class SceneClipStore:
    """Materialize immutable source-timeline scene clips for verification."""

    version = "m2b-scene-clip-v1"

    def __init__(self, root: Path, *, ffmpeg_executable: str = "ffmpeg") -> None:
        self.cache = ContentAddressedCache(root)
        self.ffmpeg = FFmpegRunner(ffmpeg_executable, timeout_seconds=600)

    def materialize(
        self,
        *,
        source_path: Path,
        source_artifact: ArtifactRef,
        video_id: str,
        scene_id: str,
        time_range: TimeRange,
    ) -> tuple[SceneClipAsset, bool]:
        source = source_path.resolve(strict=True)
        observed = ArtifactRef.from_path(
            source,
            artifact_id=source_artifact.artifact_id,
            media_type=source_artifact.media_type,
        )
        if observed.sha256 != source_artifact.sha256:
            raise ValueError("scene-clip source path does not match source artifact")
        key = CacheKeyBuilder.build(
            source_sha256=source_artifact.sha256,
            operation="m2b_native_verification_scene_clip",
            config={
                "video_id": video_id,
                "scene_id": scene_id,
                "time_range": time_range.model_dump(mode="json"),
                "codec": "mpeg4",
                "timestamp_mapping": "clip_zero_maps_to_scene_start_ms",
            },
            tool_versions={"ffmpeg": self.ffmpeg.version, "cutagent": self.version},
        )
        output = self.cache.path("scene_clip", key, "scene.mp4")
        hit = output.is_file()
        if not hit:
            self.ffmpeg.run(
                (
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{time_range.start_ms / 1000:.3f}",
                    "-i",
                    str(source),
                    "-t",
                    f"{time_range.duration_ms / 1000:.3f}",
                    "-an",
                    "-c:v",
                    "mpeg4",
                    "-q:v",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    str(output),
                ),
                operation=f"M2B scene clip {scene_id}",
            )
        reference = _artifact(output, prefix="m2b-scene-clip", media_type="video/mp4")
        return (
            SceneClipAsset(
                video_id=video_id,
                scene_id=scene_id,
                time_range=time_range,
                artifact=reference,
                path=output,
            ),
            hit,
        )


def build_native_verification_cache_key(
    *,
    query: RetrievalQuery,
    candidate: RetrievalCandidate,
    scene_clip: SceneClipAsset,
    config: AdaptiveRetrievalConfig,
    model: QwenModelSpec,
    index_id: str,
    native_video_fps: float,
    backend_version: str,
) -> str:
    """Pure version contract used by production code and cache-invalidation tests."""

    return CacheKeyBuilder.build(
        source_sha256=scene_clip.artifact.sha256,
        operation="native_video_verification",
        config={
            "query": query.model_dump(mode="json"),
            "candidate": {
                "video_id": candidate.video_id,
                "scene_id": candidate.scene_id,
                "time_range": candidate.time_range.model_dump(mode="json"),
            },
            "model": model.model_dump(mode="json"),
            "prompt_version": config.native_verifier_prompt_version,
            "routing_version": config.routing_policy_version,
            "reranker_version": config.reranker_rule_version,
            "candidate_depth": config.candidate_depth,
            "native_video_depth": config.native_video_depth,
            "native_video_fps": native_video_fps,
            "index_id": index_id,
        },
        tool_versions={"backend": backend_version},
    )


class QwenNativeVideoVerifier:
    """Pinned BF16, greedy Qwen verifier with query/config/model-aware caching."""

    def __init__(
        self,
        *,
        cache_root: Path,
        model_cache: Path,
        config: AdaptiveRetrievalConfig,
        index_id: str,
        model: QwenModelSpec | None = None,
        native_video_fps: float = 2.0,
        maximum_new_tokens: int = 160,
    ) -> None:
        self.cache = ContentAddressedCache(cache_root)
        self.model_cache = model_cache.resolve()
        self.config = config
        self.index_id = index_id
        self.spec = model or QwenModelSpec()
        self.native_video_fps = native_video_fps
        self.maximum_new_tokens = maximum_new_tokens
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._process_vision_info: Any | None = None

    @property
    def backend_version(self) -> str:
        return (
            "cutagent-m2b-native-verifier-v1+"
            f"torch-{importlib.metadata.version('torch')}+"
            f"transformers-{importlib.metadata.version('transformers')}+"
            f"qwen-vl-utils-{importlib.metadata.version('qwen-vl-utils')}"
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        torch: Any = importlib.import_module("torch")
        transformers: Any = importlib.import_module("transformers")
        self._model = transformers.AutoModelForImageTextToText.from_pretrained(
            self.spec.model_id,
            revision=self.spec.revision,
            cache_dir=self.model_cache,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self._processor = transformers.AutoProcessor.from_pretrained(
            self.spec.model_id,
            revision=self.spec.revision,
            cache_dir=self.model_cache,
        )
        self._torch = torch
        self._process_vision_info = importlib.import_module("qwen_vl_utils").process_vision_info

    def _cache_key(
        self,
        *,
        query: RetrievalQuery,
        candidate: RetrievalCandidate,
        scene_clip: SceneClipAsset,
    ) -> str:
        return build_native_verification_cache_key(
            query=query,
            candidate=candidate,
            scene_clip=scene_clip,
            config=self.config,
            model=self.spec,
            index_id=self.index_id,
            native_video_fps=self.native_video_fps,
            backend_version=self.backend_version,
        )

    def _generate(
        self, query: RetrievalQuery, candidate: RetrievalCandidate, path: Path
    ) -> tuple[str, int, int]:
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None
        assert self._process_vision_info is not None
        evidence_summary = "\n".join(
            f"- {item.evidence_type}: {item.matched_text or '[non-text evidence]'}"
            for item in candidate.evidence
        )
        prompt = (
            NATIVE_VERIFIER_PROMPT
            + f"\nUser query: {query.text}\nCandidate scene: {candidate.scene_id}; "
            + f"grounded range=[{candidate.time_range.start_ms},{candidate.time_range.end_ms}) ms."
            + "\nExisting non-authoritative retrieval evidence:\n"
            + evidence_summary
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": path.resolve(strict=True).as_uri(),
                        "fps": self.native_video_fps,
                        "max_pixels": 256 * 32 * 32,
                        "total_pixels": 8192 * 32 * 32,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = self._process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        metadata = None
        if videos is not None:
            tensors, video_metadata = zip(*videos, strict=True)
            videos = list(tensors)
            metadata = list(video_metadata)
        frames = sum(int(video.shape[0]) for video in videos or ())
        inputs = self._processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=metadata,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        ).to(self._model.device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.maximum_new_tokens,
                do_sample=False,
            )
        trimmed = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs.input_ids, generated, strict=True)
        ]
        raw = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return raw, int(inputs["input_ids"].shape[-1]), frames

    def verify(
        self,
        *,
        query: RetrievalQuery,
        candidate: RetrievalCandidate,
        scene_clip: SceneClipAsset,
    ) -> NativeVideoVerification:
        if (candidate.video_id, candidate.scene_id) != (
            scene_clip.video_id,
            scene_clip.scene_id,
        ):
            raise ValueError("candidate and scene clip identities differ")
        key = self._cache_key(query=query, candidate=candidate, scene_clip=scene_clip)
        cached = self.cache.read_json("native_video_verification", key, "result.json")
        if cached is not None:
            if not isinstance(cached, dict):
                raise ValueError("native verification cache must contain an object")
            return NativeVideoVerification.model_validate({**cached, "cache_hit": True})

        self._load()
        assert self._torch is not None
        torch = self._torch
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        attempts: list[str] = []
        raw, input_tokens, frames = self._generate(query, candidate, scene_clip.path)
        attempts.append(raw)
        try:
            parsed = NativeVerifierOutput.validate_status(raw)
        except (ValidationError, ValueError, json.JSONDecodeError) as first_error:
            repair_query = query.model_copy(
                update={
                    "text": (
                        f"{query.text}\nPrior JSON was invalid ({first_error}). "
                        "Return only a corrected object matching the required schema."
                    )
                }
            )
            raw, _, _ = self._generate(repair_query, candidate, scene_clip.path)
            attempts.append(raw)
            parsed = NativeVerifierOutput.validate_status(raw)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter_ns() - started) // 1_000_000
        evidence = EvidenceRef(
            artifact_id=scene_clip.artifact.artifact_id,
            evidence_kind="scene_clip",
            segment_id=candidate.scene_id,
            time_range=candidate.time_range,
        )
        raw_artifact = write_json_artifact(
            self.cache.path("native_video_verification", key, "raw_output.json"),
            {
                "prompt_version": self.config.native_verifier_prompt_version,
                "query": query.model_dump(mode="json"),
                "candidate": {
                    "video_id": candidate.video_id,
                    "scene_id": candidate.scene_id,
                    "time_range": candidate.time_range.model_dump(mode="json"),
                },
                "attempts": attempts,
                "validated": parsed.model_dump(mode="json"),
                "input_tokens": input_tokens,
                "frames": frames,
            },
            artifact_prefix="m2b-native-raw",
        )
        result = NativeVideoVerification(
            query_id=query.query_id,
            video_id=candidate.video_id,
            scene_id=candidate.scene_id,
            status=parsed.status,
            explanation=parsed.explanation,
            evidence_ref=evidence,
            raw_output_artifact=raw_artifact,
            prompt_version=self.config.native_verifier_prompt_version,
            model_id=self.spec.model_id,
            model_revision=self.spec.revision,
            latency_ms=latency_ms,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cache_hit=False,
            unsupported_claims=parsed.unsupported_claims,
        )
        self.cache.write_json(
            "native_video_verification",
            key,
            "result.json",
            result.model_dump(mode="json"),
        )
        return result
