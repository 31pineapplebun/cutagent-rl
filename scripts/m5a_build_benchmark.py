"""Materialize, audit, calibrate, and freeze CutAgentBench v0.1 without Agent runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

from cutagent_evaluation.m5a_benchmark import (
    canonical_json_bytes,
    licensed_qualitative_registry,
    write_benchmark,
)
from cutagent_evaluation.m5a_calibration import run_handcrafted_calibration
from cutagent_evaluation.m5a_dataset import build_benchmark_cases, materialize_source_registry
from cutagent_evaluation.m5a_schemas import BenchmarkSourceRecord

from cutagent.core.artifacts import ArtifactRef


def _artifact_path(artifact: ArtifactRef) -> Path:
    parsed = urlparse(artifact.uri)
    if parsed.scheme != "file":
        raise ValueError("CutAgentBench materialized source must be a local file artifact")
    path_text = unquote(parsed.path)
    if parsed.netloc:
        path_text = f"//{parsed.netloc}{path_text}"
    if len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":":
        path_text = path_text[1:]
    return Path(path_text).resolve(strict=True)


def _verify_source(source: BenchmarkSourceRecord) -> None:
    path = _artifact_path(source.source_artifact)
    observed = ArtifactRef.from_path(
        path,
        artifact_id=source.source_artifact.artifact_id,
        media_type=source.source_artifact.media_type,
    )
    if (
        observed.sha256 != source.source_sha256
        or observed.size_bytes != source.source_artifact.size_bytes
    ):
        raise RuntimeError(f"materialized source integrity failed for {source.source_group_id}")


def _probe_source(source: BenchmarkSourceRecord, ffprobe: str) -> dict[str, object]:
    path = _artifact_path(source.source_artifact)
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe rejected benchmark source {source.source_group_id}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    duration_ms = round(float(payload["format"]["duration"]) * 1000)
    passed = (
        isinstance(video, dict)
        and video.get("width") == 384
        and video.get("height") == 256
        and isinstance(audio, dict)
        and abs(duration_ms - 15_000) <= 120
    )
    if not passed:
        raise RuntimeError(f"media contract failed for {source.source_group_id}")
    if not isinstance(video, dict) or not isinstance(audio, dict):
        raise RuntimeError("media probe stream narrowing failed")
    return {
        "source_group_id": source.source_group_id,
        "video_id": source.video_id,
        "duration_ms": duration_ms,
        "width": video["width"],
        "height": video["height"],
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "sha256": source.source_sha256,
        "passed": True,
    }


def _load_cached_sources(path: Path) -> tuple[BenchmarkSourceRecord, ...] | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError("cached source registry must be a JSON list")
    sources = tuple(BenchmarkSourceRecord.model_validate(item) for item in raw)
    for source in sources:
        _verify_source(source)
    return sources


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/m5a/cutagentbench_v0.1"),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--licensed-provenance",
        type=Path,
        default=Path("artifacts/m1b_5/real_media/provenance.json"),
    )
    parser.add_argument("--force-media-rebuild", action="store_true")
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    registry_path = root / "private/source_group_registry.json"
    sources = None if args.force_media_rebuild else _load_cached_sources(registry_path)
    cache_hit = sources is not None
    if sources is None:
        sources = materialize_source_registry(root / "media", ffmpeg=args.ffmpeg)
    media_validation = tuple(_probe_source(source, args.ffprobe) for source in sources)
    cases = build_benchmark_cases(sources)
    result = write_benchmark(root, sources, cases)
    calibration = run_handcrafted_calibration(cases)
    _write(root / "calibration/objective_calibration.json", calibration)
    _write(root / "media_validation.json", media_validation)
    licensed_registry = None
    if args.licensed_provenance.is_file():
        licensed_registry = licensed_qualitative_registry(args.licensed_provenance)
        _write(root / "licensed_qualitative_registry.json", licensed_registry)
    locked_usage = {
        "benchmark_version": result.manifest.benchmark_version,
        "agent_model_runs_on_locked_test": 0,
        "agent_model_runs_on_adversarial_test": 0,
        "construction_operations": ["schema validation", "manifest hashing", "seal hashing"],
        "m5b_reserved_first_official_locked_run": True,
    }
    _write(root / "locked_test_usage.json", locked_usage)
    summary = {
        "manifest_sha256": result.manifest_sha256,
        "task_count": result.manifest.task_count,
        "source_group_count": result.manifest.source_group_count,
        "split_counts": result.health.split_counts,
        "leakage_audit_passed": result.audit.passed,
        "objective_calibration_cases": calibration.case_count,
        "objective_tsr_agreement": calibration.exact_tsr_agreement,
        "human_rater_count": calibration.human_rater_count,
        "media_validation_passed": sum(item["passed"] is True for item in media_validation),
        "licensed_qualitative_source_count": (
            licensed_registry.source_count if licensed_registry is not None else 0
        ),
        "licensed_qualitative_clip_count": (
            licensed_registry.clip_count if licensed_registry is not None else 0
        ),
        "source_media_cache_hit": cache_hit,
        "source_registry_sha256": hashlib.sha256(canonical_json_bytes(tuple(sources))).hexdigest(),
    }
    _write(root / "construction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
