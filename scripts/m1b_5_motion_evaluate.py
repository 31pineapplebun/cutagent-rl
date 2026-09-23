"""Run the frozen M1B.5 motion diagnostic and controlled Qwen3-VL ablations."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cutagent_evaluation.m1b5_motion import MotionCaseGold, evaluate_motion_result
from cutagent_evaluation.motion_dataset import generate_motion_dataset

from cutagent.ingestion.pipeline import VideoIngestionPipeline
from cutagent.perception.pipeline import MultimodalPerceptionPipeline
from cutagent.schemas.media import IngestionConfig, KeyframeExtractionConfig
from cutagent.schemas.perception import PerceptionConfig, PerceptionMode, TemporalPromptStyle


@dataclass(frozen=True, slots=True)
class DiagnosticVariant:
    variant_id: str
    mode: PerceptionMode
    keyframe_count: int
    native_fps: float
    prompt_style: TemporalPromptStyle
    label_phases: bool
    isolated_factor: str


VARIANTS: tuple[DiagnosticVariant, ...] = (
    DiagnosticVariant("kf3_baseline", "keyframes", 3, 2.0, "baseline", False, "keyframe baseline"),
    DiagnosticVariant(
        "kf5_baseline", "keyframes", 5, 2.0, "baseline", False, "keyframe count 3->5"
    ),
    DiagnosticVariant(
        "kf3_explicit_temporal",
        "keyframes",
        3,
        2.0,
        "explicit_comparison",
        True,
        "temporal comparison prompt package",
    ),
    DiagnosticVariant(
        "native_fps2_baseline", "native_video", 3, 2.0, "baseline", False, "native baseline"
    ),
    DiagnosticVariant(
        "native_fps4_baseline",
        "native_video",
        3,
        4.0,
        "baseline",
        False,
        "native sampling 2->4 fps",
    ),
    DiagnosticVariant(
        "native_fps2_explicit_temporal",
        "native_video",
        3,
        2.0,
        "explicit_comparison",
        False,
        "temporal comparison prompt",
    ),
)


def _set_seed(seed: int) -> dict[str, object]:
    random.seed(seed)
    numpy: Any = importlib.import_module("numpy")
    numpy.random.seed(seed)
    details: dict[str, object] = {"python": seed, "numpy": seed}
    try:
        torch: Any = importlib.import_module("torch")
    except ImportError:
        details["torch"] = "not installed"
        return details
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    details["torch"] = seed
    details["deterministic_algorithms"] = False
    return details


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def _booleans(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(value) for row in rows if isinstance((value := row.get(key)), bool)]


def aggregate(rows: list[dict[str, Any]], attempted: int) -> dict[str, object]:
    passed = [row for row in rows if row.get("status") == "passed"]
    return {
        "attempted_cases": attempted,
        "successful_cases": len(passed),
        "structured_output_validity": len(passed) / attempted,
        "entity_correctness": _mean(_numeric(passed, "entity_correctness")),
        "action_correctness": _mean(_booleans(passed, "action_correct")),
        "temporal_event_correctness": _mean(_booleans(passed, "temporal_event_correct")),
        "direction_of_motion_correctness": _mean(_booleans(passed, "direction_correct")),
        "event_order_correctness": _mean(_booleans(passed, "event_order_correct")),
        "mean_temporal_localization_error_ms": _mean(
            _numeric(passed, "temporal_localization_error_ms")
        ),
        "mean_unsupported_claim_rate": _mean(_numeric(passed, "unsupported_claim_rate")),
        "mean_latency_ms": _mean(_numeric(passed, "latency_ms")),
        "peak_allocated_bytes": max(
            (int(row["peak_allocated_bytes"]) for row in passed), default=0
        ),
        "peak_reserved_bytes": max((int(row["peak_reserved_bytes"]) for row in passed), default=0),
        "mean_frames": _mean(_numeric(passed, "frames")),
        "mean_input_tokens": _mean(_numeric(passed, "input_tokens")),
        "total_repairs": sum(int(row["repair_count"]) for row in passed),
        "heavy_model_cache_hit_rate": _mean(_booleans(passed, "vlm_cache_hit")),
    }


def failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    failures: Counter[str] = Counter()
    for row in rows:
        if row.get("status") != "passed":
            failures["malformed_structured_output"] += 1
            continue
        if float(row["entity_correctness"]) < 1:
            failures["missed_entity"] += 1
        if row["action_correct"] is False:
            failures["wrong_action"] += 1
        if row["temporal_event_correct"] is False:
            failures["temporal_misalignment"] += 1
        if row["direction_correct"] is False:
            failures["wrong_direction"] += 1
        if row["event_order_correct"] is False:
            failures["wrong_event_order"] += 1
        if float(row["unsupported_claim_rate"]) > 0:
            failures["unsupported_or_hallucinated_claim"] += 1
    return dict(sorted(failures.items()))


def _prompt_version(variant: DiagnosticVariant) -> str:
    return (
        "m1b5-explicit-temporal-v1"
        if variant.prompt_style == "explicit_comparison"
        else "m1b-structured-v1.2"
    )


def _config(variant: DiagnosticVariant) -> PerceptionConfig:
    return PerceptionConfig(
        visual_mode=variant.mode,
        prompt_template_version=_prompt_version(variant),
        temporal_prompt_style=variant.prompt_style,
        label_temporal_phases=variant.label_phases,
        native_video_fps=variant.native_fps,
        asr_enabled=False,
    )


def _dataset_hash(gold: tuple[MotionCaseGold, ...]) -> str:
    payload = json.dumps(
        [item.model_dump(mode="json") for item in gold],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("artifacts/m1b_5/motion_diagnostic"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/m1b_5/motion_results.json"))
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--case-limit", type=int, default=51, choices=range(1, 52))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[variant.variant_id for variant in VARIANTS],
        default=[variant.variant_id for variant in VARIANTS],
    )
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    model_cache = args.model_cache.resolve()
    if model_cache == repository_root or repository_root in model_cache.parents:
        parser.error("--model-cache must be outside the Git repository")
    selected_ids = set(args.variants)
    selected_variants = tuple(variant for variant in VARIANTS if variant.variant_id in selected_ids)
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    seed_record = _set_seed(args.seed)
    generated = generate_motion_dataset(work_root / "cases", seed=args.seed)
    generated = generated[: args.case_limit]
    gold = tuple(item[1] for item in generated)
    private_gold_path = work_root / "private_gold.json"
    private_gold_path.write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in gold],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ingestion_pipeline = VideoIngestionPipeline(cache_root=work_root / "ingestion-cache")
    perception_pipeline = MultimodalPerceptionPipeline(
        cache_root=work_root / "perception-cache",
        model_cache=model_cache,
    )
    rows_by_variant: dict[str, list[dict[str, Any]]] = {
        variant.variant_id: [] for variant in selected_variants
    }
    started = time.perf_counter_ns()
    for variant in selected_variants:
        for source, case_gold in generated:
            row: dict[str, Any]
            try:
                ingestion = ingestion_pipeline.ingest(
                    source,
                    config=IngestionConfig(
                        scene_threshold=0.99,
                        minimum_scene_duration_ms=300,
                        keyframes=KeyframeExtractionConfig(
                            strategy="uniform",
                            frames_per_scene=variant.keyframe_count,
                        ),
                    ),
                )
                if len(ingestion.scenes) != 1:
                    raise RuntimeError(f"motion fixture split into {len(ingestion.scenes)} scenes")
                result = perception_pipeline.run(
                    source,
                    ingestion=ingestion,
                    config=_config(variant),
                )
                metrics = evaluate_motion_result(result, case_gold)
                vlm_records = [
                    record
                    for record in result.cache_records
                    if record.operation == "vlm_perception"
                ]
                row = {
                    "case_id": case_gold.case_id,
                    "variant_id": variant.variant_id,
                    "status": "passed",
                    "vlm_cache_hit": bool(vlm_records)
                    and all(record.hit for record in vlm_records),
                    **metrics,
                }
                result_path = (
                    work_root
                    / "perception_results"
                    / variant.variant_id
                    / f"{case_gold.case_id}.json"
                )
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            except Exception as error:
                row = {
                    "case_id": case_gold.case_id,
                    "variant_id": variant.variant_id,
                    "task_type": case_gold.task_type,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            rows_by_variant[variant.variant_id].append(row)
            row_path = work_root / "rows" / variant.variant_id / f"{case_gold.case_id}.json"
            row_path.parent.mkdir(parents=True, exist_ok=True)
            row_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"variant={variant.variant_id} case={case_gold.case_id} status={row['status']}")

    variant_summaries = {
        variant.variant_id: {
            "definition": {
                "mode": variant.mode,
                "keyframe_count": variant.keyframe_count,
                "native_fps": variant.native_fps,
                "prompt_style": variant.prompt_style,
                "label_phases": variant.label_phases,
                "isolated_factor": variant.isolated_factor,
                "perception_config": _config(variant).model_dump(mode="json"),
            },
            "metrics": aggregate(rows_by_variant[variant.variant_id], len(generated)),
            "failures": failure_counts(rows_by_variant[variant.variant_id]),
        }
        for variant in selected_variants
    }
    failure_matrix: dict[str, dict[str, object]] = {}
    task_types = sorted({item.task_type for item in gold})
    for task_type in task_types:
        failure_matrix[task_type] = {}
        for variant in selected_variants:
            category_rows = [
                row
                for row in rows_by_variant[variant.variant_id]
                if row.get("task_type") == task_type
            ]
            failure_matrix[task_type][variant.variant_id] = {
                "metrics": aggregate(category_rows, len(category_rows)),
                "failures": failure_counts(category_rows),
            }
    baseline_pair: dict[str, object] = {}
    for variant_id in ("kf3_baseline", "native_fps2_baseline"):
        if variant_id in variant_summaries:
            baseline_pair[variant_id] = variant_summaries[variant_id]
    report = {
        "schema_version": "1.0.0",
        "status": "completed",
        "evaluation_protocol_version": "m1b5-motion-eval-v1",
        "seed": args.seed,
        "seed_record": seed_record,
        "case_count": len(generated),
        "task_type_count": len(task_types),
        "dataset_sha256": _dataset_hash(gold),
        "private_gold_path": str(private_gold_path),
        "models": {
            "qwen": PerceptionConfig().qwen.model_dump(mode="json"),
            "weights_changed": False,
        },
        "primary_keyframe_native_pair": baseline_pair,
        "variants": variant_summaries,
        "failure_matrix": failure_matrix,
        "rows": rows_by_variant,
        "processing_time_ms": (time.perf_counter_ns() - started) // 1_000_000,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"M1B.5 motion evaluation completed cases={len(generated)} "
        f"variants={len(selected_variants)} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
