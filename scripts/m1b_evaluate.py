"""Generate and evaluate the 20-case private M1B perception benchmark."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from cutagent_evaluation.m1b_metrics import (
    PerceptionCaseGold,
    evaluate_asr_result,
    evaluate_visual_result,
)

from cutagent.ingestion.pipeline import VideoIngestionPipeline
from cutagent.perception.pipeline import MultimodalPerceptionPipeline
from cutagent.schemas.media import IngestionConfig, KeyframeExtractionConfig, TimeRange
from cutagent.schemas.perception import PerceptionConfig

Image: Any = importlib.import_module("PIL.Image")
ImageDraw: Any = importlib.import_module("PIL.ImageDraw")
ImageFont: Any = importlib.import_module("PIL.ImageFont")


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{completed.stderr}")


def _font(size: int) -> Any:
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    windows = Path("C:/Windows/Fonts/msyh.ttc")
    if windows.is_file():
        return ImageFont.truetype(str(windows), size=size)
    return ImageFont.load_default()


def _audio_duration_ms(path: Path) -> int:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    seconds = float(json.loads(completed.stdout)["format"]["duration"])
    return max(1, min(2000, round(seconds * 1000)))


def generate_case(root: Path, index: int) -> tuple[Path, dict[str, Any]]:
    case_id = f"case-{index + 1:03d}"
    case_root = root / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    colors = ("red", "blue", "green", "yellow")
    shapes = ("square", "circle")
    color = colors[index % len(colors)]
    shape = shapes[(index // len(colors)) % len(shapes)]
    moving = index % 2 == 0
    visible_text = f"CASE {index + 1:02d}" if index % 4 < 2 else f"测试 {index + 1:02d}"
    font = _font(28)
    for frame_index in range(4):
        image = Image.new("RGB", (384, 256), "white")
        draw = ImageDraw.Draw(image)
        left = 30 + frame_index * 75 if moving else 145
        if shape == "square":
            draw.rectangle((left, 100, left + 65, 165), fill=color)
        else:
            draw.ellipse((left, 100, left + 65, 165), fill=color)
        draw.text((18, 20), visible_text, font=font, fill="black")
        image.save(case_root / f"frame_{frame_index + 1:02d}.png")
    reference_transcript = f"{color.capitalize()} {shape}."
    audio = case_root / "speech.wav"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"flite=text='{reference_transcript}'",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio),
        ]
    )
    video = case_root / "source.mp4"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "2",
            "-i",
            str(case_root / "frame_%02d.png"),
            "-i",
            str(audio),
            "-filter_complex",
            "[1:a]apad=pad_dur=2[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-t",
            "2",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video),
        ]
    )
    return video, {
        "case_id": case_id,
        "color": color,
        "shape": shape,
        "moving": moving,
        "visible_text": visible_text,
        "reference_transcript": reference_transcript,
        "speech_end_ms": _audio_duration_ms(audio),
        "generation": {
            "frames": 4,
            "fps": 2,
            "duration_ms": 2000,
            "font": "Noto Sans CJK Bold (OFL)" if "测试" in visible_text else "Noto/host font",
            "speech": "FFmpeg libflite deterministic synthesis",
        },
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_visual(rows: list[dict[str, Any]], attempted: int) -> dict[str, Any]:
    successes = [row for row in rows if row.get("status") == "passed"]
    predicted_entities = sum(len(cast_list(row.get("predicted_entities"))) for row in successes)
    hallucinated = sum(len(cast_list(row.get("hallucinated_entities"))) for row in successes)
    return {
        "attempted_cases": attempted,
        "successful_cases": len(successes),
        "structured_output_validity": len(successes) / attempted,
        "entity_correctness": _mean([float(bool(row["entity_correct"])) for row in successes]),
        "action_correctness": _mean([float(bool(row["action_correct"])) for row in successes]),
        "visible_text_exact_accuracy": _mean(
            [float(bool(row["visible_text_exact"])) for row in successes]
        ),
        "visible_text_normalized_accuracy": _mean(
            [float(bool(row["visible_text_normalized"])) for row in successes]
        ),
        "visible_text_mean_cer": _mean([float(row["visible_text_cer"]) for row in successes]),
        "ocr_temporal_evidence_correctness": _mean(
            [float(bool(row["ocr_evidence_correct"])) for row in successes]
        ),
        "temporal_event_correctness": _mean(
            [float(bool(row["temporal_event_correct"])) for row in successes]
        ),
        "hallucination_rate": hallucinated / max(1, predicted_entities),
        "evidence_integrity_rate": _mean(
            [float(bool(row["evidence_integrity"])) for row in successes]
        ),
        "mean_latency_ms": _mean([float(row["latency_ms"]) for row in successes]),
        "peak_allocated_bytes": max(
            (int(row["peak_allocated_bytes"]) for row in successes), default=0
        ),
        "peak_reserved_bytes": max(
            (int(row["peak_reserved_bytes"]) for row in successes), default=0
        ),
        "mean_frames": _mean([float(row["frames"]) for row in successes]),
        "mean_input_tokens": _mean([float(row["input_tokens"]) for row in successes]),
        "total_repairs": sum(int(row["repair_count"]) for row in successes),
    }


def cast_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("artifacts/m1b/evaluation"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/m1b/evaluation_results.json")
    )
    parser.add_argument("--cases", type=int, default=20, choices=range(20, 41))
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    model_cache = args.model_cache.resolve()
    if model_cache == repository_root or repository_root in model_cache.parents:
        parser.error("--model-cache must be outside the Git repository")
    root = args.work_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ingestion_pipeline = VideoIngestionPipeline(cache_root=root / "ingestion-cache")
    perception_pipeline = MultimodalPerceptionPipeline(
        cache_root=root / "perception-cache",
        model_cache=model_cache,
    )
    gold_records: list[PerceptionCaseGold] = []
    keyframe_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    asr_rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    case_outputs: list[dict[str, Any]] = []
    started = time.perf_counter_ns()
    for index in range(args.cases):
        source, specification = generate_case(root / "cases", index)
        ingestion = ingestion_pipeline.ingest(
            source,
            config=IngestionConfig(
                scene_threshold=0.99,
                minimum_scene_duration_ms=300,
                keyframes=KeyframeExtractionConfig(strategy="uniform", frames_per_scene=3),
            ),
        )
        if len(ingestion.scenes) != 1:
            raise RuntimeError(
                f"generated evaluation case is not one scene: {specification['case_id']}"
            )
        gold = PerceptionCaseGold(
            case_id=str(specification["case_id"]),
            source_group_id=f"generated-{specification['case_id']}",
            source_sha256=ingestion.video.source.sha256,
            license="deterministically-generated; Noto font render under OFL where used",
            expected_entities=(
                f"{specification['color']} {specification['shape']}",
                "text",
                "white background",
            ),
            expected_action="moves_right" if specification["moving"] else None,
            expected_visible_text=str(specification["visible_text"]),
            reference_transcript=str(specification["reference_transcript"]),
            reference_speech_range=TimeRange(
                start_ms=0, end_ms=int(specification["speech_end_ms"])
            ),
            expected_event_range=(
                TimeRange(start_ms=0, end_ms=2000) if specification["moving"] else None
            ),
        )
        gold_records.append(gold)
        case_record: dict[str, Any] = {
            "case_id": gold.case_id,
            "source_sha256": gold.source_sha256,
            "generation": specification["generation"],
        }
        for mode in ("keyframes", "native_video"):
            config = PerceptionConfig(
                visual_mode=mode,
                asr_enabled=mode == "keyframes",
                asr_language="en" if mode == "keyframes" else None,
            )
            try:
                result = perception_pipeline.run(source, ingestion=ingestion, config=config)
                visual_metrics = evaluate_visual_result(result, gold)
                row = {"case_id": gold.case_id, "status": "passed", **visual_metrics}
                if mode == "keyframes":
                    keyframe_rows.append(row)
                    asr_metric = evaluate_asr_result(result, gold)
                    asr_rows.append({"case_id": gold.case_id, **asr_metric})
                    wer_value = asr_metric["wer"]
                    assert isinstance(wer_value, (int, float))
                    if float(wer_value) > 0:
                        reference_words = [
                            token.strip(".,!?;:").casefold()
                            for token in str(asr_metric["reference"]).split()
                        ]
                        hypothesis_words = [
                            token.strip(".,!?;:").casefold()
                            for token in str(asr_metric["hypothesis"]).split()
                        ]
                        if not hypothesis_words or len(hypothesis_words) < len(reference_words):
                            failures["asr_deletion"] += 1
                        elif len(hypothesis_words) > len(reference_words):
                            failures["asr_insertion"] += 1
                        else:
                            failures["asr_substitution"] += 1
                else:
                    native_rows.append(row)
                if not bool(visual_metrics["entity_correct"]):
                    failures["missed_entity"] += 1
                failures["hallucinated_entity"] += len(
                    cast_list(visual_metrics["hallucinated_entities"])
                )
                if not bool(visual_metrics["action_correct"]):
                    failures["wrong_action"] += 1
                if not bool(visual_metrics["visible_text_exact"]):
                    failures["wrong_visible_text"] += 1
                if not bool(visual_metrics["temporal_event_correct"]):
                    failures[
                        "temporal_misalignment"
                        if gold.expected_event_range
                        else "unsupported_event"
                    ] += 1
                result_path = root / "results" / mode / f"{gold.case_id}.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
                case_record[mode] = row
            except Exception as error:
                failures["malformed_structured_output"] += 1
                row = {
                    "case_id": gold.case_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                (keyframe_rows if mode == "keyframes" else native_rows).append(row)
                case_record[mode] = row
        case_outputs.append(case_record)

    private_gold_path = root / "private_gold.json"
    private_gold_path.write_text(
        json.dumps(
            [gold.model_dump(mode="json") for gold in gold_records],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    asr_summary = {
        "cases": len(asr_rows),
        "mean_wer": _mean([float(row["wer"]) for row in asr_rows]),
        "mean_cer": _mean([float(row["cer"]) for row in asr_rows]),
        "mean_timestamp_alignment_error_ms": _mean(
            [
                float(row["timestamp_alignment_error_ms"])
                for row in asr_rows
                if row["timestamp_alignment_error_ms"] is not None
            ]
        ),
    }
    report = {
        "schema_version": "1.0.0",
        "status": "passed",
        "case_count": args.cases,
        "categories": {
            "deterministically_generated": args.cases,
            "real_open_licensed": 0,
            "real_open_licensed_status": (
                "not included; no clip was accepted without explicit source provenance verification"
            ),
        },
        "models": {
            "qwen": PerceptionConfig().qwen.model_dump(mode="json"),
            "whisper": PerceptionConfig().whisper.model_dump(mode="json"),
        },
        "keyframes": aggregate_visual(keyframe_rows, args.cases),
        "native_video": aggregate_visual(native_rows, args.cases),
        "asr": asr_summary,
        "failure_taxonomy": dict(sorted(failures.items())),
        "rows": case_outputs,
        "asr_rows": asr_rows,
        "private_gold_path": str(private_gold_path),
        "processing_time_ms": (time.perf_counter_ns() - started) // 1_000_000,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"M1B evaluation status=passed cases={args.cases} output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
