"""Run the mandatory M1B eager-inference compatibility spike.

This command intentionally sits outside the ordinary pytest suite. It downloads
the two pinned model revisions into an explicitly supplied external cache and
runs real local CUDA inference on synthetic, provenance-free media.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

torch: Any = importlib.import_module("torch")
Image: Any = importlib.import_module("PIL.Image")
ImageDraw: Any = importlib.import_module("PIL.ImageDraw")
process_vision_info: Any = importlib.import_module("qwen_vl_utils").process_vision_info
transformers: Any = importlib.import_module("transformers")
AutoModelForImageTextToText: Any = transformers.AutoModelForImageTextToText
AutoModelForSpeechSeq2Seq: Any = transformers.AutoModelForSpeechSeq2Seq
AutoProcessor: Any = transformers.AutoProcessor

QWEN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
QWEN_LICENSE = "apache-2.0"
WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
WHISPER_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
WHISPER_LICENSE = "mit"
SEED = 3407

STRUCTURED_PROMPT = """Inspect the supplied visual evidence. Return only a JSON object with exactly
these keys: scene_summary (string), entities (array of strings), actions (array of strings),
directly_visible_text (array of strings), uncertainties (array of strings). Do not use markdown.
Only report content directly supported by the supplied visual evidence."""


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}"
        )


def _set_determinism() -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": SEED,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "warn_only": True,
    }


def _make_visual_fixtures(root: Path) -> tuple[list[Path], Path]:
    root.mkdir(parents=True, exist_ok=True)
    colors = ("red", "green", "blue")
    frames: list[Path] = []
    for index, color in enumerate(colors):
        image = Image.new("RGB", (384, 256), "white")
        draw = ImageDraw.Draw(image)
        left = 30 + index * 105
        draw.rectangle((left, 80, left + 70, 150), fill=color)
        draw.text((15, 15), f"CUTAGENT {index + 1}", fill="black")
        path = root / f"frame_{index + 1:02d}.png"
        image.save(path)
        frames.append(path)

    video_path = root / "short_scene.mp4"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "1",
            "-i",
            str(root / "frame_%02d.png"),
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
    )
    return frames, video_path


def _make_audio_fixture(root: Path) -> tuple[Path, str, str]:
    audio_path = root / "spoken_fixture.wav"
    transcript = "The red square moves to the right."
    flite_filter = f"flite=text='{transcript}'"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        flite_filter,
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    try:
        _run(command)
        return audio_path, transcript, "FFmpeg libflite deterministic synthesis"
    except RuntimeError:
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
                "sine=frequency=440:duration=3",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(audio_path),
            ]
        )
        return audio_path, "", "FFmpeg sine fallback (non-speech compatibility input)"


def _memory_snapshot() -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _parse_structured_output(output: str) -> tuple[bool, dict[str, Any] | None, str | None]:
    try:
        value = json.loads(output)
        if not isinstance(value, dict):
            raise ValueError("top-level value is not an object")
        expected = {
            "scene_summary": str,
            "entities": list,
            "actions": list,
            "directly_visible_text": list,
            "uncertainties": list,
        }
        if set(value) != set(expected):
            raise ValueError(f"keys differ: {sorted(value)}")
        for key, expected_type in expected.items():
            if not isinstance(value[key], expected_type):
                raise ValueError(f"{key} must be {expected_type.__name__}")
        return True, value, None
    except (json.JSONDecodeError, ValueError) as error:
        return False, None, str(error)


def _prepare_qwen_inputs(
    processor: Any, messages: list[dict[str, Any]]
) -> tuple[Any, dict[str, Any]]:
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_metadata = None
    if videos is not None:
        unpacked_videos, unpacked_metadata = zip(*videos, strict=True)
        videos = list(unpacked_videos)
        video_metadata = list(unpacked_metadata)
    inputs = processor(
        text=prompt,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
    stats = {
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "image_count": 0 if images is None else len(images),
        "video_count": 0 if videos is None else len(videos),
        "video_kwargs": {
            key: value
            for key, value in video_kwargs.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
    }
    return inputs, stats


def _run_qwen_case(
    *, model: Any, processor: Any, name: str, messages: list[dict[str, Any]], frame_count: int
) -> dict[str, Any]:
    inputs, input_stats = _prepare_qwen_inputs(processor, messages)
    inputs = inputs.to(model.device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=192, do_sample=False)
    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated, strict=True)
    ]
    output = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    valid, parsed, validation_error = _parse_structured_output(output)
    return {
        "name": name,
        "status": "passed",
        "latency_seconds": latency,
        "input": {**input_stats, "frames": frame_count},
        "output": output,
        "structured_output_valid": valid,
        "parsed_output": parsed,
        "validation_error": validation_error,
        "memory": _memory_snapshot(),
    }


def _decode_audio(path: Path) -> np.ndarray:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
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


def _release_cuda_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_smoke(*, output_path: Path, artifact_root: Path, model_cache: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "running",
        "models": {
            "qwen": {
                "model_id": QWEN_MODEL_ID,
                "revision": QWEN_REVISION,
                "license": QWEN_LICENSE,
                "dtype": "bfloat16",
            },
            "whisper": {
                "model_id": WHISPER_MODEL_ID,
                "revision": WHISPER_REVISION,
                "license": WHISPER_LICENSE,
                "dtype": "float16",
            },
        },
        "determinism": _set_determinism(),
        "runtime": {
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "huggingface_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
            "video_reader_backend": os.environ.get("FORCE_QWENVL_VIDEO_READER"),
            "transformers": importlib.metadata.version("transformers"),
            "qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
            "torchcodec": importlib.metadata.version("torchcodec"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        frames, video_path = _make_visual_fixtures(artifact_root / "fixtures")
        audio_path, transcript, audio_method = _make_audio_fixture(artifact_root / "fixtures")
        report["fixtures"] = {
            "frames": [str(path) for path in frames],
            "video": str(video_path),
            "audio": str(audio_path),
            "audio_reference_transcript": transcript,
            "audio_generation": audio_method,
        }

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        qwen_model = AutoModelForImageTextToText.from_pretrained(
            QWEN_MODEL_ID,
            revision=QWEN_REVISION,
            cache_dir=model_cache,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        qwen_processor = AutoProcessor.from_pretrained(
            QWEN_MODEL_ID, revision=QWEN_REVISION, cache_dir=model_cache
        )
        torch.cuda.synchronize()
        report["models"]["qwen"].update(
            {
                "load_seconds": time.perf_counter() - load_started,
                "load_memory": _memory_snapshot(),
            }
        )
        image_uri = frames[0].resolve().as_uri()
        multi_image_content: list[dict[str, Any]] = [
            {"type": "image", "image": path.resolve().as_uri()} for path in frames
        ]
        qwen_cases: list[dict[str, Any]] = []
        report["qwen_cases"] = qwen_cases
        qwen_cases.append(
            _run_qwen_case(
                model=qwen_model,
                processor=qwen_processor,
                name="single_image",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_uri},
                            {"type": "text", "text": STRUCTURED_PROMPT},
                        ],
                    }
                ],
                frame_count=1,
            )
        )
        qwen_cases.append(
            _run_qwen_case(
                model=qwen_model,
                processor=qwen_processor,
                name="multiple_keyframes",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            *multi_image_content,
                            {"type": "text", "text": STRUCTURED_PROMPT},
                        ],
                    }
                ],
                frame_count=len(frames),
            )
        )
        qwen_cases.append(
            _run_qwen_case(
                model=qwen_model,
                processor=qwen_processor,
                name="native_short_video",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path.resolve().as_uri(),
                                "fps": 1.0,
                                "max_pixels": 128 * 32 * 32,
                                "total_pixels": 4096 * 32 * 32,
                            },
                            {"type": "text", "text": STRUCTURED_PROMPT},
                        ],
                    }
                ],
                frame_count=len(frames),
            )
        )
        del qwen_model
        del qwen_processor
        _release_cuda_memory()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL_ID,
            revision=WHISPER_REVISION,
            cache_dir=model_cache,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to("cuda:0")
        whisper_processor = AutoProcessor.from_pretrained(
            WHISPER_MODEL_ID, revision=WHISPER_REVISION, cache_dir=model_cache
        )
        torch.cuda.synchronize()
        report["models"]["whisper"].update(
            {
                "load_seconds": time.perf_counter() - load_started,
                "load_memory": _memory_snapshot(),
            }
        )
        audio = _decode_audio(audio_path)
        whisper_inputs = whisper_processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device="cuda:0", dtype=torch.float16)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = whisper_model.generate(whisper_inputs)
        torch.cuda.synchronize()
        output = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        report["whisper_case"] = {
            "name": "short_audio",
            "status": "passed",
            "duration_seconds": len(audio) / 16000,
            "samples": len(audio),
            "latency_seconds": time.perf_counter() - started,
            "output": output,
            "memory": _memory_snapshot(),
        }
        del whisper_model
        del whisper_processor
        _release_cuda_memory()
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m1b/model_smoke.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m1b/model_smoke"))
    parser.add_argument(
        "--model-cache",
        type=Path,
        required=True,
        help="External model cache; must not be inside the Git repository.",
    )
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    model_cache = args.model_cache.resolve()
    if model_cache == repository_root or repository_root in model_cache.parents:
        parser.error("--model-cache must be outside the Git repository")
    report = run_smoke(
        output_path=args.output.resolve(),
        artifact_root=args.artifact_root.resolve(),
        model_cache=model_cache,
    )
    print(f"M1B model smoke status={report['status']} output={args.output.resolve()}")
    if report["status"] != "passed":
        print(report.get("error"))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
