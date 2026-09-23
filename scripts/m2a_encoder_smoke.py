"""Verify the pinned M2A text and visual encoders on the real CUDA runtime."""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

TEXT_MODEL_ID = "BAAI/bge-m3"
TEXT_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
TEXT_MODEL_LICENSE = "mit"
VISUAL_MODEL_ID = "google/siglip2-base-patch16-224"
VISUAL_MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
VISUAL_MODEL_LICENSE = "apache-2.0"


def _pooled_feature_tensor(value: Any) -> Any:
    pooled = getattr(value, "pooler_output", None)
    return pooled if pooled is not None else value


def _cuda_measure(torch: Any, operation: Callable[[], Any]) -> tuple[Any, dict[str, int]]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    value = operation()
    torch.cuda.synchronize()
    return value, {
        "latency_ms": (time.perf_counter_ns() - started) // 1_000_000,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _to_cuda_batch(torch: Any, batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value.to(device="cuda:0", dtype=torch.bfloat16)
            if value.is_floating_point()
            else value.to(device="cuda:0")
        )
        for key, value in batch.items()
    }


def _release(torch: Any) -> None:
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/m2a/encoder_smoke"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/m2a/encoder_smoke.json"))
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    model_cache = args.model_cache.resolve()
    if model_cache == repository_root or repository_root in model_cache.parents:
        parser.error("--model-cache must be outside the Git repository")
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    torch: Any = importlib.import_module("torch")
    transformers: Any = importlib.import_module("transformers")
    image_module: Any = importlib.import_module("PIL.Image")
    image_draw: Any = importlib.import_module("PIL.ImageDraw")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real M2A encoder smoke")
    torch.manual_seed(20260822)
    torch.cuda.manual_seed_all(20260822)

    text_load_started = time.perf_counter_ns()
    torch.cuda.reset_peak_memory_stats()
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(
        TEXT_MODEL_ID,
        revision=TEXT_MODEL_REVISION,
        cache_dir=model_cache,
    )
    text_model = transformers.AutoModel.from_pretrained(
        TEXT_MODEL_ID,
        revision=TEXT_MODEL_REVISION,
        cache_dir=model_cache,
        dtype=torch.bfloat16,
    ).to("cuda:0")
    text_model.eval()
    torch.cuda.synchronize()
    text_load = {
        "latency_ms": (time.perf_counter_ns() - text_load_started) // 1_000_000,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    text_samples = [
        "A red square moves to the left.",
        "红色方块向左移动。",
        "A blue circle remains stationary.",
    ]

    def encode_text() -> Any:
        batch = text_tokenizer(
            text_samples,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch = _to_cuda_batch(torch, dict(batch))
        with torch.inference_mode():
            hidden = text_model(**batch).last_hidden_state[:, 0]
            return torch.nn.functional.normalize(hidden.float(), p=2, dim=-1)

    text_embeddings, text_inference = _cuda_measure(torch, encode_text)
    text_similarity = (text_embeddings @ text_embeddings.T).cpu().tolist()
    text_result = {
        "model_id": TEXT_MODEL_ID,
        "revision": TEXT_MODEL_REVISION,
        "license": TEXT_MODEL_LICENSE,
        "dtype": "bfloat16",
        "device": "cuda:0",
        "embedding_dimension": int(text_embeddings.shape[1]),
        "sample_count": len(text_samples),
        "load": text_load,
        "inference": text_inference,
        "embeddings_per_second": len(text_samples) / (max(1, text_inference["latency_ms"]) / 1000),
        "similarity_matrix": text_similarity,
        "finite": bool(torch.isfinite(text_embeddings).all().item()),
    }
    del text_embeddings
    text_model = None
    text_tokenizer = None
    _release(torch)

    image_specs = (("red_square", "red", "square"), ("blue_circle", "blue", "circle"))
    images = []
    image_paths = []
    for name, color, shape in image_specs:
        image = image_module.new("RGB", (224, 224), "white")
        draw = image_draw.Draw(image)
        if shape == "square":
            draw.rectangle((52, 52, 172, 172), fill=color)
        else:
            draw.ellipse((52, 52, 172, 172), fill=color)
        path = artifact_root / f"{name}.png"
        image.save(path)
        images.append(image)
        image_paths.append(path.name)

    visual_load_started = time.perf_counter_ns()
    torch.cuda.reset_peak_memory_stats()
    visual_processor = transformers.AutoProcessor.from_pretrained(
        VISUAL_MODEL_ID,
        revision=VISUAL_MODEL_REVISION,
        cache_dir=model_cache,
    )
    visual_model = transformers.AutoModel.from_pretrained(
        VISUAL_MODEL_ID,
        revision=VISUAL_MODEL_REVISION,
        cache_dir=model_cache,
        dtype=torch.bfloat16,
    ).to("cuda:0")
    visual_model.eval()
    torch.cuda.synchronize()
    visual_load = {
        "latency_ms": (time.perf_counter_ns() - visual_load_started) // 1_000_000,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    visual_queries = ["a red square", "红色正方形", "a blue circle", "蓝色圆形"]

    def encode_visual_pair() -> tuple[Any, Any]:
        text_batch = visual_processor(
            text=visual_queries,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        image_batch = visual_processor(images=images, return_tensors="pt")
        text_batch = _to_cuda_batch(torch, dict(text_batch))
        image_batch = _to_cuda_batch(torch, dict(image_batch))
        with torch.inference_mode():
            text_features = _pooled_feature_tensor(visual_model.get_text_features(**text_batch))
            image_features = _pooled_feature_tensor(visual_model.get_image_features(**image_batch))
        return (
            torch.nn.functional.normalize(text_features.float(), p=2, dim=-1),
            torch.nn.functional.normalize(image_features.float(), p=2, dim=-1),
        )

    (visual_text_embeddings, image_embeddings), visual_inference = _cuda_measure(
        torch, encode_visual_pair
    )
    cross_modal_similarity = (visual_text_embeddings @ image_embeddings.T).cpu().tolist()
    visual_result = {
        "model_id": VISUAL_MODEL_ID,
        "revision": VISUAL_MODEL_REVISION,
        "license": VISUAL_MODEL_LICENSE,
        "dtype": "bfloat16",
        "device": "cuda:0",
        "embedding_dimension": int(image_embeddings.shape[1]),
        "query_count": len(visual_queries),
        "image_count": len(images),
        "image_paths": image_paths,
        "load": visual_load,
        "inference": visual_inference,
        "images_per_second": len(images) / (max(1, visual_inference["latency_ms"]) / 1000),
        "cross_modal_similarity": cross_modal_similarity,
        "finite": bool(
            torch.isfinite(visual_text_embeddings).all().item()
            and torch.isfinite(image_embeddings).all().item()
        ),
    }
    expected_top_images = [0, 0, 1, 1]
    visual_result["top_image_indices"] = [
        max(range(len(row)), key=row.__getitem__) for row in cross_modal_similarity
    ]
    visual_result["expected_top_image_indices"] = expected_top_images
    visual_result["semantic_smoke_passed"] = (
        visual_result["top_image_indices"] == expected_top_images
    )

    status = "passed" if text_result["finite"] and visual_result["finite"] else "failed"
    report = {
        "schema_version": "1.0.0",
        "status": status,
        "seed": 20260822,
        "runtime": {
            "python": importlib.import_module("platform").python_version(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "gpu": torch.cuda.get_device_name(0),
        },
        "text_encoder": text_result,
        "visual_encoder": visual_result,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"M2A encoder smoke status={status} output={output}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
