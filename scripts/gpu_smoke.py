#!/usr/bin/env python3
"""Validate the real CUDA BF16 research runtime without loading a model."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
DEFAULT_EXPECTED_GPU = ""
UTC = timezone.utc  # noqa: UP017 -- keep the standalone diagnostic Python 3.10 compatible


def _write_result(result: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(f"{serialized}\n", encoding="utf-8")
    temporary.replace(output)


def _base_result(seed: int, expected_gpu: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(UTC).isoformat(),
        "status": "failed",
        "seed": seed,
        "expected_gpu": expected_gpu,
        "checks": {},
        "error": None,
    }


def run_gpu_smoke(*, seed: int, expected_gpu: str) -> dict[str, object]:
    """Run small CUDA computations and return a machine-readable result."""

    result = _base_result(seed, expected_gpu)
    checks: dict[str, object] = {}
    result["checks"] = checks

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)

    try:
        torch: Any = importlib.import_module("torch")
        cuda: Any = torch.cuda
        checks["torch_imported"] = True
        result["torch_version"] = str(torch.__version__)
        result["torch_cuda_runtime"] = getattr(torch.version, "cuda", None)
        result["python_version"] = sys.version.split()[0]

        cuda_available = bool(cuda.is_available())
        checks["cuda_available"] = cuda_available
        if not cuda_available:
            raise RuntimeError("torch.cuda.is_available() returned False")

        device_index = int(cuda.current_device())
        device_name = str(cuda.get_device_name(device_index))
        properties: Any = cuda.get_device_properties(device_index)
        total_memory = int(properties.total_memory)
        capability = tuple(int(value) for value in cuda.get_device_capability(device_index))
        result["device"] = {
            "index": device_index,
            "name": device_name,
            "total_memory_bytes": total_memory,
            "compute_capability": list(capability),
            "compiled_arch_list": list(cuda.get_arch_list()),
        }
        checks["expected_gpu_detected"] = not expected_gpu or device_name == expected_gpu
        checks["gpu_memory_positive"] = total_memory > 0
        if expected_gpu and device_name != expected_gpu:
            raise RuntimeError(f"expected GPU {expected_gpu!r}, detected {device_name!r}")

        bf16_supported = bool(cuda.is_bf16_supported())
        checks["bf16_supported"] = bf16_supported
        if not bf16_supported:
            raise RuntimeError("torch.cuda.is_bf16_supported() returned False")

        torch.manual_seed(seed)
        cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        checks["deterministic_algorithms_enabled"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )

        cuda.empty_cache()
        cuda.reset_peak_memory_stats(device_index)

        bf16_tensor: Any = torch.ones(
            (256, 256),
            device="cuda",
            dtype=torch.bfloat16,
        )
        checks["bf16_cuda_allocation"] = bf16_tensor.is_cuda and bf16_tensor.dtype == torch.bfloat16

        left: Any = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
        right: Any = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
        product: Any = left @ right
        cuda.synchronize()
        matmul_finite = bool(torch.isfinite(product).all().item())
        checks["bf16_cuda_matmul"] = matmul_finite
        result["bf16_matmul_checksum"] = float(product.float().sum().item())
        if not matmul_finite:
            raise RuntimeError("CUDA BF16 matrix multiplication produced non-finite values")

        torch.manual_seed(seed + 1)
        cuda.manual_seed_all(seed + 1)
        input_tensor: Any = torch.randn((128, 128), device="cuda", requires_grad=True)
        weight: Any = torch.randn((128, 64), device="cuda", requires_grad=True)
        output: Any = torch.relu(input_tensor @ weight)
        loss: Any = output.square().mean()
        loss.backward()
        cuda.synchronize()
        gradients_finite = bool(
            input_tensor.grad is not None
            and weight.grad is not None
            and torch.isfinite(input_tensor.grad).all().item()
            and torch.isfinite(weight.grad).all().item()
        )
        checks["cuda_forward_backward_autograd"] = gradients_finite
        result["autograd"] = {
            "loss": float(loss.item()),
            "input_grad_norm": float(input_tensor.grad.norm().item()),
            "weight_grad_norm": float(weight.grad.norm().item()),
        }
        if not gradients_finite:
            raise RuntimeError("CUDA autograd produced missing or non-finite gradients")

        torch.manual_seed(seed + 2)
        cuda.manual_seed_all(seed + 2)
        seeded_first: Any = torch.rand((64,), device="cuda")
        torch.manual_seed(seed + 2)
        cuda.manual_seed_all(seed + 2)
        seeded_second: Any = torch.rand((64,), device="cuda")
        seed_replay_equal = bool(torch.equal(seeded_first, seeded_second))
        checks["deterministic_seed_replay"] = seed_replay_equal
        if not seed_replay_equal:
            raise RuntimeError("CUDA random sequence did not replay after resetting the seed")

        cuda.synchronize()
        peak_memory = int(cuda.max_memory_allocated(device_index))
        result["peak_allocated_gpu_memory_bytes"] = peak_memory
        checks["peak_memory_recorded"] = peak_memory > 0
        if peak_memory <= 0:
            raise RuntimeError("CUDA peak allocated memory was not recorded")

        required_checks = (
            "cuda_available",
            "expected_gpu_detected",
            "gpu_memory_positive",
            "bf16_supported",
            "bf16_cuda_allocation",
            "bf16_cuda_matmul",
            "cuda_forward_backward_autograd",
            "deterministic_seed_replay",
            "peak_memory_recorded",
        )
        if not all(checks.get(name) is True for name in required_checks):
            raise RuntimeError("one or more required GPU checks did not pass")
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--expected-gpu", default=DEFAULT_EXPECTED_GPU)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m0_5/gpu_smoke.json"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    result = run_gpu_smoke(seed=args.seed, expected_gpu=args.expected_gpu)
    _write_result(result, args.output)
    print(
        f"GPU smoke status={result['status']} output={args.output.resolve()} "
        f"error={result['error']}"
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
