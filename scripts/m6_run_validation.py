"""Run the M6 SFT adapter through the frozen M5B validation harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import scripts.m5b_run_baseline as frozen_runner
from cutagent.models.qwen_policy_m6 import Qwen3VLSFTPolicyBackend


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    adapter = known.adapter_checkpoint.resolve(strict=True)

    def factory(*, model_cache: Any) -> Qwen3VLSFTPolicyBackend:
        return Qwen3VLSFTPolicyBackend(
            model_cache=model_cache,
            adapter_checkpoint=adapter,
        )

    frozen_runner.Qwen3VLPolicyBackendM4B = factory  # type: ignore[attr-defined,assignment]
    sys.argv = [sys.argv[0], *remaining]
    return frozen_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
