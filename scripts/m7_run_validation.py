"""Run the composed M6 SFT + M7 DPO policy through the frozen validation harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import scripts.m5b_run_baseline as frozen_runner
from cutagent.models.qwen_policy_m7 import Qwen3VLDPOPolicyBackend


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sft-adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--dpo-adapter-checkpoint", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    sft_adapter = known.sft_adapter_checkpoint.resolve(strict=True)
    dpo_adapter = known.dpo_adapter_checkpoint.resolve(strict=True)

    def factory(*, model_cache: Any) -> Qwen3VLDPOPolicyBackend:
        return Qwen3VLDPOPolicyBackend(
            model_cache=model_cache,
            sft_adapter_checkpoint=sft_adapter,
            dpo_adapter_checkpoint=dpo_adapter,
        )

    frozen_runner.Qwen3VLPolicyBackendM4B = factory  # type: ignore[attr-defined,assignment]
    sys.argv = [sys.argv[0], *remaining]
    return frozen_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
