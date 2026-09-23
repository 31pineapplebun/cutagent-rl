"""Compute agreement only after two valid real-human submissions exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.m5a_human_gate_lib import agreement_payload, validate_pair
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from m5a_human_gate_lib import (  # type: ignore[no-redef,import-not-found]
        agreement_payload,
        validate_pair,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--rater-1", type=Path, required=True)
    parser.add_argument("--rater-2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet, first, second = validate_pair(
        args.packet.resolve(strict=True),
        args.rater_1.resolve(strict=True),
        args.rater_2.resolve(strict=True),
    )
    result = agreement_payload(packet, first, second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
