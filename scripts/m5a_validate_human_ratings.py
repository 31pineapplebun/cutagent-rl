"""Validate two independent real-human CutAgentBench calibration submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.m5a_human_gate_lib import validate_pair
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from m5a_human_gate_lib import validate_pair  # type: ignore[no-redef,import-not-found]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--rater-1", type=Path, required=True)
    parser.add_argument("--rater-2", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    packet, first, second = validate_pair(
        args.packet.resolve(strict=True),
        args.rater_1.resolve(strict=True),
        args.rater_2.resolve(strict=True),
    )
    result = {
        "schema_version": "1.0",
        "status": "passed",
        "packet_version": packet.version,
        "packet_sha256": packet.sha256,
        "case_count": len(packet.case_ids),
        "distinct_real_human_rater_count": 2,
        "rater_ids": [first.rater_id, second.rater_id],
        "rater_file_sha256": [first.sha256, second.sha256],
        "protected_agent_results_used": False,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
