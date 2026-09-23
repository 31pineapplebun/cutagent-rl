"""Prepare blank human-rating materials without generating any ratings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.m5a_human_gate_lib import RATING_FIELDS, load_packet, template_rows, write_csv
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from m5a_human_gate_lib import (  # type: ignore[no-redef,import-not-found]
        RATING_FIELDS,
        load_packet,
        template_rows,
        write_csv,
    )


def _guide(packet_version: str, packet_sha256: str, case_count: int) -> str:
    return f"""# CutAgentBench v0.1 Human Calibration Rater Guide

Packet version: `{packet_version}`

Packet SHA-256: `{packet_sha256}`

Required cases: {case_count}

## Independence and identity

Complete the packet independently. Do not inspect another rater's answers, model identity,
aggregate results, private benchmark Gold, or protected-test outputs. `rater_id` must identify
the actual human reviewer. An LLM, VLM, automated evaluator, copied submission, or fictional
identity cannot satisfy this gate.

## Fields

- `task_success`: `success`, `failure`, or `uncertain` based only on the presented case.
- `semantic_constraint_score`: integer 0-4, where 0 means no semantic requirement is met and
  4 means all presented semantic requirements are met.
- `failure_type`: one frozen benchmark failure label, `none`, or `uncertain`.
- `confidence`: integer 1-5.
- `notes`: concise evidence or uncertainty; do not paste secrets or filesystem paths.

Do not edit `packet_version`, `packet_sha256`, or `case_id`. Fill every row and return the CSV
under `artifacts/m5a/human_calibration/submitted/` using the documented filename.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=Path("artifacts/m5a/cutagentbench_v0.1/calibration/human_calibration_packet.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/m5a/human_calibration"))
    args = parser.parse_args()
    packet = load_packet(args.packet.resolve(strict=True))
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "submitted").mkdir(exist_ok=True)
    (root / "final").mkdir(exist_ok=True)
    (root / "RATER_GUIDE.md").write_text(
        _guide(packet.version, packet.sha256, len(packet.case_ids)), encoding="utf-8"
    )
    rows = template_rows(packet)
    write_csv(root / "rater_1_template.csv", rows)
    write_csv(root / "rater_2_template.csv", rows)
    adjudication_fields = (
        "adjudicator_id",
        "packet_version",
        "packet_sha256",
        "case_id",
        "adjudicated_task_success",
        "adjudicated_semantic_constraint_score",
        "adjudicated_failure_type",
        "rationale",
    )
    (root / "adjudication_template.csv").write_text(
        ",".join(adjudication_fields) + "\n", encoding="utf-8"
    )
    result = {
        "status": "prepared_blank_templates_only",
        "packet_version": packet.version,
        "packet_sha256": packet.sha256,
        "case_count": len(packet.case_ids),
        "rating_columns": list(RATING_FIELDS),
        "human_ratings_generated": 0,
    }
    (root / "preparation_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
