"""Create or validate human adjudication for genuine rater disagreements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import cast

try:
    from scripts.m5a_human_gate_lib import (
        FAILURE_LABELS,
        TASK_SUCCESS_LABELS,
        agreement_payload,
        validate_pair,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from m5a_human_gate_lib import (  # type: ignore[no-redef,import-not-found]
        FAILURE_LABELS,
        TASK_SUCCESS_LABELS,
        agreement_payload,
        validate_pair,
    )

ADJUDICATION_FIELDS = (
    "adjudicator_id",
    "packet_version",
    "packet_sha256",
    "case_id",
    "adjudicated_task_success",
    "adjudicated_semantic_constraint_score",
    "adjudicated_failure_type",
    "rationale",
)


def _write_template(path: Path, packet_version: str, packet_hash: str, cases: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ADJUDICATION_FIELDS, lineterminator="\n")
        writer.writeheader()
        for case_id in cases:
            writer.writerow(
                {
                    "adjudicator_id": "",
                    "packet_version": packet_version,
                    "packet_sha256": packet_hash,
                    "case_id": case_id,
                    "adjudicated_task_success": "",
                    "adjudicated_semantic_constraint_score": "",
                    "adjudicated_failure_type": "",
                    "rationale": "",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--rater-1", type=Path, required=True)
    parser.add_argument("--rater-2", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    packet, first, second = validate_pair(
        args.packet.resolve(strict=True),
        args.rater_1.resolve(strict=True),
        args.rater_2.resolve(strict=True),
    )
    agreement = agreement_payload(packet, first, second)
    disagreements = cast(list[str], agreement["disagreement_case_ids"])
    if args.prepare:
        _write_template(args.adjudication, packet.version, packet.sha256, disagreements)
        print(json.dumps({"status": "template_prepared", "disagreement_count": len(disagreements)}))
        return 0
    with args.adjudication.resolve(strict=True).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ADJUDICATION_FIELDS:
            raise ValueError("invalid adjudication columns")
        rows = tuple(dict(row) for row in reader)
    if tuple(row["case_id"] for row in rows) != tuple(disagreements):
        raise ValueError("adjudication must cover every disagreement exactly once")
    adjudicator_ids = {row["adjudicator_id"].strip() for row in rows}
    if disagreements and (len(adjudicator_ids) != 1 or not next(iter(adjudicator_ids))):
        raise ValueError("a real adjudicator identity is required")
    for row in rows:
        if row["packet_version"] != packet.version or row["packet_sha256"] != packet.sha256:
            raise ValueError("adjudication packet identity mismatch")
        if row["adjudicated_task_success"].casefold() not in TASK_SUCCESS_LABELS:
            raise ValueError("invalid adjudicated task-success label")
        if row["adjudicated_semantic_constraint_score"] not in {"0", "1", "2", "3", "4"}:
            raise ValueError("invalid adjudicated semantic score")
        if row["adjudicated_failure_type"].casefold() not in FAILURE_LABELS:
            raise ValueError("invalid adjudicated failure type")
        if not row["rationale"].strip():
            raise ValueError("adjudication rationale is required")
    result = {
        "schema_version": "1.0",
        "status": "adjudicated",
        "packet_version": packet.version,
        "packet_sha256": packet.sha256,
        "disagreement_count": len(disagreements),
        "adjudicator_ids": sorted(adjudicator_ids),
        "decisions": list(rows),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
