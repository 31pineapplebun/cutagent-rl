from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.task_input import TaskInput
from scripts import m5b_prepare_public_media as preparation_cli
from scripts.m10_protected_preparation import (
    ProtectedPreparationAuthorization,
    authorization_sha256,
    file_sha256,
    issue_protected_preparation_authorization,
    validate_preparation_request,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, TaskInput]:
    benchmark = tmp_path / "benchmark"
    source = tmp_path / "public-source.mp4"
    source.write_bytes(b"synthetic-public-media-fixture")
    artifact = ArtifactRef.from_path(
        source,
        artifact_id="synthetic-public-video",
        media_type="video/mp4",
    )
    task = TaskInput(
        task_id="synthetic-public-task",
        video_ref=artifact,
        instruction="Export the observable scene.",
    )
    for split in ("dev", "validation", "locked_test", "adversarial_test"):
        _write_json(benchmark / "public" / f"{split}.json", [task.model_dump(mode="json")])
    benchmark_hash = "a" * 64
    (benchmark / "benchmark_manifest.sha256").write_text(
        f"{benchmark_hash}  benchmark_manifest.json\n", encoding="utf-8"
    )
    (benchmark / "benchmark_manifest.json").write_text("{}\n", encoding="utf-8")
    gate = tmp_path / "human_gate.json"
    _write_json(gate, {"status": "PASSED"})
    plan = tmp_path / "protected_plan.json"
    _write_json(
        plan,
        {
            "benchmark": {
                "version": "cutagentbench-v0.1",
                "manifest_sha256": benchmark_hash,
            },
            "human_gate_acceptance_sha256": file_sha256(gate),
            "declared_splits": ["locked_test", "adversarial_test"],
        },
    )
    return benchmark, gate, plan, source, task


def _authorization_file(
    tmp_path: Path,
    *,
    split: str = "locked_test",
    issued_at: datetime | None = None,
) -> tuple[Path, ProtectedPreparationAuthorization, Path, Path, Path, TaskInput]:
    benchmark, gate, plan, _source, task = _fixture(tmp_path)
    authorization = issue_protected_preparation_authorization(
        split=split,
        benchmark_root=benchmark,
        human_gate_path=gate,
        protected_plan_path=plan,
        finalization_run_id="m10c-synthetic-contract",
        issued_at=issued_at,
    )
    path = tmp_path / f"{split}-authorization.json"
    _write_json(path, authorization.model_dump(mode="json"))
    return path, authorization, benchmark, gate, plan, task


def _validate(
    *,
    split: str,
    benchmark: Path,
    gate: Path | None = None,
    plan: Path | None = None,
    authorization: Path | None = None,
    now: datetime | None = None,
) -> ProtectedPreparationAuthorization | None:
    return validate_preparation_request(
        split=split,
        benchmark_root=benchmark,
        authorization_path=authorization,
        human_gate_path=gate,
        protected_plan_path=plan,
        now=now,
    )


def test_development_preparation_remains_unprivileged() -> None:
    for split in ("dev", "validation"):
        assert _validate(split=split, benchmark=Path("unused")) is None


@pytest.mark.parametrize("split", ["locked_test", "adversarial_test"])
def test_direct_protected_cli_without_authorization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    benchmark, _, _, _, _ = _fixture(tmp_path)
    model_cache = tmp_path / "models"
    model_cache.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "m5b_prepare_public_media.py",
            "--split",
            split,
            "--benchmark-root",
            str(benchmark),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--model-cache",
            str(model_cache),
        ],
    )
    with pytest.raises(PermissionError, match="finalizer authorization"):
        preparation_cli.main()


@pytest.mark.parametrize("split", ["locked_test", "adversarial_test"])
def test_valid_synthetic_protected_authorization_succeeds(tmp_path: Path, split: str) -> None:
    auth_path, authorization, benchmark, gate, plan, _ = _authorization_file(tmp_path, split=split)
    assert (
        _validate(
            split=split,
            benchmark=benchmark,
            gate=gate,
            plan=plan,
            authorization=auth_path,
        )
        == authorization
    )
    assert len(authorization_sha256(authorization)) == 64


def test_wrong_benchmark_hash_fails(tmp_path: Path) -> None:
    auth_path, authorization, benchmark, gate, plan, _ = _authorization_file(tmp_path)
    tampered = authorization.model_copy(update={"benchmark_manifest_sha256": "b" * 64})
    _write_json(auth_path, tampered.model_dump(mode="json"))
    with pytest.raises(PermissionError, match="benchmark_manifest_sha256"):
        _validate(
            split="locked_test",
            benchmark=benchmark,
            gate=gate,
            plan=plan,
            authorization=auth_path,
        )


def test_wrong_human_gate_hash_fails(tmp_path: Path) -> None:
    auth_path, _, benchmark, gate, plan, _ = _authorization_file(tmp_path)
    _write_json(gate, {"status": "PASSED", "changed": True})
    with pytest.raises(ValueError, match="human-gate hash"):
        _validate(
            split="locked_test",
            benchmark=benchmark,
            gate=gate,
            plan=plan,
            authorization=auth_path,
        )


def test_undeclared_split_fails(tmp_path: Path) -> None:
    benchmark, _, _, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="not authorized"):
        _validate(split="train", benchmark=benchmark)


def test_expired_and_malformed_authorization_fail(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    auth_path, _, benchmark, gate, plan, _ = _authorization_file(
        tmp_path, issued_at=now - timedelta(days=2)
    )
    with pytest.raises(PermissionError, match="expired"):
        _validate(
            split="locked_test",
            benchmark=benchmark,
            gate=gate,
            plan=plan,
            authorization=auth_path,
            now=now,
        )
    _write_json(auth_path, {"schema_version": "1.0", "unexpected": "field"})
    with pytest.raises(ValidationError):
        _validate(
            split="locked_test",
            benchmark=benchmark,
            gate=gate,
            plan=plan,
            authorization=auth_path,
        )


def test_public_preparation_never_reads_private_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_path, _, benchmark, gate, plan, _ = _authorization_file(tmp_path)
    observed: list[Path] = []
    original = Path.read_text

    def audited_read(path: Path, *args: object, **kwargs: object) -> str:
        observed.append(path)
        return original(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", audited_read)
    _validate(
        split="locked_test",
        benchmark=benchmark,
        gate=gate,
        plan=plan,
        authorization=auth_path,
    )
    tasks = preparation_cli._load_public_tasks(benchmark, "locked_test")
    assert len(tasks) == 1
    assert all("private" not in path.parts and "sealed" not in path.parts for path in observed)


def test_public_task_and_media_identity_are_unchanged(tmp_path: Path) -> None:
    auth_path, _, benchmark, gate, plan, expected_task = _authorization_file(tmp_path)
    before = {
        path.relative_to(benchmark): file_sha256(path)
        for path in benchmark.rglob("*")
        if path.is_file()
    }
    _validate(
        split="locked_test",
        benchmark=benchmark,
        gate=gate,
        plan=plan,
        authorization=auth_path,
    )
    task = preparation_cli._load_public_tasks(benchmark, "locked_test")[0]
    after = {
        path.relative_to(benchmark): file_sha256(path)
        for path in benchmark.rglob("*")
        if path.is_file()
    }
    assert task == expected_task
    assert before == after
    source = preparation_cli._artifact_path(task.video_ref.uri)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == task.video_ref.sha256
    visible = task.model_dump(mode="json")
    assert "split" not in visible
    assert "source_group_id" not in visible
    serialized = task.model_dump_json().casefold()
    assert "benchmarkgold" not in serialized
    assert "expected_scene" not in serialized


def test_authorization_schema_forbids_gold_and_extra_fields(tmp_path: Path) -> None:
    _, authorization, _, _, _, _ = _authorization_file(tmp_path)
    payload = authorization.model_dump(mode="json")
    payload["expected_scene"] = "private-answer"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProtectedPreparationAuthorization.model_validate(payload)


def test_preparation_direct_script_import_path_is_supported() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/m5b_prepare_public_media.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--protected-authorization" in result.stdout
