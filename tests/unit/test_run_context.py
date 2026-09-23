"""Run reproducibility metadata tests."""

from datetime import UTC, datetime
from pathlib import Path

from cutagent.core.artifacts import ArtifactRef
from cutagent.core.run_context import RunContext, config_sha256


def test_config_hash_is_order_independent() -> None:
    assert config_sha256({"alpha": 1, "beta": [2, 3]}) == config_sha256(
        {"beta": [2, 3], "alpha": 1}
    )


def test_run_manifest_records_reproducibility_metadata(
    tmp_path: Path,
) -> None:
    environment_path = tmp_path / "environment.json"
    environment_path.write_text('{"schema_version":"1.0"}\n', encoding="utf-8")
    environment_ref = ArtifactRef.from_path(
        environment_path,
        artifact_id="environment-test",
        media_type="application/json",
    )
    context = RunContext.create(
        repository_root=Path.cwd(),
        seed=123,
        config={"b": 2, "a": 1},
        environment_manifest_ref=environment_ref,
        run_id="run-test",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    manifest = context.create_manifest(
        model_versions={"policy": "metadata-only"},
        dataset_versions={"tasks": "metadata-only"},
    )

    assert manifest.seed == 123
    assert len(manifest.config_sha256) == 64
    assert manifest.code_version.identifier
    assert manifest.environment_manifest_ref == environment_ref
    assert manifest.model_versions == {"policy": "metadata-only"}
    assert manifest.dataset_versions == {"tasks": "metadata-only"}
