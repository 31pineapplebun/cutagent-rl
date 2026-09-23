"""Licensed real-media manifest boundary and provenance contracts."""

from pathlib import Path

import pytest
from cutagent_evaluation.licensed_media import (
    LicensedClipSpec,
    LicensedMediaManifest,
)


def test_frozen_manifest_has_verified_target_shape() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    manifest = LicensedMediaManifest.model_validate_json(
        (repository_root / "evaluation/data/m1b_5_licensed_sources.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(manifest.sources) == 5
    assert sum(len(source.clips) for source in manifest.sources) == 18
    assert all(
        source.source_page.startswith("https://commons.wikimedia.org/")
        for source in manifest.sources
    )
    assert all(
        source.license_url.startswith("https://creativecommons.org/") for source in manifest.sources
    )
    assert all(source.license_evidence for source in manifest.sources)


def test_clip_contract_rejects_empty_half_open_interval() -> None:
    with pytest.raises(ValueError, match="start_ms < end_ms"):
        LicensedClipSpec(clip_id="invalid", start_ms=1000, end_ms=1000)
