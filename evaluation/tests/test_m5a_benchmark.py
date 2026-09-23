from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cutagent_evaluation.m5a_benchmark import (
    ProtectedGoldAccessError,
    ProtectedGoldStore,
    audit_benchmark,
    benchmark_health,
    licensed_qualitative_registry,
    scan_runtime_for_benchmark_task_rules,
    write_benchmark,
)
from cutagent_evaluation.m5a_calibration import handcrafted_facts, run_handcrafted_calibration
from cutagent_evaluation.m5a_dataset import (
    M5A_GENERATOR_VERSION,
    SCENE_DURATION_MS,
    build_benchmark_cases,
    build_source_blueprints,
)
from cutagent_evaluation.m5a_evaluator import evaluate_facts
from cutagent_evaluation.m5a_schemas import (
    BenchmarkSourceRecord,
    GeneratedSceneAnnotation,
    LockedTestAccessRecord,
    MediaFamily,
    TaskFamily,
)
from cutagent_evaluation.schemas import DatasetSplit

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.media import TimeRange


def _fixture_sources(root: Path) -> tuple[BenchmarkSourceRecord, ...]:
    root.mkdir(parents=True, exist_ok=True)
    sources: list[BenchmarkSourceRecord] = []
    for blueprint in build_source_blueprints():
        path = root / f"{blueprint.source_group_id}.mp4"
        path.write_bytes(
            json.dumps(blueprint.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        )
        artifact = ArtifactRef.from_path(
            path,
            artifact_id=f"cab-video-{blueprint.global_index:03d}",
            media_type="video/mp4",
        )
        scenes = tuple(
            GeneratedSceneAnnotation(
                scene_id=f"{blueprint.source_group_id}-scene-{index + 1}",
                time_range=TimeRange(
                    start_ms=index * SCENE_DURATION_MS,
                    end_ms=(index + 1) * SCENE_DURATION_MS,
                ),
                entities=(blueprint.primary_entity, blueprint.companion_entities[index]),
                action=blueprint.actions[index],
                visible_text=blueprint.visible_texts[index],
                transcript=blueprint.transcripts[index],
            )
            for index in range(5)
        )
        structural = hashlib.sha256(
            f"{blueprint.source_group_id}|{blueprint.pattern_variant}".encode()
        ).hexdigest()
        sources.append(
            BenchmarkSourceRecord(
                source_group_id=blueprint.source_group_id,
                split=blueprint.split,
                media_family=MediaFamily.GENERATED,
                video_id=artifact.artifact_id,
                source_artifact=artifact,
                source_sha256=artifact.sha256,
                structural_fingerprint=structural,
                generator_version=M5A_GENERATOR_VERSION,
                provenance_reference="deterministic contract fixture",
                license="CC0-1.0",
                scenes=scenes,
            )
        )
    return tuple(sources)


@pytest.fixture
def benchmark_cases(tmp_path: Path):  # type: ignore[no-untyped-def]
    sources = _fixture_sources(tmp_path / "sources")
    cases = build_benchmark_cases(sources)
    return sources, cases


def test_frozen_scale_and_split_distribution(benchmark_cases) -> None:  # type: ignore[no-untyped-def]
    sources, cases = benchmark_cases
    assert len(sources) == 44
    assert len(cases) == 220
    assert {
        split: sum(case.private_gold.split == split for case in cases) for split in DatasetSplit
    } == {
        DatasetSplit.TRAIN: 40,
        DatasetSplit.DEV: 40,
        DatasetSplit.VALIDATION: 50,
        DatasetSplit.LOCKED_TEST: 60,
        DatasetSplit.ADVERSARIAL_TEST: 30,
    }
    assert {case.private_gold.task_family for case in cases} == set(TaskFamily)
    assert len({case.public_task.task_id for case in cases}) == 220
    assert len({source.source_group_id for source in sources}) == 44


def test_generated_speech_is_ffmpeg_flite_safe() -> None:
    safe = re.compile(r"^[A-Za-z0-9 ]+$")
    assert all(
        safe.fullmatch(transcript)
        for blueprint in build_source_blueprints()
        for transcript in blueprint.transcripts
    )


def test_leakage_audit_and_policy_projection_pass(benchmark_cases) -> None:  # type: ignore[no-untyped-def]
    sources, cases = benchmark_cases
    result = audit_benchmark(sources, cases)
    assert result.passed
    assert result.cross_split_source_groups == ()
    assert result.cross_split_content_hashes == ()
    assert result.policy_context_leaks == ()
    assert result.task_id_prompt_leaks == ()


def test_cross_split_content_and_derivative_leakage_is_detected(benchmark_cases) -> None:  # type: ignore[no-untyped-def]
    sources, cases = benchmark_cases
    cross_index = 16
    copied = sources[cross_index].model_copy(
        update={
            "source_sha256": sources[0].source_sha256,
            "source_artifact": sources[cross_index].source_artifact.model_copy(
                update={"sha256": sources[0].source_sha256}
            ),
            "derivative_of_source_group_id": sources[0].source_group_id,
        }
    )
    changed_sources = (*sources[:cross_index], copied, *sources[cross_index + 1 :])
    result = audit_benchmark(changed_sources, cases)
    assert not result.passed
    assert sources[0].source_sha256 in result.cross_split_content_hashes
    assert copied.source_group_id in result.cross_split_derivative_violations


def test_protected_gold_requires_complete_logged_access(benchmark_cases) -> None:  # type: ignore[no-untyped-def]
    _, cases = benchmark_cases
    gold_by_id = {case.public_task.task_id: case.private_gold for case in cases}
    store = ProtectedGoldStore(gold_by_id)
    locked = next(case for case in cases if case.private_gold.split == DatasetSplit.LOCKED_TEST)
    with pytest.raises(ProtectedGoldAccessError):
        store.get(locked.public_task.task_id)
    access = LockedTestAccessRecord(
        access_id="locked-access-test",
        split=DatasetSplit.LOCKED_TEST,
        git_commit="8172841",
        model_revision="frozen-prompt-only-revision",
        adapter_or_checkpoint_hash="none-prompt-only",
        config_sha256="1" * 64,
        access_reason="unit-test access-policy validation; no model execution",
        declared_metrics=("none",),
        accessed_at=datetime.now(UTC),
    )
    assert store.get(locked.public_task.task_id, access=access) == locked.private_gold
    assert store.access_log == (access,)
    dev = next(case for case in cases if case.private_gold.split == DatasetSplit.DEV)
    assert store.get(dev.public_task.task_id) == dev.private_gold


def test_manifest_freeze_is_deterministic_and_seals_protected_gold(
    tmp_path: Path, benchmark_cases: Any
) -> None:
    sources, cases = benchmark_cases
    frozen_at = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)
    first = write_benchmark(tmp_path / "first", sources, cases, frozen_at=frozen_at)
    second = write_benchmark(tmp_path / "second", sources, cases, frozen_at=frozen_at)
    assert first.manifest == second.manifest
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest.task_count == 220
    assert first.manifest.locked_test_seal_sha256
    public_locked = json.loads(
        (tmp_path / "first/public/locked_test.json").read_text(encoding="utf-8")
    )
    serialized = json.dumps(public_locked).casefold()
    for forbidden in ("source_group_id", '"split"', "difficulty", "failure_injection"):
        assert forbidden not in serialized


def test_benchmark_health_and_sanity_baselines(benchmark_cases) -> None:  # type: ignore[no-untyped-def]
    sources, cases = benchmark_cases
    health = benchmark_health(sources, cases)
    assert health.task_count == 220
    assert health.sanity_baselines["unchanged_source_strict_tsr"] == 0
    assert 0 < health.impossible_proportion < 0.2
    assert 0 < health.recovery_proportion < 0.2
    assert health.sanity_baselines["oracle_objective_evaluator_control"] == 1


def test_licensed_registry_preserves_provenance_without_host_commands(tmp_path: Path) -> None:
    provenance = {
        "sources": [
            {
                "source_id": "commons-example",
                "title": "Explicitly licensed example",
                "source_page": "https://commons.wikimedia.org/example",
                "license": "CC BY 4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "attribution": "Example author",
                "sha256": "1" * 64,
                "license_page_sha256": "2" * 64,
                "download_date_utc": "2026-08-22T00:00:00Z",
                "relative_path": "sources/example.webm",
                "clips": [
                    {
                        "clip_id": "example-01",
                        "start_ms": 0,
                        "end_ms": 3000,
                        "sha256": "3" * 64,
                        "size_bytes": 10,
                        "extraction_command": ["ffmpeg", "/tmp/fixture/private/example"],
                    }
                ],
            }
        ]
    }
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(provenance), encoding="utf-8")
    registry = licensed_qualitative_registry(path)
    serialized = registry.model_dump_json()
    assert registry.source_count == 1
    assert registry.clip_count == 1
    assert registry.quantitative_task_count == 0
    assert "extraction_command" not in serialized
    assert "/tmp/fixture/private" not in serialized


def test_handcrafted_calibration_uses_only_dev_and_validation(benchmark_cases) -> None:  # type: ignore[no-untyped-def]
    _, cases = benchmark_cases
    result = run_handcrafted_calibration(cases)
    selected = {item.task_id for item in result.results}
    split_by_task = {case.public_task.task_id: case.private_gold.split for case in cases}
    assert result.case_count == 50
    assert result.exact_tsr_agreement == 1
    assert {split_by_task[item] for item in selected} <= {
        DatasetSplit.DEV,
        DatasetSplit.VALIDATION,
    }
    assert result.human_rater_count == 0
    assert result.inter_rater_agreement is None


def test_runtime_has_no_task_specific_benchmark_rules(benchmark_cases) -> None:  # type: ignore[no-untyped-def]
    _, cases = benchmark_cases
    project_root = Path(__file__).resolve().parents[2]
    assert (
        scan_runtime_for_benchmark_task_rules(
            project_root, [case.public_task.task_id for case in cases]
        )
        == ()
    )


def test_handcrafted_evaluator_cases_have_exact_expected_outcomes(benchmark_cases) -> None:  # type: ignore[no-untyped-def]
    _, cases = benchmark_cases
    possible = [
        case
        for case in cases
        if case.private_gold.task_family
        not in {TaskFamily.IMPOSSIBLE, TaskFamily.OBSERVABLE_RECOVERY}
    ]
    impossible = next(
        case for case in cases if case.private_gold.task_family == TaskFamily.IMPOSSIBLE
    )
    recovery = next(
        case for case in cases if case.private_gold.task_family == TaskFamily.OBSERVABLE_RECOVERY
    )
    subtitle = next(
        case
        for case in possible
        if any(
            item.constraint_type == "subtitle" for item in case.private_gold.objective_constraints
        )
    )
    reframe = next(
        case
        for case in possible
        if any(
            item.constraint_type == "resolution" for item in case.private_gold.objective_constraints
        )
    )
    composition = next(
        case
        for case in possible
        if any(
            item.constraint_type == "scene_order"
            for item in case.private_gold.objective_constraints
        )
    )
    duration = next(
        case
        for case in possible
        if any(
            item.constraint_type == "duration" for item in case.private_gold.objective_constraints
        )
    )
    checks = (
        (possible[0], "exact_success", True),
        (possible[0], "wrong_scene", False),
        (duration, "wrong_duration", False),
        (reframe, "wrong_aspect_ratio", False),
        (subtitle, "wrong_subtitle", False),
        (composition, "wrong_order", False),
        (possible[0], "partial_constraints", False),
        (impossible, "correct_refusal", True),
        (possible[0], "false_refusal", False),
        (impossible, "premature_finish", False),
        (possible[0], "loop", False),
        (possible[0], "invalid_arguments", False),
        (recovery, "recovery_success", True),
        (recovery, "recovery_failure", False),
    )
    for case, mutation, expected in checks:
        evaluation = evaluate_facts(
            handcrafted_facts(case, mutation),  # type: ignore[arg-type]
            case.private_gold,
        )
        assert evaluation.task_success is expected, mutation
        if mutation == "invalid_arguments":
            assert evaluation.primary_failure is not None
            assert evaluation.primary_failure.value == "invalid_arguments"
        if mutation == "wrong_scene":
            assert evaluation.primary_failure is not None
            assert evaluation.primary_failure.value == "retrieval_error"
        if mutation == "wrong_duration":
            assert evaluation.primary_failure is not None
            assert evaluation.primary_failure.value == "invalid_arguments"
    partial = evaluate_facts(
        handcrafted_facts(possible[0], "partial_constraints"), possible[0].private_gold
    )
    assert 0 < partial.hard_constraint_satisfaction < 1
