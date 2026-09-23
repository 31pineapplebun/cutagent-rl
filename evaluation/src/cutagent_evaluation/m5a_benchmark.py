"""CutAgentBench v0.1 leakage audit, versioning, sealing, and health reports."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from cutagent.agent.policy_context import PolicyContextBuilder, PolicyViewSerializer
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.state import AgentState, ExecutionBudget
from cutagent_evaluation.m5a_dataset import M5A_GENERATOR_VERSION
from cutagent_evaluation.m5a_schemas import (
    EVALUATOR_VERSION,
    FAILURE_TAXONOMY_VERSION,
    METRIC_DEFINITION_VERSION,
    BenchmarkHealthSummary,
    BenchmarkManifest,
    BenchmarkSourceRecord,
    BenchmarkSplitSummary,
    CutAgentBenchCase,
    CutAgentBenchGold,
    HumanCalibrationPacket,
    LicensedClipProvenance,
    LicensedQualitativeRegistry,
    LicensedQualitativeSource,
    LockedTestAccessRecord,
    SemanticEvaluationProtocol,
    TaskFamily,
)
from cutagent_evaluation.schemas import DatasetSplit

PROTECTED_SPLITS = frozenset(
    {DatasetSplit.VALIDATION, DatasetSplit.LOCKED_TEST, DatasetSplit.ADVERSARIAL_TEST}
)
SEALED_SPLITS = frozenset({DatasetSplit.LOCKED_TEST, DatasetSplit.ADVERSARIAL_TEST})
FORBIDDEN_POLICY_TOKENS = (
    "benchmarkgold",
    "benchmark_gold",
    "source_group_id",
    '"split"',
    "ground_truth",
    "evaluator_metadata",
)


class LeakageAuditResult(SchemaModel):
    audit_version: Literal["m5a-leakage-audit-v1"] = "m5a-leakage-audit-v1"
    passed: bool
    task_count: int = Field(ge=0)
    source_group_count: int = Field(ge=0)
    cross_split_source_groups: tuple[Identifier, ...] = ()
    cross_split_content_hashes: tuple[str, ...] = ()
    cross_split_structural_duplicates: tuple[str, ...] = ()
    cross_split_derivative_violations: tuple[Identifier, ...] = ()
    duplicate_instruction_answer_pairs: tuple[str, ...] = ()
    public_private_field_leaks: tuple[Identifier, ...] = ()
    policy_context_leaks: tuple[Identifier, ...] = ()
    task_id_prompt_leaks: tuple[Identifier, ...] = ()
    methods: tuple[NonEmptyStr, ...]


class BenchmarkWriteResult(SchemaModel):
    root: NonEmptyStr
    manifest: BenchmarkManifest
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit: LeakageAuditResult
    health: BenchmarkHealthSummary
    public_files: tuple[NonEmptyStr, ...]
    private_files: tuple[NonEmptyStr, ...]


class ProtectedGoldAccessError(PermissionError):
    """Raised when sealed Gold is requested without a complete access declaration."""


class ProtectedGoldStore:
    """Evaluator-side capability boundary for protected split labels."""

    def __init__(self, gold: Mapping[str, CutAgentBenchGold]) -> None:
        self._gold = dict(gold)
        self._audit: list[LockedTestAccessRecord] = []

    @property
    def access_log(self) -> tuple[LockedTestAccessRecord, ...]:
        return tuple(self._audit)

    def get(
        self,
        task_id: str,
        *,
        access: LockedTestAccessRecord | None = None,
    ) -> CutAgentBenchGold:
        try:
            gold = self._gold[task_id]
        except KeyError as error:
            raise KeyError(f"unknown benchmark task {task_id!r}") from error
        if gold.split in SEALED_SPLITS:
            if access is None:
                raise ProtectedGoldAccessError(
                    f"{gold.split.value} Gold requires a declared, logged access record"
                )
            if access.split != gold.split:
                raise ProtectedGoldAccessError("access record split differs from requested Gold")
            self._audit.append(access)
        return gold


def canonical_json_bytes(value: object) -> bytes:
    if isinstance(value, SchemaModel):
        payload: Any = value.model_dump(mode="json")
    elif isinstance(value, tuple):
        payload = [
            item.model_dump(mode="json") if isinstance(item, SchemaModel) else item
            for item in value
        ]
    else:
        payload = value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _cross_split_values(
    sources: Sequence[BenchmarkSourceRecord],
    attribute: str,
) -> tuple[str, ...]:
    splits: dict[str, set[DatasetSplit]] = defaultdict(set)
    for source in sources:
        splits[str(getattr(source, attribute))].add(source.split)
    return tuple(sorted(value for value, observed in splits.items() if len(observed) > 1))


def _policy_context_leaks(cases: Sequence[CutAgentBenchCase]) -> tuple[str, ...]:
    leaks: list[str] = []
    budget = ExecutionBudget(
        max_steps=12,
        max_tool_calls=10,
        max_search_calls=3,
        max_edit_calls=8,
        max_wall_time_ms=300_000,
    )
    for case in cases:
        state = AgentState.initial(case.public_task, budget)
        serialized = PolicyViewSerializer.to_json(PolicyContextBuilder().build(state)).casefold()
        if any(token in serialized for token in FORBIDDEN_POLICY_TOKENS):
            leaks.append(case.public_task.task_id)
        if case.private_gold.source_group_id.casefold() in serialized:
            leaks.append(case.public_task.task_id)
    return tuple(sorted(set(leaks)))


def audit_benchmark(
    sources: Sequence[BenchmarkSourceRecord],
    cases: Sequence[CutAgentBenchCase],
) -> LeakageAuditResult:
    """Audit source grouping, content/structural duplication, and policy isolation."""

    group_splits: dict[str, set[DatasetSplit]] = defaultdict(set)
    by_group = {source.source_group_id: source for source in sources}
    for source in sources:
        group_splits[source.source_group_id].add(source.split)
    for case in cases:
        group_splits[case.private_gold.source_group_id].add(case.private_gold.split)
    cross_groups = tuple(sorted(group for group, splits in group_splits.items() if len(splits) > 1))
    derivatives: list[str] = []
    for source in sources:
        parent = source.derivative_of_source_group_id
        if parent is None:
            continue
        parent_source = by_group.get(parent)
        if parent_source is None or parent_source.split != source.split:
            derivatives.append(source.source_group_id)
    pair_splits: dict[str, set[DatasetSplit]] = defaultdict(set)
    for case in cases:
        gold_answer = {
            "video": case.private_gold.relevant_video_ids,
            "scenes": case.private_gold.relevant_scene_ids,
            "ranges": [
                item.model_dump(mode="json") for item in case.private_gold.acceptable_time_ranges
            ],
            "terminal": case.private_gold.expected_terminal_behavior.value,
        }
        key = sha256_json({"instruction": case.public_task.instruction, "answer": gold_answer})
        pair_splits[key].add(case.private_gold.split)
    duplicate_pairs = tuple(sorted(key for key, splits in pair_splits.items() if len(splits) > 1))
    forbidden_public = {
        "split",
        "source_group_id",
        "difficulty",
        "ground_truth",
        "benchmark_gold",
        "evaluator_metadata",
        "failure_injection",
    }
    public_leaks: list[str] = []
    task_prompt_leaks: list[str] = []
    for case in cases:
        public_keys = set(case.public_task.model_dump(mode="json"))
        if public_keys & forbidden_public:
            public_leaks.append(case.public_task.task_id)
        instruction = case.public_task.instruction.casefold()
        if case.public_task.task_id.casefold() in instruction:
            task_prompt_leaks.append(case.public_task.task_id)
    policy_leaks = _policy_context_leaks(cases)
    content_duplicates = _cross_split_values(sources, "source_sha256")
    structural_duplicates = _cross_split_values(sources, "structural_fingerprint")
    passed = not any(
        (
            cross_groups,
            content_duplicates,
            structural_duplicates,
            derivatives,
            duplicate_pairs,
            public_leaks,
            policy_leaks,
            task_prompt_leaks,
        )
    )
    return LeakageAuditResult(
        passed=passed,
        task_count=len(cases),
        source_group_count=len(group_splits),
        cross_split_source_groups=cross_groups,
        cross_split_content_hashes=content_duplicates,
        cross_split_structural_duplicates=structural_duplicates,
        cross_split_derivative_violations=tuple(sorted(derivatives)),
        duplicate_instruction_answer_pairs=duplicate_pairs,
        public_private_field_leaks=tuple(sorted(public_leaks)),
        policy_context_leaks=policy_leaks,
        task_id_prompt_leaks=tuple(sorted(task_prompt_leaks)),
        methods=(
            "exact source-group split membership",
            "streaming source SHA-256 duplicate detection",
            "generated-media structural fingerprint duplicate detection",
            "derivative-parent split inheritance",
            "exact instruction-plus-answer collision detection",
            "PolicyContextBuilder/PolicyViewSerializer sentinel scan",
        ),
    )


def benchmark_health(
    sources: Sequence[BenchmarkSourceRecord],
    cases: Sequence[CutAgentBenchCase],
) -> BenchmarkHealthSummary:
    split_counts = Counter(case.private_gold.split.value for case in cases)
    family_counts = Counter(case.private_gold.task_family.value for case in cases)
    difficulty_counts = Counter(case.private_gold.difficulty.value for case in cases)
    task_count = len(cases)
    if task_count == 0:
        raise ValueError("benchmark cannot be empty")
    simple_first_scene = 0
    for case in cases:
        gold = case.private_gold
        if gold.expected_terminal_behavior.value != "SUCCESS" or not gold.acceptable_time_ranges:
            continue
        special = {"resolution", "subtitle", "scene_order", "speed"}
        kinds = {item.constraint_type for item in gold.objective_constraints}
        if gold.acceptable_time_ranges[0].start_ms == 0 and not (kinds & special):
            simple_first_scene += 1
    return BenchmarkHealthSummary(
        task_count=task_count,
        source_group_count=len({item.source_group_id for item in sources}),
        split_counts=dict(sorted(split_counts.items())),
        task_family_counts=dict(sorted(family_counts.items())),
        difficulty_counts=dict(sorted(difficulty_counts.items())),
        average_required_tool_count=sum(len(case.private_gold.required_tools) for case in cases)
        / task_count,
        average_mandatory_constraints=sum(
            sum(constraint.mandatory for constraint in case.private_gold.objective_constraints)
            for case in cases
        )
        / task_count,
        hard_negative_proportion=family_counts[TaskFamily.HARD_NEGATIVE.value] / task_count,
        impossible_proportion=family_counts[TaskFamily.IMPOSSIBLE.value] / task_count,
        recovery_proportion=family_counts[TaskFamily.OBSERVABLE_RECOVERY.value] / task_count,
        sanity_baselines={
            "always_refuse_strict_tsr": family_counts[TaskFamily.IMPOSSIBLE.value] / task_count,
            "unchanged_source_strict_tsr": 0.0,
            "always_first_scene_upper_bound_strict_tsr": simple_first_scene / task_count,
            "oracle_objective_evaluator_control": 1.0,
        },
        metadata={
            "keyword_only_ocr_is_not_an_editing_solution": True,
            "all_quantitative_success_requires_objective_outcome_rules": True,
        },
    )


def licensed_qualitative_registry(provenance_path: Path) -> LicensedQualitativeRegistry:
    """Sanitize the frozen M1B.5 licensed-media provenance for M5A review.

    Host paths and raw extraction commands are deliberately excluded.  These clips
    receive no quantitative Gold until genuine manual annotation exists.
    """

    raw = json.loads(provenance_path.read_text(encoding="utf-8"))
    raw_sources = raw.get("sources") if isinstance(raw, dict) else None
    if not isinstance(raw_sources, list):
        raise ValueError("licensed provenance must contain a source list")
    sources: list[LicensedQualitativeSource] = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            raise ValueError("licensed source provenance must be an object")
        raw_clips = raw_source.get("clips")
        if not isinstance(raw_clips, list):
            raise ValueError("licensed source clips must be a list")
        clips = tuple(
            LicensedClipProvenance(
                clip_id=clip["clip_id"],
                start_ms=clip["start_ms"],
                end_ms=clip["end_ms"],
                sha256=clip["sha256"],
                size_bytes=clip["size_bytes"],
            )
            for clip in raw_clips
            if isinstance(clip, dict)
        )
        sources.append(
            LicensedQualitativeSource(
                source_id=raw_source["source_id"],
                title=raw_source["title"],
                source_page=raw_source["source_page"],
                license=raw_source["license"],
                license_url=raw_source["license_url"],
                attribution=raw_source["attribution"],
                source_sha256=raw_source["sha256"],
                license_page_sha256=raw_source["license_page_sha256"],
                download_date_utc=raw_source["download_date_utc"],
                clips=clips,
            )
        )
    return LicensedQualitativeRegistry(
        source_count=len(sources),
        clip_count=sum(len(source.clips) for source in sources),
        exclusion_reason=(
            "Existing licensed clips remain qualitative external-validity material because "
            "M5A has no genuine human semantic Gold for them."
        ),
        sources=tuple(sources),
    )


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    path.write_bytes(payload + b"\n")
    return hashlib.sha256(payload).hexdigest()


def _seal_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Access is still enforced by ProtectedGoldStore; chmod support is platform-specific.
        return


def _evaluator_contract_hash() -> str:
    return sha256_json(
        {
            "evaluator_version": EVALUATOR_VERSION,
            "failure_taxonomy": FAILURE_TAXONOMY_VERSION,
            "metrics": METRIC_DEFINITION_VERSION,
            "tsr": (
                "correct terminal AND valid final artifact when required AND all mandatory "
                "objective constraints AND required tools"
            ),
            "temporal_iou_threshold_for_grounding": 0.9,
        }
    )


def write_benchmark(
    root: Path,
    sources: Sequence[BenchmarkSourceRecord],
    cases: Sequence[CutAgentBenchCase],
    *,
    frozen_at: datetime | None = None,
) -> BenchmarkWriteResult:
    """Persist public tasks, private/sealed Gold, audit evidence, and a frozen manifest."""

    audit = audit_benchmark(sources, cases)
    if not audit.passed:
        raise ValueError("CutAgentBench leakage audit failed; refusing to freeze v0.1")
    health = benchmark_health(sources, cases)
    freeze_time = frozen_at or datetime.now(UTC)
    public_files: list[str] = []
    private_files: list[str] = []
    split_summaries: list[BenchmarkSplitSummary] = []
    locked_hash = ""
    adversarial_hash = ""
    for split in DatasetSplit:
        split_cases = tuple(case for case in cases if case.private_gold.split == split)
        public_path = root / "public" / f"{split.value}.json"
        public_hash = _write_json(public_path, tuple(case.public_task for case in split_cases))
        public_files.append(str(public_path))
        private_directory = "private/sealed" if split in SEALED_SPLITS else "private"
        private_path = root / private_directory / f"{split.value}_gold.json"
        private_hash = _write_json(private_path, tuple(case.private_gold for case in split_cases))
        private_files.append(str(private_path))
        if split in SEALED_SPLITS:
            _seal_file(private_path)
        if split == DatasetSplit.LOCKED_TEST:
            locked_hash = private_hash
            ids_path = root / "private/sealed/locked_test_task_ids.json"
            _write_json(ids_path, tuple(case.public_task.task_id for case in split_cases))
            private_files.append(str(ids_path))
            _seal_file(ids_path)
        elif split == DatasetSplit.ADVERSARIAL_TEST:
            adversarial_hash = private_hash
        split_summaries.append(
            BenchmarkSplitSummary(
                split=split,
                task_count=len(split_cases),
                source_group_count=len({case.private_gold.source_group_id for case in split_cases}),
                public_manifest_sha256=public_hash,
                private_gold_sha256=private_hash,
            )
        )
    source_registry_path = root / "private/source_group_registry.json"
    source_hash = _write_json(source_registry_path, tuple(sources))
    private_files.append(str(source_registry_path))
    task_registry_hash = sha256_json(
        tuple(
            {
                "task_id": case.public_task.task_id,
                "source_group_id": case.private_gold.source_group_id,
                "split": case.private_gold.split.value,
            }
            for case in cases
        )
    )
    manifest = BenchmarkManifest(
        generator_version=M5A_GENERATOR_VERSION,
        task_count=len(cases),
        source_group_count=len({source.source_group_id for source in sources}),
        split_summaries=tuple(split_summaries),
        source_registry_sha256=source_hash,
        task_registry_sha256=task_registry_hash,
        evaluator_contract_sha256=_evaluator_contract_hash(),
        locked_test_seal_sha256=locked_hash,
        adversarial_test_seal_sha256=adversarial_hash,
        frozen_at=freeze_time,
    )
    manifest_path = root / "benchmark_manifest.json"
    manifest_hash = _write_json(manifest_path, manifest)
    (root / "benchmark_manifest.sha256").write_text(
        f"{manifest_hash}  benchmark_manifest.json\n", encoding="utf-8"
    )
    audit_path = root / "leakage_audit.json"
    health_path = root / "benchmark_health.json"
    _write_json(audit_path, audit)
    _write_json(health_path, health)
    human_ids = tuple(
        case.public_task.task_id
        for case in cases
        if case.private_gold.split in {DatasetSplit.DEV, DatasetSplit.VALIDATION}
    )[:50]
    packet = HumanCalibrationPacket(
        case_ids=human_ids,
        status="pending_human_review",
    )
    protocol = SemanticEvaluationProtocol(
        rationale=(
            "CutAgentBench v0.1 quantitative tasks have deterministic media and objective "
            "constraints; a VLM Judge is disabled until real human calibration exists."
        )
    )
    packet_path = root / "calibration/human_calibration_packet.json"
    protocol_path = root / "calibration/semantic_evaluation_protocol.json"
    _write_json(packet_path, packet)
    _write_json(protocol_path, protocol)
    private_files.append(str(packet_path))
    return BenchmarkWriteResult(
        root=str(root),
        manifest=manifest,
        manifest_sha256=manifest_hash,
        audit=audit,
        health=health,
        public_files=tuple(public_files),
        private_files=tuple(private_files),
    )


def append_locked_access_log(path: Path, record: LockedTestAccessRecord) -> None:
    """Append one canonical JSONL access declaration for an official protected run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json_bytes(record).decode("utf-8") + "\n")


def scan_runtime_for_benchmark_task_rules(root: Path, task_ids: Sequence[str]) -> tuple[str, ...]:
    """Detect accidental per-task IDs in runtime prompts/rules without reading private Gold."""

    task_pattern = re.compile("|".join(re.escape(item) for item in task_ids)) if task_ids else None
    if task_pattern is None:
        return ()
    hits: list[str] = []
    for path in sorted((root / "src/cutagent").rglob("*.py")):
        if task_pattern.search(path.read_text(encoding="utf-8")):
            hits.append(str(path.relative_to(root)))
    return tuple(hits)
