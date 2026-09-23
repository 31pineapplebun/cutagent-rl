"""Evaluator-private CutAgentBench v0.1 contracts.

Nothing in this module is part of the deployed ``cutagent`` package.  Public
tasks are represented by :class:`cutagent.schemas.task_input.TaskInput`; all
split, source-group, difficulty, answer, and evaluator fields live here.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.media import TimeRange
from cutagent.schemas.task_input import TaskInput
from cutagent.schemas.tools import ToolName
from cutagent_evaluation.schemas import DatasetSplit

BENCHMARK_VERSION: Literal["cutagentbench-v0.1"] = "cutagentbench-v0.1"
EVALUATOR_VERSION: Literal["m5a-objective-evaluator-v1"] = "m5a-objective-evaluator-v1"
FAILURE_TAXONOMY_VERSION: Literal["m5a-failure-taxonomy-v1"] = "m5a-failure-taxonomy-v1"
METRIC_DEFINITION_VERSION: Literal["m5a-metrics-v1"] = "m5a-metrics-v1"


class MediaFamily(StrEnum):
    GENERATED = "deterministic_generated"
    LICENSED_REAL = "licensed_real"


class TaskFamily(StrEnum):
    SEARCH_GROUNDING = "search_grounding"
    SINGLE_EDIT = "single_edit"
    MULTI_SCENE_COMPOSITION = "multi_scene_composition"
    MULTI_CONSTRAINT = "multi_constraint_editing"
    HARD_NEGATIVE = "hard_negative"
    IMPOSSIBLE = "impossible_unanswerable"
    OBSERVABLE_RECOVERY = "observable_failure_recovery"


class DifficultyLevel(StrEnum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


class ExpectedTerminalBehavior(StrEnum):
    SUCCESS = "SUCCESS"
    CANNOT_COMPLETE = "CANNOT_COMPLETE"


class BenchmarkFailureCause(StrEnum):
    UPSTREAM_PERCEPTION_ERROR = "upstream_perception_error"
    RETRIEVAL_ERROR = "retrieval_error"
    PLANNING_ERROR = "planning_error"
    WRONG_TOOL = "wrong_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    TOOL_EXECUTION_FAILURE = "tool_execution_failure"
    VERIFICATION_ERROR = "verification_error"
    HANDOFF_ERROR = "handoff_error"
    RECOVERY_ERROR = "recovery_error"
    PREMATURE_FINISH = "premature_finish"
    PREMATURE_REFUSAL = "premature_refusal"
    LOOP_STAGNATION = "loop_stagnation"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    MODEL_FORMAT_ERROR = "model_format_error"


class GeneratedSceneAnnotation(SchemaModel):
    scene_id: Identifier
    time_range: TimeRange
    entities: tuple[NonEmptyStr, ...]
    action: NonEmptyStr
    visible_text: NonEmptyStr
    transcript: NonEmptyStr


class BenchmarkSourceRecord(SchemaModel):
    source_group_id: Identifier
    split: DatasetSplit
    media_family: MediaFamily
    video_id: Identifier
    source_artifact: ArtifactRef
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    structural_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivative_of_source_group_id: Identifier | None = None
    generator_version: NonEmptyStr | None = None
    provenance_reference: NonEmptyStr
    license: NonEmptyStr
    scenes: tuple[GeneratedSceneAnnotation, ...]

    @model_validator(mode="after")
    def validate_source_identity(self) -> BenchmarkSourceRecord:
        if self.source_artifact.sha256 != self.source_sha256:
            raise ValueError("source artifact and source record hashes differ")
        if self.media_family == MediaFamily.GENERATED and self.generator_version is None:
            raise ValueError("generated media requires a generator version")
        return self


class LicensedClipProvenance(SchemaModel):
    clip_id: Identifier
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_clip_interval(self) -> LicensedClipProvenance:
        if self.start_ms >= self.end_ms:
            raise ValueError("licensed clip interval must be non-empty")
        return self


class LicensedQualitativeSource(SchemaModel):
    source_id: Identifier
    title: NonEmptyStr
    source_page: NonEmptyStr
    license: NonEmptyStr
    license_url: NonEmptyStr
    attribution: NonEmptyStr
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license_page_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    download_date_utc: datetime
    clips: tuple[LicensedClipProvenance, ...] = Field(min_length=1)

    @field_validator("download_date_utc")
    @classmethod
    def licensed_download_time_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("licensed source download time must be timezone-aware")
        return value


class LicensedQualitativeRegistry(SchemaModel):
    registry_version: Literal["m5a-licensed-qualitative-v1"] = "m5a-licensed-qualitative-v1"
    source_count: int = Field(gt=0)
    clip_count: int = Field(gt=0)
    quantitative_task_count: Literal[0] = 0
    exclusion_reason: NonEmptyStr
    sources: tuple[LicensedQualitativeSource, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_licensed_counts(self) -> LicensedQualitativeRegistry:
        if self.source_count != len(self.sources):
            raise ValueError("licensed source count differs from registry")
        if self.clip_count != sum(len(source.clips) for source in self.sources):
            raise ValueError("licensed clip count differs from registry")
        return self


class GroundingConstraint(SchemaModel):
    constraint_type: Literal["grounding"] = "grounding"
    mandatory: bool = True
    relevant_video_ids: tuple[Identifier, ...] = Field(min_length=1)
    relevant_scene_ids: tuple[Identifier, ...] = Field(min_length=1)
    acceptable_time_ranges: tuple[TimeRange, ...] = Field(min_length=1)


class DurationGoldConstraint(SchemaModel):
    constraint_type: Literal["duration"] = "duration"
    mandatory: bool = True
    target_ms: int = Field(gt=0)
    tolerance_ms: int = Field(default=120, ge=0)


class ResolutionGoldConstraint(SchemaModel):
    constraint_type: Literal["resolution"] = "resolution"
    mandatory: bool = True
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class SubtitleGoldConstraint(SchemaModel):
    constraint_type: Literal["subtitle"] = "subtitle"
    mandatory: bool = True
    text: NonEmptyStr
    time_range: TimeRange


class SceneOrderGoldConstraint(SchemaModel):
    constraint_type: Literal["scene_order"] = "scene_order"
    mandatory: bool = True
    ordered_scene_ids: tuple[Identifier, ...] = Field(min_length=2)


class SpeedGoldConstraint(SchemaModel):
    constraint_type: Literal["speed"] = "speed"
    mandatory: bool = True
    speed_factor: float = Field(gt=0.24, le=4.0)


class StreamGoldConstraint(SchemaModel):
    constraint_type: Literal["streams"] = "streams"
    mandatory: bool = True
    require_video: bool = True
    require_audio: bool = False


class DerivedArtifactGoldConstraint(SchemaModel):
    constraint_type: Literal["derived_artifact"] = "derived_artifact"
    mandatory: bool = True


class AbsenceGoldConstraint(SchemaModel):
    constraint_type: Literal["requested_evidence_absent"] = "requested_evidence_absent"
    mandatory: bool = True
    requested_description: NonEmptyStr


ObjectiveConstraint = Annotated[
    GroundingConstraint
    | DurationGoldConstraint
    | ResolutionGoldConstraint
    | SubtitleGoldConstraint
    | SceneOrderGoldConstraint
    | SpeedGoldConstraint
    | StreamGoldConstraint
    | DerivedArtifactGoldConstraint
    | AbsenceGoldConstraint,
    Field(discriminator="constraint_type"),
]


class FailureInjectionGold(SchemaModel):
    category: Literal[
        "search_no_results",
        "timeout",
        "invalid_arguments",
        "artifact_not_allowed",
        "post_validation_failure",
        "corrupt_media",
        "incompatible_concat",
        "invalid_subtitle",
        "output_too_large",
        "repeated_editor",
    ]
    inject_on_tool: ToolName
    occurrence: int = Field(default=1, gt=0)
    expected_online_status: Literal["invalid", "timeout", "error"]


class CutAgentBenchGold(SchemaModel):
    gold_id: Identifier
    task_id: Identifier
    source_group_id: Identifier
    split: DatasetSplit
    task_family: TaskFamily
    task_subtype: Identifier
    difficulty: DifficultyLevel
    relevant_video_ids: tuple[Identifier, ...]
    relevant_scene_ids: tuple[Identifier, ...]
    acceptable_time_ranges: tuple[TimeRange, ...]
    required_tool_capabilities: tuple[Identifier, ...]
    required_tools: tuple[ToolName, ...]
    acceptable_tool_sequences: tuple[tuple[ToolName, ...], ...] = ()
    objective_constraints: tuple[ObjectiveConstraint, ...]
    expected_terminal_behavior: ExpectedTerminalBehavior
    failure_injection: FailureInjectionGold | None = None
    primary_failure_if_unsolved: BenchmarkFailureCause
    annotation_version: Literal["m5a-gold-v1"] = "m5a-gold-v1"

    @field_validator(
        "relevant_video_ids",
        "relevant_scene_ids",
        "required_tool_capabilities",
        "required_tools",
    )
    @classmethod
    def unique_tuple(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("private Gold tuple values must be unique")
        return value

    @model_validator(mode="after")
    def validate_gold_semantics(self) -> CutAgentBenchGold:
        impossible = self.task_family == TaskFamily.IMPOSSIBLE
        if impossible != (
            self.expected_terminal_behavior == ExpectedTerminalBehavior.CANNOT_COMPLETE
        ):
            raise ValueError("only impossible tasks may require CANNOT_COMPLETE")
        if self.task_family == TaskFamily.OBSERVABLE_RECOVERY and self.failure_injection is None:
            raise ValueError("recovery tasks require private failure injection metadata")
        if (
            self.task_family != TaskFamily.OBSERVABLE_RECOVERY
            and self.failure_injection is not None
        ):
            raise ValueError("failure injection is restricted to recovery tasks")
        mandatory = [item for item in self.objective_constraints if item.mandatory]
        if not mandatory:
            raise ValueError("every task requires at least one mandatory objective constraint")
        return self


class CutAgentBenchCase(SchemaModel):
    public_task: TaskInput
    private_gold: CutAgentBenchGold

    @model_validator(mode="after")
    def bind_public_private(self) -> CutAgentBenchCase:
        if self.public_task.task_id != self.private_gold.task_id:
            raise ValueError("public task and private Gold task IDs differ")
        return self


class BenchmarkSplitSummary(SchemaModel):
    split: DatasetSplit
    task_count: int = Field(ge=0)
    source_group_count: int = Field(ge=0)
    public_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    private_gold_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BenchmarkManifest(SchemaModel):
    benchmark_version: Literal["cutagentbench-v0.1"] = BENCHMARK_VERSION
    evaluator_version: Literal["m5a-objective-evaluator-v1"] = EVALUATOR_VERSION
    failure_taxonomy_version: Literal["m5a-failure-taxonomy-v1"] = FAILURE_TAXONOMY_VERSION
    metric_definition_version: Literal["m5a-metrics-v1"] = METRIC_DEFINITION_VERSION
    generator_version: NonEmptyStr
    task_count: int = Field(gt=0)
    source_group_count: int = Field(gt=0)
    split_summaries: tuple[BenchmarkSplitSummary, ...] = Field(min_length=5, max_length=5)
    source_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locked_test_seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adversarial_test_seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_at: datetime

    @field_validator("frozen_at")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("benchmark freeze time must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_split_manifest(self) -> BenchmarkManifest:
        expected = set(DatasetSplit)
        observed = {item.split for item in self.split_summaries}
        if observed != expected:
            raise ValueError("benchmark manifest must contain all five splits")
        if sum(item.task_count for item in self.split_summaries) != self.task_count:
            raise ValueError("split task counts do not sum to manifest task count")
        return self


class LockedTestAccessRecord(SchemaModel):
    access_id: Identifier
    benchmark_version: Literal["cutagentbench-v0.1"] = BENCHMARK_VERSION
    split: Literal[DatasetSplit.LOCKED_TEST, DatasetSplit.ADVERSARIAL_TEST]
    git_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    model_revision: NonEmptyStr
    adapter_or_checkpoint_hash: NonEmptyStr
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    access_reason: NonEmptyStr
    declared_metrics: tuple[NonEmptyStr, ...] = Field(min_length=1)
    accessed_at: datetime

    @field_validator("accessed_at")
    @classmethod
    def access_time_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("access time must be timezone-aware")
        return value


class HumanCalibrationRating(SchemaModel):
    case_id: Identifier
    rater_id: Identifier
    task_success: bool | None = None
    semantic_constraint_score: float | None = Field(default=None, ge=0, le=1)
    primary_failure: BenchmarkFailureCause | None = None
    confidence: Literal["low", "medium", "high"] | None = None
    notes: str = ""


class HumanCalibrationPacket(SchemaModel):
    packet_version: Literal["m5a-human-calibration-v1"] = "m5a-human-calibration-v1"
    case_ids: tuple[Identifier, ...] = Field(min_length=40, max_length=60)
    required_independent_raters: Literal[2] = 2
    ratings: tuple[HumanCalibrationRating, ...] = ()
    status: Literal["pending_human_review", "partially_reviewed", "complete"]


class SemanticEvaluationProtocol(SchemaModel):
    rubric_version: Literal["m5a-semantic-rubric-v1"] = "m5a-semantic-rubric-v1"
    vlm_judge_enabled: Literal[False] = False
    judge_model_id: None = None
    rationale: NonEmptyStr
    human_calibration_required_before_enablement: Literal[True] = True


class BenchmarkHealthSummary(SchemaModel):
    task_count: int = Field(gt=0)
    source_group_count: int = Field(gt=0)
    split_counts: dict[str, int]
    task_family_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    average_required_tool_count: float = Field(ge=0)
    average_mandatory_constraints: float = Field(ge=0)
    hard_negative_proportion: float = Field(ge=0, le=1)
    impossible_proportion: float = Field(ge=0, le=1)
    recovery_proportion: float = Field(ge=0, le=1)
    sanity_baselines: dict[str, float]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
