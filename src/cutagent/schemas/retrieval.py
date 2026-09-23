"""Public M2A contracts for evidence-linked scene retrieval."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel
from cutagent.schemas.media import TimeRange
from cutagent.schemas.perception import EvidenceRef

RetrievalEvidenceType = Literal[
    "transcript",
    "ocr",
    "structured_text",
    "keyframe",
    "native_video_verification",
]
RetrievalMethod = Literal["bm25", "dense_text", "visual", "fusion", "adaptive_hybrid"]
DocumentField = Literal["transcript", "ocr", "structured_semantic", "combined"]
QueryIntent = Literal[
    "speech_or_quote",
    "visible_text",
    "static_visual_entity",
    "semantic_scene",
    "action_or_motion",
    "ambiguous",
]
RetrievalChannel = Literal[
    "bm25_transcript",
    "bm25_ocr",
    "bm25_structured",
    "bm25_combined",
    "dense_text",
    "visual",
]
NativeVideoPolicy = Literal["disabled", "ambiguous_motion_only"]
NativeVerificationStatus = Literal["supports", "contradicts", "uncertain"]


class RetrievalQuery(SchemaModel):
    """Only observable user/query information; contains no benchmark labels."""

    query_id: Identifier
    text: NonEmptyStr
    top_k: int = Field(default=10, ge=1, le=100)
    video_id: Identifier | None = None
    approximate_time_range: TimeRange | None = None
    required_evidence_types: tuple[RetrievalEvidenceType, ...] = ()

    @field_validator("required_evidence_types")
    @classmethod
    def unique_evidence_types(
        cls, value: tuple[RetrievalEvidenceType, ...]
    ) -> tuple[RetrievalEvidenceType, ...]:
        if len(value) != len(set(value)):
            raise ValueError("required evidence types must be unique")
        return value


class RetrievalEvidence(SchemaModel):
    """Why a retriever selected a candidate, linked to existing evidence."""

    evidence_type: RetrievalEvidenceType
    evidence_ref: EvidenceRef
    document_field: DocumentField | None = None
    matched_text: NonEmptyStr | None = None
    score: float


class RetrievalCandidate(SchemaModel):
    video_id: Identifier
    scene_id: Identifier
    time_range: TimeRange
    total_score: float
    component_scores: dict[Identifier, float] = Field(default_factory=dict)
    evidence_refs: tuple[EvidenceRef, ...] = Field(min_length=1)
    evidence_types: tuple[RetrievalEvidenceType, ...] = Field(min_length=1)
    evidence: tuple[RetrievalEvidence, ...] = Field(min_length=1)
    rank: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_evidence_summary(self) -> RetrievalCandidate:
        described_refs = {item.evidence_ref for item in self.evidence}
        if set(self.evidence_refs) != described_refs:
            raise ValueError("evidence_refs must exactly summarize evidence entries")
        described_types = {item.evidence_type for item in self.evidence}
        if set(self.evidence_types) != described_types:
            raise ValueError("evidence_types must exactly summarize evidence entries")
        return self


class AppliedRetrievalFilters(SchemaModel):
    video_id: Identifier | None = None
    approximate_time_range: TimeRange | None = None
    required_evidence_types: tuple[RetrievalEvidenceType, ...] = ()


class RetrievalResponse(SchemaModel):
    query_id: Identifier
    method: RetrievalMethod
    candidates: tuple[RetrievalCandidate, ...]
    applied_filters: AppliedRetrievalFilters
    latency_ms: float = Field(ge=0)
    cache_hit: bool

    @model_validator(mode="after")
    def validate_ranking(self) -> RetrievalResponse:
        ranks = tuple(candidate.rank for candidate in self.candidates)
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("candidate ranks must be consecutive and ordered")
        identities = {(item.video_id, item.scene_id) for item in self.candidates}
        if len(identities) != len(self.candidates):
            raise ValueError("retrieval candidates must be unique")
        return self


class TextEncoderSpec(SchemaModel):
    model_id: Literal["BAAI/bge-m3"] = "BAAI/bge-m3"
    revision: Literal["5617a9f61b028005a4858fdac845db406aefb181"] = (
        "5617a9f61b028005a4858fdac845db406aefb181"
    )
    license: Literal["mit"] = "mit"
    embedding_dimension: Literal[1024] = 1024
    dtype: Literal["bfloat16"] = "bfloat16"
    pooling: Literal["normalized_cls"] = "normalized_cls"


class VisualEncoderSpec(SchemaModel):
    model_id: Literal["google/siglip2-base-patch16-224"] = "google/siglip2-base-patch16-224"
    revision: Literal["75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"] = (
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    )
    license: Literal["apache-2.0"] = "apache-2.0"
    dtype: Literal["bfloat16"] = "bfloat16"
    scene_pooling: Literal["max_per_keyframe"] = "max_per_keyframe"


class RetrievalConfig(SchemaModel):
    text_encoder: TextEncoderSpec = Field(default_factory=TextEncoderSpec)
    visual_encoder: VisualEncoderSpec = Field(default_factory=VisualEncoderSpec)
    dense_document_field: DocumentField = "combined"
    sparse_document_fields: tuple[DocumentField, ...] = (
        "transcript",
        "ocr",
        "structured_semantic",
        "combined",
    )
    document_template_version: NonEmptyStr = "m2a-scene-document-v1"
    text_encoder_max_tokens: int = Field(default=512, ge=8, le=8192)
    text_batch_size: int = Field(default=16, ge=1, le=256)
    visual_batch_size: int = Field(default=16, ge=1, le=256)
    rrf_rank_constant: int = Field(default=60, ge=1)


class RetrievalIndexManifest(SchemaModel):
    index_id: Identifier
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_world_state_sha256: tuple[str, ...] = Field(min_length=1)
    scene_count: int = Field(gt=0)
    keyframe_count: int = Field(ge=0)
    document_template_version: NonEmptyStr
    component_cache_keys: dict[Identifier, str]
    component_artifacts: dict[Identifier, ArtifactRef]
    encoder_models: dict[Identifier, NonEmptyStr]
    library_versions: dict[Identifier, NonEmptyStr]
    build_time_ms: int = Field(ge=0)
    metadata: dict[NonEmptyStr, JsonValue] = Field(default_factory=dict)

    @field_validator("source_world_state_sha256")
    @classmethod
    def validate_world_hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("world-state hashes must be unique and sorted")
        if any(
            len(item) != 64 or any(char not in "0123456789abcdef" for char in item)
            for item in value
        ):
            raise ValueError("world-state hashes must be lowercase SHA-256 digests")
        return value

    @field_validator("component_cache_keys")
    @classmethod
    def validate_cache_keys(cls, value: dict[str, str]) -> dict[str, str]:
        for key in value.values():
            if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
                raise ValueError("component cache keys must be lowercase SHA-256 digests")
        return value


class RetrievalMetricsSummary(SchemaModel):
    method: NonEmptyStr
    query_count: int = Field(ge=0)
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_10: float = Field(ge=0, le=1)
    mean_temporal_iou: float = Field(ge=0, le=1)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)


class QueryAnalysis(SchemaModel):
    """Deterministic analysis derived exclusively from public query text."""

    analyzer_version: NonEmptyStr
    primary_intent: QueryIntent
    intent_scores: dict[QueryIntent, float]
    quoted_phrases: tuple[NonEmptyStr, ...] = ()
    action_terms: tuple[NonEmptyStr, ...] = ()
    entity_terms: tuple[NonEmptyStr, ...] = ()
    reasons: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @field_validator("intent_scores")
    @classmethod
    def validate_intent_scores(cls, value: dict[QueryIntent, float]) -> dict[QueryIntent, float]:
        expected: set[QueryIntent] = {
            "speech_or_quote",
            "visible_text",
            "static_visual_entity",
            "semantic_scene",
            "action_or_motion",
            "ambiguous",
        }
        if set(value) != expected:
            raise ValueError("intent_scores must contain every public query intent exactly once")
        if any(score < 0 or score > 1 for score in value.values()):
            raise ValueError("intent scores must be within [0, 1]")
        return value


class RetrievalPlan(SchemaModel):
    """Inspectable query-time plan; no evaluator label can be represented."""

    query_id: Identifier
    analyzer_version: NonEmptyStr
    routing_policy_version: NonEmptyStr
    primary_intent: QueryIntent
    enabled_channels: tuple[RetrievalChannel, ...] = Field(min_length=1)
    channel_weights: dict[RetrievalChannel, float]
    candidate_depth: int = Field(ge=1, le=100)
    rerank_depth: int = Field(ge=1, le=20)
    reranking_policy: Literal["none", "evidence_rules"]
    native_video_policy: NativeVideoPolicy
    native_video_depth: int = Field(ge=0, le=5)
    reasons: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> RetrievalPlan:
        if len(set(self.enabled_channels)) != len(self.enabled_channels):
            raise ValueError("enabled retrieval channels must be unique")
        if set(self.channel_weights) != set(self.enabled_channels):
            raise ValueError("channel weights must exactly match enabled channels")
        if any(weight <= 0 for weight in self.channel_weights.values()):
            raise ValueError("channel weights must be positive")
        if self.rerank_depth > self.candidate_depth:
            raise ValueError("rerank depth cannot exceed candidate depth")
        if self.native_video_policy == "disabled" and self.native_video_depth != 0:
            raise ValueError("disabled native-video policy requires depth zero")
        if self.native_video_policy != "disabled" and self.native_video_depth == 0:
            raise ValueError("enabled native-video policy requires positive depth")
        return self


class CandidateRerankTrace(SchemaModel):
    video_id: Identifier
    scene_id: Identifier
    original_rank: int = Field(ge=1)
    final_rank: int = Field(ge=1)
    adjustment: float
    matched_features: tuple[NonEmptyStr, ...] = ()
    penalties: tuple[NonEmptyStr, ...] = ()


class NativeVideoVerification(SchemaModel):
    query_id: Identifier
    video_id: Identifier
    scene_id: Identifier
    status: NativeVerificationStatus
    explanation: NonEmptyStr
    evidence_ref: EvidenceRef
    raw_output_artifact: ArtifactRef
    prompt_version: NonEmptyStr
    model_id: NonEmptyStr
    model_revision: NonEmptyStr
    latency_ms: int = Field(ge=0)
    peak_allocated_bytes: int | None = Field(default=None, ge=0)
    peak_reserved_bytes: int | None = Field(default=None, ge=0)
    cache_hit: bool
    unsupported_claims: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def validate_scene_clip_evidence(self) -> NativeVideoVerification:
        if self.evidence_ref.evidence_kind != "scene_clip":
            raise ValueError("native-video verification must cite scene-clip evidence")
        if self.evidence_ref.segment_id != self.scene_id:
            raise ValueError("native-video evidence must belong to the verified scene")
        return self


class AdaptiveRetrievalConfig(SchemaModel):
    analyzer_version: NonEmptyStr = "m2b-query-analyzer-v1"
    routing_policy_version: NonEmptyStr = "m2b-routing-policy-v1"
    weighted_rrf_version: NonEmptyStr = "m2b-weighted-rrf-v1"
    reranker_rule_version: NonEmptyStr = "m2b-evidence-reranker-v1"
    native_verifier_prompt_version: NonEmptyStr = "m2b-native-verifier-v1"
    rank_constant: int = Field(default=60, ge=1)
    candidate_depth: int = Field(default=20, ge=5, le=100)
    rerank_depth: int = Field(default=5, ge=1, le=20)
    native_video_depth: int = Field(default=2, ge=1, le=3)
    native_score_margin: float = Field(default=0.003, ge=0, le=1)
    native_video_enabled: bool = True
    disabled_channels: tuple[RetrievalChannel, ...] = ()

    @field_validator("disabled_channels")
    @classmethod
    def validate_disabled_channels(
        cls, value: tuple[RetrievalChannel, ...]
    ) -> tuple[RetrievalChannel, ...]:
        if len(set(value)) != len(value):
            raise ValueError("disabled channels must be unique")
        return value


class AdaptiveRetrievalResult(SchemaModel):
    query_analysis: QueryAnalysis
    plan: RetrievalPlan
    response: RetrievalResponse
    rerank_trace: tuple[CandidateRerankTrace, ...]
    native_verifications: tuple[NativeVideoVerification, ...]
    component_latency_ms: dict[Identifier, float]
    total_latency_ms: float = Field(ge=0)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_query_identity(self) -> AdaptiveRetrievalResult:
        query_id = self.response.query_id
        if self.plan.query_id != query_id:
            raise ValueError("retrieval plan and response query IDs differ")
        if any(item.query_id != query_id for item in self.native_verifications):
            raise ValueError("native verification belongs to another query")
        return self
