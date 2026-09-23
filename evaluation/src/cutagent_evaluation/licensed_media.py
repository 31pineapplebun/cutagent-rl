"""Private provenance contracts for the M1B.5 licensed-media validation set."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from cutagent.schemas.base import Identifier, NonEmptyStr, SchemaModel


class LicensedClipSpec(SchemaModel):
    clip_id: Identifier
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> LicensedClipSpec:
        if self.start_ms >= self.end_ms:
            raise ValueError("licensed clip must satisfy start_ms < end_ms")
        return self


class LicensedSourceSpec(SchemaModel):
    source_id: Identifier
    title: NonEmptyStr
    source_page: NonEmptyStr
    download_url: NonEmptyStr
    license: NonEmptyStr
    license_url: NonEmptyStr
    attribution: NonEmptyStr
    license_evidence: NonEmptyStr
    intended_coverage: tuple[NonEmptyStr, ...] = Field(min_length=1)
    clips: tuple[LicensedClipSpec, ...] = Field(min_length=1)

    @field_validator("source_page", "download_url", "license_url")
    @classmethod
    def validate_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("licensed-media URLs must use HTTPS")
        return value

    @model_validator(mode="after")
    def validate_clips(self) -> LicensedSourceSpec:
        ordered = sorted(self.clips, key=lambda item: (item.start_ms, item.end_ms, item.clip_id))
        if tuple(ordered) != self.clips:
            raise ValueError("licensed clips must be deterministically ordered")
        if len({item.clip_id for item in self.clips}) != len(self.clips):
            raise ValueError("licensed clip IDs must be unique within a source")
        return self


class LicensedMediaManifest(SchemaModel):
    dataset_id: Identifier
    selection_frozen_at: NonEmptyStr
    sources: tuple[LicensedSourceSpec, ...] = Field(min_length=3, max_length=5)

    @model_validator(mode="after")
    def validate_dataset(self) -> LicensedMediaManifest:
        if len({source.source_id for source in self.sources}) != len(self.sources):
            raise ValueError("licensed source IDs must be unique")
        clip_ids = [clip.clip_id for source in self.sources for clip in source.clips]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("licensed clip IDs must be globally unique")
        if not 15 <= len(clip_ids) <= 30:
            raise ValueError("licensed validation set must contain 15-30 clips")
        return self


class DerivedClipProvenance(SchemaModel):
    clip_id: Identifier
    source_id: Identifier
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    relative_path: NonEmptyStr
    extraction_command: tuple[NonEmptyStr, ...]


class SourceMediaProvenance(SchemaModel):
    source_id: Identifier
    title: NonEmptyStr
    source_page: NonEmptyStr
    resolved_download_url: NonEmptyStr
    license: NonEmptyStr
    license_url: NonEmptyStr
    attribution: NonEmptyStr
    license_page_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license_page_relative_path: NonEmptyStr
    download_date_utc: datetime
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    relative_path: NonEmptyStr
    clips: tuple[DerivedClipProvenance, ...] = Field(min_length=1)


class LicensedMediaProvenance(SchemaModel):
    dataset_id: Identifier
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_at_utc: datetime
    ffmpeg_version: NonEmptyStr
    ffprobe_version: NonEmptyStr
    sources: tuple[SourceMediaProvenance, ...] = Field(min_length=3, max_length=5)

    @property
    def clip_count(self) -> int:
        return sum(len(source.clips) for source in self.sources)
