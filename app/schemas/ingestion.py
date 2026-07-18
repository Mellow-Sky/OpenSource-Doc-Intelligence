"""Validated knowledge-source configuration schemas."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from app.ingestion.incremental import JobStatus, SyncStats


def _reject_control_characters(value: str | None) -> str | None:
    """Keep identifiers safe for structured logs and HTTP response headers."""
    if value is not None and any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError("value must not contain control characters")
    return value


class SourceConfig(BaseModel):
    """One generic source definition loaded from YAML and persisted in PostgreSQL."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    source_type: str = Field(min_length=1, max_length=64)
    base_url: HttpUrl | None = None
    repository: str | None = None
    branch: str | None = None
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("repository")
    @classmethod
    def validate_optional_repository(cls, value: str | None) -> str | None:
        if value is not None and (
            value.count("/") != 1 or any(not part for part in value.split("/"))
        ):
            msg = "repository must use owner/name format"
            raise ValueError(msg)
        return value


class SourceConfigFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources: list[SourceConfig]


def load_source_configs(path: Path) -> list[SourceConfig]:
    """Read and strictly validate source definitions from a YAML file."""
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return SourceConfigFile.model_validate(payload).sources


class IngestionJobOptions(BaseModel):
    """Bounded options understood by the ingestion worker."""

    model_config = ConfigDict(extra="forbid")

    dry_run: bool = False
    allow_delete_missing: bool = True


class IngestionJobCreateRequest(BaseModel):
    """Request for the generic ingestion job endpoint."""

    model_config = ConfigDict(extra="forbid")

    source_id: UUID
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=255)
    requested_by: str | None = Field(default=None, min_length=1, max_length=255)
    options: IngestionJobOptions = Field(default_factory=IngestionJobOptions)

    @field_validator("idempotency_key", "requested_by")
    @classmethod
    def validate_log_fields(cls, value: str | None) -> str | None:
        return _reject_control_characters(value)


class SourceSyncRequest(BaseModel):
    """Request body for the source-scoped synchronization shortcut."""

    model_config = ConfigDict(extra="forbid")

    idempotency_key: str | None = Field(default=None, min_length=1, max_length=255)
    requested_by: str | None = Field(default=None, min_length=1, max_length=255)
    options: IngestionJobOptions = Field(default_factory=IngestionJobOptions)

    @field_validator("idempotency_key", "requested_by")
    @classmethod
    def validate_log_fields(cls, value: str | None) -> str | None:
        return _reject_control_characters(value)


class IngestionJobResponse(BaseModel):
    """Durable state returned by enqueue and polling endpoints."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    source_id: UUID
    idempotency_key: str
    status: JobStatus
    requested_by: str | None
    options: IngestionJobOptions
    stats: SyncStats
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    heartbeat_at: datetime | None
    created_at: datetime
    updated_at: datetime
