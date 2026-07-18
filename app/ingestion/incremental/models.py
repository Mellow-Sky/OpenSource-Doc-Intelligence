"""Pure data structures used to plan idempotent source synchronization."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class SyncStats(BaseModel):
    """Counters emitted by every synchronization run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scanned: int = Field(default=0, ge=0)
    created: int = Field(default=0, ge=0)
    updated: int = Field(default=0, ge=0)
    unchanged: int = Field(default=0, ge=0)
    deleted: int = Field(default=0, ge=0)
    chunks_created: int = Field(default=0, ge=0)
    chunks_updated: int = Field(default=0, ge=0)
    chunks_deleted: int = Field(default=0, ge=0)
    duplicates_skipped: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)

    def add(self, **increments: int) -> SyncStats:
        """Return a new counter set after applying non-negative increments."""
        unknown = set(increments).difference(type(self).model_fields)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown sync statistic fields: {names}")
        if any(value < 0 for value in increments.values()):
            raise ValueError("sync statistic increments must be non-negative")
        values = self.model_dump()
        for name, value in increments.items():
            values[name] += value
        return type(self).model_validate(values)

    def merge(self, other: SyncStats) -> SyncStats:
        """Combine counters from independently processed batches."""
        return type(self).model_validate(
            {name: getattr(self, name) + getattr(other, name) for name in type(self).model_fields}
        )


class DocumentAction(StrEnum):
    """Write required for one logical source document."""

    CREATE = "create"
    UPDATE = "update"
    UNCHANGED = "unchanged"


class IncomingDocumentState(BaseModel):
    """Identity and content fingerprint produced by the current source scan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: UUID
    external_id: str = Field(min_length=1, max_length=1024)
    content_hash: str = Field(pattern=SHA256_PATTERN)


class StoredDocumentState(IncomingDocumentState):
    """Minimal persisted document state needed for incremental decisions."""

    id: UUID
    deleted_at: datetime | None = None


class DocumentDecision(BaseModel):
    """Decision for one incoming document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    incoming: IncomingDocumentState
    action: DocumentAction
    document_id: UUID | None = None
    reason: str

    @model_validator(mode="after")
    def validate_existing_identity(self) -> DocumentDecision:
        if self.action is DocumentAction.CREATE and self.document_id is not None:
            raise ValueError("a create decision cannot reference an existing document")
        if self.action is not DocumentAction.CREATE and self.document_id is None:
            raise ValueError("an existing document decision requires document_id")
        return self


class DocumentSyncPlan(BaseModel):
    """Complete document-level plan for one authoritative source snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decisions: list[DocumentDecision] = Field(default_factory=list)
    missing_document_ids: list[UUID] = Field(default_factory=list)
    stats: SyncStats


class ChunkAction(StrEnum):
    """Write and embedding work required for one incoming chunk."""

    CREATE = "create"
    UPDATE = "update"
    MOVE = "move"
    RESTORE = "restore"
    UNCHANGED = "unchanged"


class IncomingChunkState(BaseModel):
    """Stable chunk position and content fingerprint for a new parse."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: UUID
    chunk_index: int = Field(ge=0)
    content_hash: str = Field(pattern=SHA256_PATTERN)


class StoredChunkState(IncomingChunkState):
    """Minimal persisted chunk state needed for hash-diff planning."""

    id: UUID
    deleted_at: datetime | None = None


class ChunkDecision(BaseModel):
    """Incremental action for an incoming chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    incoming: IncomingChunkState
    action: ChunkAction
    chunk_id: UUID | None = None
    reason: str

    @model_validator(mode="after")
    def validate_existing_identity(self) -> ChunkDecision:
        if self.action is ChunkAction.CREATE and self.chunk_id is not None:
            raise ValueError("a create decision cannot reference an existing chunk")
        if self.action is not ChunkAction.CREATE and self.chunk_id is None:
            raise ValueError("an existing chunk decision requires chunk_id")
        return self

    @property
    def requires_embedding(self) -> bool:
        """Whether this chunk's searchable text changed or is newly created."""
        return self.action in {ChunkAction.CREATE, ChunkAction.UPDATE}


class ChunkSyncPlan(BaseModel):
    """Chunk hash diff and the exact embedding workload it implies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decisions: list[ChunkDecision] = Field(default_factory=list)
    removed_chunk_ids: list[UUID] = Field(default_factory=list)
    stats: SyncStats

    @property
    def embedding_chunk_indices(self) -> list[int]:
        """Return only new or content-changed chunk indexes."""
        return [
            decision.incoming.chunk_index
            for decision in self.decisions
            if decision.requires_embedding
        ]


class CursorCheckpoint(BaseModel):
    """Durable upstream checkpoint written only after a successful sync."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: UUID
    cursor_type: str = Field(min_length=1, max_length=64)
    cursor_value: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobStatus(StrEnum):
    """Durable ingestion worker lifecycle."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
