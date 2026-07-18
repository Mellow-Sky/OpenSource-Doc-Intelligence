"""Chunk models that retain precise provenance for citations."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourcePosition(BaseModel):
    """Offsets and optional line numbers locating a chunk in its source document."""

    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_order(self) -> SourcePosition:
        if self.end_offset < self.start_offset:
            msg = "end_offset must not precede start_offset"
            raise ValueError(msg)
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            msg = "end_line must not precede start_line"
            raise ValueError(msg)
        return self


class ChunkDraft(BaseModel):
    """A structure-aware chunk awaiting persistence and embedding."""

    model_config = ConfigDict(extra="forbid")

    chunk_index: int = Field(ge=0)
    parent_index: int | None = Field(default=None, ge=0)
    heading_path: list[str] = Field(default_factory=list)
    content: str = Field(min_length=1)
    contextualized_content: str = Field(min_length=1)
    token_count: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    position: SourcePosition
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(ChunkDraft):
    """Persisted chunk used by retrieval and citation services."""

    id: UUID
    document_id: UUID
    parent_chunk_id: UUID | None = None
    document_title: str
    canonical_url: str | None = None
    document_type: str
    embedding_model: str | None = None
    embedding_dimension: int | None = Field(default=None, ge=1)
