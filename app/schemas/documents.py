"""Public document, chunk, and citation provenance schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    id: UUID
    source_id: UUID
    external_id: str
    document_type: str
    title: str
    canonical_url: str | None
    repository_path: str | None
    source_version: str | None
    language: str
    content_hash: str
    metadata: dict[str, Any]
    status: str
    first_seen_at: datetime
    last_seen_at: datetime
    indexed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class DocumentPageResponse(BaseModel):
    items: list[DocumentResponse]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class DocumentDetailResponse(DocumentResponse):
    source_name: str
    source_type: str
    active_chunk_count: int = Field(ge=0)


class ChunkResponse(BaseModel):
    id: UUID
    document_id: UUID
    chunk_index: int
    parent_chunk_id: UUID | None
    document_title: str
    document_type: str
    heading_path: list[str]
    content: str
    contextualized_content: str
    token_count: int
    content_hash: str
    start_offset: int
    end_offset: int
    start_line: int | None
    end_line: int | None
    metadata: dict[str, Any]
    canonical_url: str | None
    embedding_model: str | None
    embedding_dimension: int | None
    created_at: datetime
    updated_at: datetime


class CitationDetailResponse(BaseModel):
    chunk: ChunkResponse
    previous_chunk: ChunkResponse | None
    next_chunk: ChunkResponse | None
    document_metadata: dict[str, Any]
    source_url: str | None
