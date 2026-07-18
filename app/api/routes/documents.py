"""Read-only document, chunk, and exact citation provenance endpoints."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db_session
from app.core.exceptions import NotFoundError
from app.core.security import require_api_key
from app.db.models.source_document import Chunk, Document
from app.repositories.chunk_repository import ChunkDetailRecord, ChunkRepository
from app.repositories.document_repository import DocumentListFilters, DocumentRepository
from app.schemas.documents import (
    ChunkResponse,
    CitationDetailResponse,
    DocumentDetailResponse,
    DocumentPageResponse,
    DocumentResponse,
)

router = APIRouter(
    prefix="/api/v1",
    tags=["documents"],
    dependencies=[Depends(require_api_key)],
)


@router.get("/documents", response_model=DocumentPageResponse)
async def list_documents(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    source_id: Annotated[list[UUID] | None, Query()] = None,
    source_type: Annotated[list[str] | None, Query()] = None,
    document_type: Annotated[list[str] | None, Query()] = None,
    version: Annotated[list[str] | None, Query()] = None,
    language: Annotated[list[str] | None, Query()] = None,
    search: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentPageResponse:
    """Return a stable, filterable page of active documents."""
    page = await DocumentRepository(session).list_page(
        filters=DocumentListFilters(
            source_ids=source_id or (),
            source_types=source_type or (),
            document_types=document_type or (),
            versions=version or (),
            languages=language or (),
            search=search,
        ),
        limit=limit,
        offset=offset,
    )
    return DocumentPageResponse(
        items=[_document_response(item) for item in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def document_detail(
    document_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> DocumentDetailResponse:
    """Return one active document with its source and chunk cardinality."""
    detail = await DocumentRepository(session).get_detail(document_id)
    if detail is None:
        raise NotFoundError("Document was not found")
    base = _document_response(detail.document).model_dump()
    return DocumentDetailResponse(
        **base,
        source_name=detail.source_name,
        source_type=detail.source_type,
        active_chunk_count=detail.active_chunk_count,
    )


@router.get("/chunks/{chunk_id}", response_model=ChunkResponse)
async def chunk_detail(
    chunk_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ChunkResponse:
    """Return a full active chunk and exact source offsets."""
    detail = await ChunkRepository(session).get_detail(chunk_id)
    if detail is None:
        raise NotFoundError("Chunk was not found")
    return _chunk_response(detail.chunk, detail)


@router.get("/citations/{chunk_id}", response_model=CitationDetailResponse)
async def citation_detail(
    chunk_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> CitationDetailResponse:
    """Return a cited chunk, neighbours, metadata, URL, and source positions."""
    neighbourhood = await ChunkRepository(session).get_neighbourhood(chunk_id)
    if neighbourhood is None:
        raise NotFoundError("Citation chunk was not found")
    detail = neighbourhood.detail
    return CitationDetailResponse(
        chunk=_chunk_response(detail.chunk, detail),
        previous_chunk=(
            _chunk_response(neighbourhood.previous_chunk, detail)
            if neighbourhood.previous_chunk is not None
            else None
        ),
        next_chunk=(
            _chunk_response(neighbourhood.next_chunk, detail)
            if neighbourhood.next_chunk is not None
            else None
        ),
        document_metadata=detail.document_metadata,
        source_url=detail.canonical_url,
    )


def _document_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        source_id=document.source_id,
        external_id=document.external_id,
        document_type=document.document_type,
        title=document.title,
        canonical_url=document.canonical_url,
        repository_path=document.repository_path,
        source_version=document.source_version,
        language=document.language,
        content_hash=document.content_hash,
        metadata=dict(document.metadata_ or {}),
        status=document.status,
        first_seen_at=document.first_seen_at,
        last_seen_at=document.last_seen_at,
        indexed_at=document.indexed_at,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _chunk_response(chunk: Chunk, detail: ChunkDetailRecord) -> ChunkResponse:
    return ChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        parent_chunk_id=chunk.parent_chunk_id,
        document_title=detail.document_title,
        document_type=detail.document_type,
        heading_path=list(chunk.heading_path or []),
        content=chunk.content,
        contextualized_content=chunk.contextualized_content,
        token_count=chunk.token_count,
        content_hash=chunk.content_hash,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        metadata=dict(chunk.metadata_ or {}),
        canonical_url=detail.canonical_url,
        embedding_model=chunk.embedding_model,
        embedding_dimension=chunk.embedding_dimension,
        created_at=chunk.created_at,
        updated_at=chunk.updated_at,
    )
