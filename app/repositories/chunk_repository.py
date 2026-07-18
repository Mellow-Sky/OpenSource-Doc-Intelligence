"""Batch-oriented async persistence for retrievable chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import Integer, Table, all_, and_, any_, bindparam, case, func, or_, select, update
from sqlalchemy.dialects.postgresql import ARRAY, insert
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.source_document import Chunk, Document
from app.ingestion.incremental.models import StoredChunkState
from app.repositories.batching import database_batches


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkUpsert:
    """Complete chunk row plus an optional newly generated embedding."""

    document_id: UUID
    chunk_index: int
    document_title: str
    heading_path: list[str]
    content: str
    contextualized_content: str
    token_count: int
    content_hash: str
    start_offset: int
    end_offset: int
    parent_chunk_id: UUID | None = None
    start_line: int | None = None
    end_line: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class PendingEmbeddingChunk:
    """Minimal immutable input read before model inference starts."""

    id: UUID
    content_hash: str
    contextualized_content: str


@dataclass(frozen=True, slots=True)
class ChunkEmbeddingUpdate:
    """Optimistic embedding update tied to the exact indexed text hash."""

    id: UUID
    content_hash: str
    embedding: list[float]
    model: str
    dimension: int


@dataclass(frozen=True, slots=True)
class ChunkDetailRecord:
    """A chunk with the document fields required for public provenance."""

    chunk: Chunk
    document_title: str
    document_type: str
    canonical_url: str | None
    document_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChunkNeighbourhood:
    """A cited chunk and its active document-local neighbours."""

    detail: ChunkDetailRecord
    previous_chunk: Chunk | None
    next_chunk: Chunk | None


class ChunkRepository:
    """Persist chunk batches without a per-chunk flush or commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, chunk_id: UUID, *, include_deleted: bool = False) -> Chunk | None:
        """Return one chunk, excluding soft-deleted state by default."""
        statement = select(Chunk).where(Chunk.id == chunk_id)
        if not include_deleted:
            statement = statement.where(Chunk.deleted_at.is_(None))
        return cast(Chunk | None, await self._session.scalar(statement))

    async def get_detail(
        self,
        chunk_id: UUID,
        *,
        include_deleted: bool = False,
    ) -> ChunkDetailRecord | None:
        """Load one chunk and its document provenance in a single query."""
        statement = (
            select(
                Chunk,
                Document.title.label("document_title"),
                Document.document_type.label("document_type"),
                Document.canonical_url.label("canonical_url"),
                Document.metadata_.label("document_metadata"),
            )
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id == chunk_id)
        )
        if not include_deleted:
            statement = statement.where(
                Chunk.deleted_at.is_(None),
                Document.deleted_at.is_(None),
            )
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None
        return ChunkDetailRecord(
            chunk=row[0],
            document_title=str(row.document_title),
            document_type=str(row.document_type),
            canonical_url=cast(str | None, row.canonical_url),
            document_metadata=dict(cast(dict[str, Any] | None, row.document_metadata) or {}),
        )

    async def get_neighbourhood(self, chunk_id: UUID) -> ChunkNeighbourhood | None:
        """Load citation detail in two fixed queries without relationship lazy loads."""
        detail = await self.get_detail(chunk_id)
        if detail is None:
            return None
        chunk = detail.chunk
        rows = list(
            await self._session.scalars(
                select(Chunk)
                .where(
                    Chunk.document_id == chunk.document_id,
                    Chunk.deleted_at.is_(None),
                    Chunk.chunk_index.in_([chunk.chunk_index - 1, chunk.chunk_index + 1]),
                )
                .order_by(Chunk.chunk_index)
            )
        )
        previous = next((item for item in rows if item.chunk_index < chunk.chunk_index), None)
        following = next((item for item in rows if item.chunk_index > chunk.chunk_index), None)
        return ChunkNeighbourhood(
            detail=detail,
            previous_chunk=previous,
            next_chunk=following,
        )

    async def list_states_for_document(self, document_id: UUID) -> list[StoredChunkState]:
        """Load minimal state for the pure chunk hash-diff planner."""
        return (await self.list_states_for_documents([document_id]))[document_id]

    async def list_states_for_documents(
        self,
        document_ids: Sequence[UUID],
    ) -> dict[UUID, list[StoredChunkState]]:
        """Load and group chunk state for a document batch with one SQL query."""
        unique_ids = list(dict.fromkeys(document_ids))
        grouped: dict[UUID, list[StoredChunkState]] = {
            document_id: [] for document_id in unique_ids
        }
        if not unique_ids:
            return grouped
        ids_parameter = bindparam(
            "chunk_state_document_ids",
            value=unique_ids,
            type_=ARRAY(PG_UUID(as_uuid=True)),
        )
        rows = await self._session.execute(
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.chunk_index,
                Chunk.content_hash,
                Chunk.deleted_at,
            )
            .where(Chunk.document_id == any_(ids_parameter))
            .order_by(Chunk.document_id, Chunk.chunk_index)
        )
        for row in rows:
            grouped.setdefault(row.document_id, []).append(
                StoredChunkState(
                    id=row.id,
                    document_id=row.document_id,
                    chunk_index=row.chunk_index,
                    content_hash=row.content_hash,
                    deleted_at=row.deleted_at,
                )
            )
        return grouped

    async def list_active_for_document(self, document_id: UUID) -> list[Chunk]:
        """Load active chunks in source order with one query."""
        result = await self._session.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id, Chunk.deleted_at.is_(None))
            .order_by(Chunk.chunk_index)
        )
        return list(result)

    async def list_needing_embedding(
        self,
        *,
        model: str,
        dimension: int,
        limit: int,
    ) -> list[PendingEmbeddingChunk]:
        """Read a bounded active batch without holding a model-inference transaction."""
        rows = await self._session.execute(
            select(Chunk.id, Chunk.content_hash, Chunk.contextualized_content)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.deleted_at.is_(None),
                Document.deleted_at.is_(None),
                Document.status == "active",
                or_(
                    Chunk.embedding.is_(None),
                    Chunk.embedding_model.is_distinct_from(model),
                    Chunk.embedding_dimension.is_distinct_from(dimension),
                ),
            )
            .order_by(Chunk.updated_at, Chunk.id)
            .limit(limit)
        )
        return [
            PendingEmbeddingChunk(
                id=row.id,
                content_hash=row.content_hash,
                contextualized_content=row.contextualized_content,
            )
            for row in rows
        ]

    async def update_embeddings(
        self,
        records: Sequence[ChunkEmbeddingUpdate],
        *,
        updated_at: datetime,
    ) -> int:
        """Write a batch iff each chunk still contains the text that was embedded."""
        if not records:
            return 0
        table = cast(Table, Chunk.__table__)
        statement = (
            update(table)
            .where(
                table.c.id == bindparam("p_id", type_=table.c.id.type),
                table.c.content_hash
                == bindparam("p_content_hash", type_=table.c.content_hash.type),
                table.c.deleted_at.is_(None),
            )
            .values(
                embedding=bindparam("p_embedding", type_=table.c.embedding.type),
                embedding_model=bindparam("p_embedding_model"),
                embedding_dimension=bindparam("p_embedding_dimension"),
                updated_at=bindparam("p_updated_at"),
            )
        )
        affected = 0
        for batch in database_batches(records):
            result = await self._session.execute(
                statement,
                [
                    {
                        "p_id": record.id,
                        "p_content_hash": record.content_hash,
                        "p_embedding": record.embedding,
                        "p_embedding_model": record.model,
                        "p_embedding_dimension": record.dimension,
                        "p_updated_at": updated_at,
                    }
                    for record in batch
                ],
            )
            affected += _non_negative_rowcount(result)
        return affected

    async def upsert_many(
        self,
        records: Sequence[ChunkUpsert],
        *,
        written_at: datetime,
    ) -> list[Chunk]:
        """Reindex a document batch by stable UUID and preserve reusable embeddings.

        Existing indexes move to a transaction-local negative namespace first. This
        avoids unique-key conflicts when an insertion shifts following chunks while UUIDs,
        citations, parent links, and unchanged vectors remain attached to their content.
        """
        if not records:
            return []
        desired_ids = [record.id for record in records]
        existing_ids: set[UUID] = set()
        for id_batch in database_batches(desired_ids):
            existing_ids.update(
                await self._session.scalars(select(Chunk.id).where(Chunk.id.in_(id_batch)))
            )
        document_ids = list(dict.fromkeys(record.document_id for record in records))
        for document_id_batch in database_batches(document_ids):
            ranked = (
                select(
                    Chunk.id.label("chunk_id"),
                    func.row_number()
                    .over(partition_by=Chunk.document_id, order_by=Chunk.id)
                    .label("ordinal"),
                )
                .where(Chunk.document_id.in_(document_id_batch))
                .cte("ranked_chunks")
            )
            await self._session.execute(
                update(Chunk)
                .where(Chunk.id == ranked.c.chunk_id)
                .values(chunk_index=-(1_000_000 + ranked.c.ordinal))
            )

        new_records = [record for record in records if record.id not in existing_ids]
        for record_batch in database_batches(new_records):
            # Parent rows can land in later batches. Insert every new row without its
            # parent first, then attach links after all referenced UUIDs exist.
            await self._session.execute(
                insert(Chunk).values(
                    [
                        _chunk_values(record, written_at=written_at, include_parent=False)
                        for record in record_batch
                    ]
                )
            )

        existing_records = [record for record in records if record.id in existing_ids]
        if existing_records:
            await self._update_existing(existing_records, written_at=written_at)
        await self._update_parent_links(new_records)

        persisted: list[Chunk] = []
        for id_batch in database_batches(desired_ids):
            result = await self._session.scalars(select(Chunk).where(Chunk.id.in_(id_batch)))
            persisted.extend(result)
        persisted.sort(key=lambda chunk: chunk.chunk_index)
        return persisted

    async def _update_existing(
        self,
        records: Sequence[ChunkUpsert],
        *,
        written_at: datetime,
    ) -> None:
        table = cast(Table, Chunk.__table__)
        incoming_hash = bindparam("p_content_hash", type_=table.c.content_hash.type)
        incoming_embedding = bindparam("p_embedding", type_=table.c.embedding.type)
        keep_embedding = and_(
            table.c.content_hash == incoming_hash,
            incoming_embedding.is_(None),
        )
        statement = (
            update(table)
            .where(table.c.id == bindparam("p_id", type_=table.c.id.type))
            .values(
                document_id=bindparam("p_document_id", type_=table.c.document_id.type),
                chunk_index=bindparam("p_chunk_index"),
                parent_chunk_id=bindparam("p_parent_chunk_id", type_=table.c.id.type),
                document_title=bindparam("p_document_title"),
                heading_path=bindparam("p_heading_path", type_=table.c.heading_path.type),
                content=bindparam("p_content"),
                contextualized_content=bindparam("p_contextualized_content"),
                token_count=bindparam("p_token_count"),
                content_hash=incoming_hash,
                start_offset=bindparam("p_start_offset"),
                end_offset=bindparam("p_end_offset"),
                start_line=bindparam("p_start_line"),
                end_line=bindparam("p_end_line"),
                metadata=bindparam("p_metadata", type_=table.c.metadata.type),
                embedding=case((keep_embedding, table.c.embedding), else_=incoming_embedding),
                embedding_model=case(
                    (keep_embedding, table.c.embedding_model),
                    else_=bindparam("p_embedding_model"),
                ),
                embedding_dimension=case(
                    (keep_embedding, table.c.embedding_dimension),
                    else_=bindparam("p_embedding_dimension"),
                ),
                deleted_at=None,
                updated_at=bindparam("p_updated_at"),
            )
        )
        for batch in database_batches(records):
            await self._session.execute(
                statement,
                [_chunk_update_values(record, written_at=written_at) for record in batch],
            )

    async def _update_parent_links(self, records: Sequence[ChunkUpsert]) -> None:
        linked_records = [record for record in records if record.parent_chunk_id is not None]
        if not linked_records:
            return
        table = cast(Table, Chunk.__table__)
        statement = (
            update(table)
            .where(table.c.id == bindparam("p_id", type_=table.c.id.type))
            .values(
                parent_chunk_id=bindparam("p_parent_chunk_id", type_=table.c.id.type),
            )
        )
        for batch in database_batches(linked_records):
            await self._session.execute(
                statement,
                [
                    {
                        "p_id": record.id,
                        "p_parent_chunk_id": record.parent_chunk_id,
                    }
                    for record in batch
                ],
            )

    async def reactivate(self, chunk_ids: Sequence[UUID], *, restored_at: datetime) -> int:
        """Restore unchanged chunks without regenerating their embeddings."""
        if not chunk_ids:
            return 0
        affected = 0
        for batch in database_batches(chunk_ids):
            result = await self._session.execute(
                update(Chunk)
                .where(Chunk.id.in_(batch), Chunk.deleted_at.is_not(None))
                .values(deleted_at=None, updated_at=restored_at)
            )
            affected += _non_negative_rowcount(result)
        return affected

    async def soft_delete(self, chunk_ids: Sequence[UUID], *, deleted_at: datetime) -> int:
        """Soft-delete removed chunks in bounded statements."""
        if not chunk_ids:
            return 0
        affected = 0
        for batch in database_batches(chunk_ids):
            result = await self._session.execute(
                update(Chunk)
                .where(Chunk.id.in_(batch), Chunk.deleted_at.is_(None))
                .values(deleted_at=deleted_at, updated_at=deleted_at)
            )
            affected += _non_negative_rowcount(result)
        return affected

    async def soft_delete_for_documents(
        self,
        document_ids: Sequence[UUID],
        *,
        deleted_at: datetime,
    ) -> int:
        """Exclude every active chunk belonging to soft-deleted documents."""
        if not document_ids:
            return 0
        affected = 0
        for batch in database_batches(document_ids):
            result = await self._session.execute(
                update(Chunk)
                .where(Chunk.document_id.in_(batch), Chunk.deleted_at.is_(None))
                .values(deleted_at=deleted_at, updated_at=deleted_at)
            )
            affected += _non_negative_rowcount(result)
        return affected

    async def soft_delete_missing(
        self,
        document_id: UUID,
        retained_indexes: Sequence[int],
        *,
        deleted_at: datetime,
    ) -> list[UUID]:
        """Soft-delete chunks absent from the document's latest complete parse."""
        predicates = [Chunk.document_id == document_id, Chunk.deleted_at.is_(None)]
        if retained_indexes:
            retained = bindparam(
                "retained_chunk_indexes",
                value=list(retained_indexes),
                type_=ARRAY(Integer()),
            )
            predicates.append(Chunk.chunk_index != all_(retained))
        statement = (
            update(Chunk)
            .where(*predicates)
            .values(deleted_at=deleted_at, updated_at=deleted_at)
            .returning(Chunk.id)
        )
        result = await self._session.scalars(statement)
        return list(result)


def _chunk_values(
    record: ChunkUpsert,
    *,
    written_at: datetime,
    include_parent: bool = True,
) -> dict[str, Any]:
    return {
        "id": record.id,
        "document_id": record.document_id,
        "chunk_index": record.chunk_index,
        "parent_chunk_id": record.parent_chunk_id if include_parent else None,
        "document_title": record.document_title,
        "heading_path": record.heading_path,
        "content": record.content,
        "contextualized_content": record.contextualized_content,
        "token_count": record.token_count,
        "content_hash": record.content_hash,
        "start_offset": record.start_offset,
        "end_offset": record.end_offset,
        "start_line": record.start_line,
        "end_line": record.end_line,
        "metadata_": record.metadata,
        "embedding": record.embedding,
        "embedding_model": record.embedding_model,
        "embedding_dimension": record.embedding_dimension,
        "deleted_at": None,
        "created_at": written_at,
        "updated_at": written_at,
    }


def _chunk_update_values(record: ChunkUpsert, *, written_at: datetime) -> dict[str, Any]:
    return {
        "p_id": record.id,
        "p_document_id": record.document_id,
        "p_chunk_index": record.chunk_index,
        "p_parent_chunk_id": record.parent_chunk_id,
        "p_document_title": record.document_title,
        "p_heading_path": record.heading_path,
        "p_content": record.content,
        "p_contextualized_content": record.contextualized_content,
        "p_token_count": record.token_count,
        "p_content_hash": record.content_hash,
        "p_start_offset": record.start_offset,
        "p_end_offset": record.end_offset,
        "p_start_line": record.start_line,
        "p_end_line": record.end_line,
        "p_metadata": record.metadata,
        "p_embedding": record.embedding,
        "p_embedding_model": record.embedding_model,
        "p_embedding_dimension": record.embedding_dimension,
        "p_updated_at": written_at,
    }


def _non_negative_rowcount(result: Any) -> int:
    return max(int(cast(CursorResult[Any], result).rowcount or 0), 0)
