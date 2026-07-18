"""Batch-oriented async persistence for documents and immutable versions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import Text, all_, bindparam, case, func, or_, select, update
from sqlalchemy.dialects.postgresql import ARRAY, insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.source_document import Chunk, Document, DocumentVersion, Source
from app.ingestion.deduplication import (
    DeduplicationCandidate,
    normalized_content_hash,
    simhash64,
)
from app.ingestion.incremental.models import StoredDocumentState
from app.repositories.batching import database_batches


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentUpsert:
    """Complete current-state row used by a batch document upsert."""

    source_id: UUID
    external_id: str
    document_type: str
    title: str
    content_hash: str
    canonical_url: str | None = None
    repository_path: str | None = None
    source_version: str | None = None
    language: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentVersionAppend:
    """Immutable source and parsed content snapshot."""

    document_id: UUID
    content_hash: str
    raw_content: str
    parsed_content: str
    source_version: str | None = None
    id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentListFilters:
    """Bound values accepted by the document collection read model."""

    source_ids: Sequence[UUID] = ()
    source_types: Sequence[str] = ()
    document_types: Sequence[str] = ()
    versions: Sequence[str] = ()
    languages: Sequence[str] = ()
    statuses: Sequence[str] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    search: str | None = None
    include_deleted: bool = False


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """A stable document page and total number of matching rows."""

    items: list[Document]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class DocumentDetailRecord:
    """One document plus source identity and active chunk cardinality."""

    document: Document
    source_name: str
    source_type: str
    active_chunk_count: int


@dataclass(frozen=True, slots=True)
class StoredDeduplicationFingerprint:
    """One active persisted document represented without loading its full history."""

    candidate: DeduplicationCandidate
    exact_hash: str
    simhash: int


class DocumentRepository:
    """Persist document batches without committing per document."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, document_id: UUID, *, include_deleted: bool = False) -> Document | None:
        """Return one document, excluding soft-deleted state by default."""
        statement = select(Document).where(Document.id == document_id)
        if not include_deleted:
            statement = statement.where(Document.deleted_at.is_(None))
        return cast(Document | None, await self._session.scalar(statement))

    async def get_detail(
        self,
        document_id: UUID,
        *,
        include_deleted: bool = False,
    ) -> DocumentDetailRecord | None:
        """Return document/source details and chunk count in one database query."""
        active_chunks = (
            select(func.count(Chunk.id))
            .where(Chunk.document_id == Document.id, Chunk.deleted_at.is_(None))
            .correlate(Document)
            .scalar_subquery()
        )
        statement = (
            select(
                Document,
                Source.name.label("source_name"),
                Source.source_type.label("source_type"),
                active_chunks.label("active_chunk_count"),
            )
            .join(Source, Source.id == Document.source_id)
            .where(Document.id == document_id)
        )
        if not include_deleted:
            statement = statement.where(Document.deleted_at.is_(None))
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None
        return DocumentDetailRecord(
            document=row[0],
            source_name=str(row.source_name),
            source_type=str(row.source_type),
            active_chunk_count=int(row.active_chunk_count),
        )

    async def list_page(
        self,
        *,
        filters: DocumentListFilters | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> DocumentPage:
        """List documents with bounded pagination and parameterized filters."""
        _validate_page(limit, offset)
        predicates = _document_predicates(filters or DocumentListFilters())
        total = int(
            await self._session.scalar(
                select(func.count(Document.id))
                .select_from(Document)
                .join(Source, Source.id == Document.source_id)
                .where(*predicates)
            )
            or 0
        )
        rows = await self._session.scalars(
            select(Document)
            .join(Source, Source.id == Document.source_id)
            .where(*predicates)
            .order_by(Document.updated_at.desc(), Document.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return DocumentPage(list(rows), total, limit, offset)

    async def list_versions(
        self,
        document_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DocumentVersion]:
        """List immutable versions newest-first without loading large relations."""
        _validate_page(limit, offset)
        rows = await self._session.scalars(
            select(DocumentVersion)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                DocumentVersion.document_id == document_id,
                Document.deleted_at.is_(None),
            )
            .order_by(DocumentVersion.created_at.desc(), DocumentVersion.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(rows)

    async def list_states_for_source(self, source_id: UUID) -> list[StoredDocumentState]:
        """Load the minimal state required by the pure incremental planner."""
        rows = await self._session.execute(
            select(
                Document.id,
                Document.source_id,
                Document.external_id,
                Document.content_hash,
                Document.deleted_at,
            ).where(Document.source_id == source_id)
        )
        return [
            StoredDocumentState(
                id=row.id,
                source_id=row.source_id,
                external_id=row.external_id,
                content_hash=row.content_hash,
                deleted_at=row.deleted_at,
            )
            for row in rows
        ]

    async def list_deduplication_fingerprints(
        self,
        source_id: UUID,
        *,
        exclude_external_ids: Sequence[str] = (),
    ) -> list[StoredDeduplicationFingerprint]:
        """Load active cross-run fingerprints in one query.

        New ingestion writes store fingerprints in document metadata. For rows created
        before that metadata existed, a correlated scalar subquery returns only the
        newest parsed version so compatibility does not introduce an N+1 query.
        """

        latest_parsed_content = (
            select(DocumentVersion.parsed_content)
            .where(DocumentVersion.document_id == Document.id)
            .order_by(DocumentVersion.created_at.desc(), DocumentVersion.id.desc())
            .limit(1)
            .correlate(Document)
            .scalar_subquery()
        )
        predicates = [
            Document.source_id == source_id,
            Document.status == "active",
            Document.deleted_at.is_(None),
        ]
        if exclude_external_ids:
            excluded_external_ids = bindparam(
                "excluded_external_ids",
                value=list(exclude_external_ids),
                type_=ARRAY(Text()),
            )
            predicates.append(Document.external_id != all_(excluded_external_ids))
        rows = await self._session.execute(
            select(
                Document.external_id,
                Document.source_version,
                Document.document_type,
                Document.metadata_.label("document_metadata"),
                Source.source_type,
                latest_parsed_content.label("latest_parsed_content"),
            )
            .join(Source, Source.id == Document.source_id)
            .where(*predicates)
            .order_by(Document.external_id)
        )

        fingerprints: list[StoredDeduplicationFingerprint] = []
        for row in rows:
            metadata = row.document_metadata if isinstance(row.document_metadata, dict) else {}
            exact_hash, simhash = _metadata_fingerprint(metadata)
            if exact_hash is None or simhash is None:
                parsed_content = row.latest_parsed_content
                if not isinstance(parsed_content, str):
                    continue
                exact_hash = normalized_content_hash(parsed_content)
                simhash = simhash64(parsed_content)
            fingerprints.append(
                StoredDeduplicationFingerprint(
                    candidate=DeduplicationCandidate(
                        external_id=str(row.external_id),
                        content="",
                        source_type=str(row.source_type),
                        source_version=(
                            str(row.source_version) if row.source_version is not None else None
                        ),
                        document_type=str(row.document_type),
                        metadata=metadata,
                    ),
                    exact_hash=exact_hash,
                    simhash=simhash,
                )
            )
        return fingerprints

    async def get_by_external_ids(
        self,
        source_id: UUID,
        external_ids: Sequence[str],
        *,
        include_deleted: bool = True,
    ) -> list[Document]:
        """Fetch a source-scoped document batch with one parameterized query."""
        if not external_ids:
            return []
        statement = select(Document).where(
            Document.source_id == source_id,
            Document.external_id.in_(external_ids),
        )
        if not include_deleted:
            statement = statement.where(Document.deleted_at.is_(None))
        result = await self._session.scalars(statement)
        return list(result)

    async def upsert_many(
        self,
        records: Sequence[DocumentUpsert],
        *,
        seen_at: datetime,
    ) -> list[Document]:
        """Insert or update a document batch and reactivate matching identities.

        An unchanged content hash preserves ``indexed_at``. A changed hash clears it so
        downstream indexing cannot mistake stale chunks for a completed index.
        """
        if not records:
            return []
        documents: list[Document] = []
        for batch in database_batches(records):
            insert_statement = insert(Document).values(
                [_document_values(record, seen_at=seen_at, duplicate=False) for record in batch]
            )
            unchanged_hash = Document.content_hash == insert_statement.excluded.content_hash
            statement = insert_statement.on_conflict_do_update(
                constraint="uq_documents_source_external",
                set_={
                    "document_type": insert_statement.excluded.document_type,
                    "title": insert_statement.excluded.title,
                    "canonical_url": insert_statement.excluded.canonical_url,
                    "repository_path": insert_statement.excluded.repository_path,
                    "source_version": insert_statement.excluded.source_version,
                    "language": insert_statement.excluded.language,
                    "content_hash": insert_statement.excluded.content_hash,
                    "metadata": insert_statement.excluded.metadata,
                    "status": "active",
                    "last_seen_at": seen_at,
                    "deleted_at": None,
                    "indexed_at": case((unchanged_hash, Document.indexed_at), else_=None),
                    "updated_at": seen_at,
                },
            ).returning(Document)
            result = await self._session.execute(statement)
            documents.extend(result.scalars())
        return documents

    async def upsert_duplicates(
        self,
        records: Sequence[DocumentUpsert],
        *,
        seen_at: datetime,
    ) -> list[Document]:
        """Persist skipped duplicate identities while excluding them from retrieval.

        Keeping a logical row makes the decision auditable and prevents a document
        that was previously active from retaining stale chunks when its new content
        is rejected as a duplicate. A later non-duplicate scan reactivates the same
        ``source_id + external_id`` identity through :meth:`upsert_many`.
        """
        if not records:
            return []
        documents: list[Document] = []
        for batch in database_batches(records):
            insert_statement = insert(Document).values(
                [_document_values(record, seen_at=seen_at, duplicate=True) for record in batch]
            )
            statement = insert_statement.on_conflict_do_update(
                constraint="uq_documents_source_external",
                set_={
                    "document_type": insert_statement.excluded.document_type,
                    "title": insert_statement.excluded.title,
                    "canonical_url": insert_statement.excluded.canonical_url,
                    "repository_path": insert_statement.excluded.repository_path,
                    "source_version": insert_statement.excluded.source_version,
                    "language": insert_statement.excluded.language,
                    "content_hash": insert_statement.excluded.content_hash,
                    "metadata": insert_statement.excluded.metadata,
                    "status": "duplicate",
                    "last_seen_at": seen_at,
                    "indexed_at": None,
                    "deleted_at": seen_at,
                    "updated_at": seen_at,
                },
            ).returning(Document)
            result = await self._session.execute(statement)
            documents.extend(result.scalars())
        return documents

    async def append_versions(
        self,
        versions: Sequence[DocumentVersionAppend],
    ) -> int:
        """Append new content snapshots, ignoring idempotent hash conflicts."""
        if not versions:
            return 0
        appended = 0
        for batch in database_batches(versions):
            values = [
                {
                    "id": version.id,
                    "document_id": version.document_id,
                    "source_version": version.source_version,
                    "content_hash": version.content_hash,
                    "raw_content": version.raw_content,
                    "parsed_content": version.parsed_content,
                }
                for version in batch
            ]
            statement = (
                insert(DocumentVersion)
                .values(values)
                .on_conflict_do_nothing(constraint="uq_document_versions_document_hash")
                .returning(DocumentVersion.id)
            )
            result = await self._session.scalars(statement)
            appended += len(list(result))
        return appended

    async def touch_seen(self, document_ids: Sequence[UUID], *, seen_at: datetime) -> int:
        """Advance last-seen time for unchanged documents in one statement."""
        if not document_ids:
            return 0
        affected = 0
        for batch in database_batches(document_ids):
            result = await self._session.execute(
                update(Document)
                .where(Document.id.in_(batch), Document.deleted_at.is_(None))
                .values(last_seen_at=seen_at, updated_at=seen_at)
            )
            affected += _non_negative_rowcount(result)
        return affected

    async def mark_indexed(self, document_ids: Sequence[UUID], *, indexed_at: datetime) -> int:
        """Mark successfully indexed active documents as complete in one batch."""
        if not document_ids:
            return 0
        affected = 0
        for batch in database_batches(document_ids):
            result = await self._session.execute(
                update(Document)
                .where(Document.id.in_(batch), Document.deleted_at.is_(None))
                .values(indexed_at=indexed_at, updated_at=indexed_at)
            )
            affected += _non_negative_rowcount(result)
        return affected

    async def soft_delete(self, document_ids: Sequence[UUID], *, deleted_at: datetime) -> int:
        """Soft-delete an explicit batch while leaving history auditable."""
        if not document_ids:
            return 0
        affected = 0
        for batch in database_batches(document_ids):
            result = await self._session.execute(
                update(Document)
                .where(Document.id.in_(batch), Document.deleted_at.is_(None))
                .values(status="deleted", deleted_at=deleted_at, updated_at=deleted_at)
            )
            affected += _non_negative_rowcount(result)
        return affected

    async def soft_delete_missing(
        self,
        source_id: UUID,
        seen_external_ids: Sequence[str],
        *,
        deleted_at: datetime,
    ) -> list[UUID]:
        """Soft-delete active source documents absent from a complete snapshot."""
        predicates = [Document.source_id == source_id, Document.deleted_at.is_(None)]
        if seen_external_ids:
            seen_ids = bindparam(
                "seen_external_ids",
                value=list(seen_external_ids),
                type_=ARRAY(Text()),
            )
            predicates.append(Document.external_id != all_(seen_ids))
        statement = (
            update(Document)
            .where(*predicates)
            .values(status="deleted", deleted_at=deleted_at, updated_at=deleted_at)
            .returning(Document.id)
        )
        result = await self._session.scalars(statement)
        return list(result)


def _document_values(
    record: DocumentUpsert,
    *,
    seen_at: datetime,
    duplicate: bool,
) -> dict[str, Any]:
    return {
        "id": record.id,
        "source_id": record.source_id,
        "external_id": record.external_id,
        "document_type": record.document_type,
        "title": record.title,
        "canonical_url": record.canonical_url,
        "repository_path": record.repository_path,
        "source_version": record.source_version,
        "language": record.language,
        "content_hash": record.content_hash,
        "metadata_": record.metadata,
        "status": "duplicate" if duplicate else "active",
        "first_seen_at": seen_at,
        "last_seen_at": seen_at,
        "indexed_at": None,
        "deleted_at": seen_at if duplicate else None,
    }


def _non_negative_rowcount(result: Any) -> int:
    return max(int(cast(CursorResult[Any], result).rowcount or 0), 0)


def _document_predicates(filters: DocumentListFilters) -> list[Any]:
    predicates: list[Any] = []
    if not filters.include_deleted:
        predicates.append(Document.deleted_at.is_(None))
    if filters.source_ids:
        predicates.append(Document.source_id.in_(filters.source_ids))
    if filters.source_types:
        predicates.append(Source.source_type.in_(filters.source_types))
    if filters.document_types:
        predicates.append(Document.document_type.in_(filters.document_types))
    if filters.versions:
        predicates.append(Document.source_version.in_(filters.versions))
    if filters.languages:
        predicates.append(Document.language.in_(filters.languages))
    if filters.statuses:
        predicates.append(Document.status.in_(filters.statuses))
    if filters.metadata:
        predicates.append(Document.metadata_.contains(filters.metadata))
    if filters.search is not None:
        search = filters.search.strip()
        if not search:
            raise ValueError("search must not be blank")
        if len(search) > 1000:
            raise ValueError("search exceeds maximum length")
        pattern = f"%{search}%"
        predicates.append(
            or_(
                Document.title.ilike(pattern),
                Document.external_id.ilike(pattern),
                Document.repository_path.ilike(pattern),
            )
        )
    return predicates


def _metadata_fingerprint(metadata: dict[str, Any]) -> tuple[str | None, int | None]:
    value = metadata.get("deduplication_fingerprint")
    if not isinstance(value, dict):
        return None, None
    exact_hash = value.get("normalized_sha256")
    simhash = value.get("simhash64")
    if not (
        isinstance(exact_hash, str)
        and len(exact_hash) == 64
        and exact_hash == exact_hash.lower()
        and all(character in "0123456789abcdef" for character in exact_hash)
    ):
        return None, None
    if isinstance(simhash, bool) or not isinstance(simhash, int) or not 0 <= simhash < 2**64:
        return None, None
    return exact_hash, simhash


def _validate_page(limit: int, offset: int) -> None:
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")
    if offset < 0:
        raise ValueError("offset must be non-negative")
