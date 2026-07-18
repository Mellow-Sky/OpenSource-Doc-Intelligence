"""Async persistence for source definitions and durable sync cursors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.ingestion import SyncCursor
from app.db.models.source_document import Source
from app.ingestion.incremental.models import CursorCheckpoint


class SourceRepository:
    """Persist sources and checkpoints without owning the surrounding transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, source_id: UUID) -> Source | None:
        """Return a source by primary key."""
        return await self._session.get(Source, source_id)

    async def list_enabled(self) -> list[Source]:
        """Return enabled sources in deterministic creation order."""
        result = await self._session.scalars(
            select(Source).where(Source.enabled.is_(True)).order_by(Source.created_at, Source.id)
        )
        return list(result)

    async def get_cursor(self, source_id: UUID, cursor_type: str) -> SyncCursor | None:
        """Load one source-specific incremental checkpoint."""
        return cast(
            SyncCursor | None,
            await self._session.scalar(
                select(SyncCursor).where(
                    SyncCursor.source_id == source_id,
                    SyncCursor.cursor_type == cursor_type,
                )
            ),
        )

    async def acquire_sync_lock(self, source_id: UUID) -> None:
        """Serialize writes for one source for the surrounding transaction lifetime."""
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(CAST(:source_id AS text), 0))"),
            {"source_id": str(source_id)},
        )

    async def get_cursors(
        self,
        source_id: UUID,
        cursor_types: Sequence[str] | None = None,
    ) -> list[SyncCursor]:
        """Load all requested checkpoints for a source."""
        statement = select(SyncCursor).where(SyncCursor.source_id == source_id)
        if cursor_types is not None:
            if not cursor_types:
                return []
            statement = statement.where(SyncCursor.cursor_type.in_(cursor_types))
        result = await self._session.scalars(statement.order_by(SyncCursor.cursor_type))
        return list(result)

    async def upsert_cursor(self, checkpoint: CursorCheckpoint) -> SyncCursor:
        """Atomically insert or advance a checkpoint.

        Callers should execute this in the same transaction as successful document and
        chunk writes. A failed transaction therefore cannot advance the upstream cursor.
        """
        values = {
            "id": uuid4(),
            "source_id": checkpoint.source_id,
            "cursor_type": checkpoint.cursor_type,
            "cursor_value": checkpoint.cursor_value,
            "metadata_": checkpoint.metadata,
        }
        insert_statement = insert(SyncCursor).values(values)
        statement = insert_statement.on_conflict_do_update(
            constraint="uq_sync_cursors_source_type",
            set_={
                "cursor_value": insert_statement.excluded.cursor_value,
                "metadata": insert_statement.excluded.metadata,
                "updated_at": func.now(),
            },
        ).returning(SyncCursor)
        result = await self._session.execute(statement)
        return result.scalar_one()
