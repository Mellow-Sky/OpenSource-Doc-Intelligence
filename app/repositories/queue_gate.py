"""PostgreSQL transaction gates shared by durable queue producers and workers."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

INGESTION_QUEUE = "opensource-doc-intelligence:ingestion-queue"
EVALUATION_QUEUE = "opensource-doc-intelligence:evaluation-queue"


async def acquire_queue_advisory_lock(session: AsyncSession, queue_name: str) -> None:
    """Serialize a queue's count-and-mutate decisions across all processes.

    The lock is transaction-scoped, so callers cannot accidentally leak it when a
    request is cancelled or a transaction rolls back.
    """

    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(CAST(:queue_name AS text), 0))"),
        {"queue_name": queue_name},
    )
