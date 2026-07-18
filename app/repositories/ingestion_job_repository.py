"""Reliable queue persistence for asynchronous ingestion workers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.ingestion import IngestionJob
from app.ingestion.incremental.models import JobStatus, SyncStats
from app.repositories.queue_gate import INGESTION_QUEUE, acquire_queue_advisory_lock


class IngestionJobRepository:
    """Manage ingestion jobs using leases and atomic PostgreSQL claims.

    Methods never commit. A worker must commit the transaction containing ``claim_next``
    before doing slow work. ``started_at`` is the lease token passed to later mutations;
    an expired worker therefore cannot overwrite a job claimed by another worker.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, job_id: UUID) -> IngestionJob | None:
        """Return a durable job by primary key."""
        return await self._session.get(IngestionJob, job_id)

    async def get_by_idempotency_key(self, idempotency_key: str) -> IngestionJob | None:
        """Return an existing request so a capacity limit never rejects a replay."""

        return cast(
            IngestionJob | None,
            await self._session.scalar(
                select(IngestionJob).where(IngestionJob.idempotency_key == idempotency_key)
            ),
        )

    async def acquire_queue_lock(self) -> None:
        """Serialize global ingestion queue admission and claim decisions."""

        await acquire_queue_advisory_lock(self._session, INGESTION_QUEUE)

    async def outstanding_count(self) -> int:
        """Count durable jobs that still consume queue capacity."""

        value = await self._session.scalar(
            select(func.count(IngestionJob.id)).where(
                IngestionJob.status.in_((JobStatus.PENDING.value, JobStatus.RUNNING.value))
            )
        )
        return int(value or 0)

    async def enqueue(
        self,
        *,
        source_id: UUID,
        idempotency_key: str,
        options: dict[str, Any] | None = None,
        requested_by: str | None = None,
    ) -> IngestionJob:
        """Create one pending job or return the existing idempotent request."""
        statement = (
            insert(IngestionJob)
            .values(
                id=uuid4(),
                source_id=source_id,
                idempotency_key=idempotency_key,
                status=JobStatus.PENDING.value,
                requested_by=requested_by,
                options=options or {},
                stats=SyncStats().model_dump(),
            )
            .on_conflict_do_nothing(index_elements=[IngestionJob.idempotency_key])
            .returning(IngestionJob)
        )
        result = await self._session.execute(statement)
        job = result.scalar_one_or_none()
        if job is not None:
            return job
        existing = await self._session.scalar(
            select(IngestionJob).where(IngestionJob.idempotency_key == idempotency_key)
        )
        if existing is None:
            raise RuntimeError("idempotent ingestion job disappeared during enqueue")
        return existing

    async def claim_next(
        self,
        *,
        claimed_at: datetime,
        stale_before: datetime,
        max_running: int | None = None,
    ) -> IngestionJob | None:
        """Atomically lease one pending or abandoned job with SKIP LOCKED."""
        if max_running is not None:
            if max_running < 1:
                raise ValueError("max_running must be positive")
            await self.acquire_queue_lock()
            live_running = await self._session.scalar(
                select(func.count(IngestionJob.id)).where(
                    IngestionJob.status == JobStatus.RUNNING.value,
                    IngestionJob.heartbeat_at.is_not(None),
                    IngestionJob.heartbeat_at >= stale_before,
                )
            )
            if int(live_running or 0) >= max_running:
                return None
        stale_running = and_(
            IngestionJob.status == JobStatus.RUNNING.value,
            or_(
                IngestionJob.heartbeat_at.is_(None),
                IngestionJob.heartbeat_at < stale_before,
            ),
        )
        candidate = (
            select(IngestionJob.id)
            .where(
                or_(
                    IngestionJob.status == JobStatus.PENDING.value,
                    stale_running,
                )
            )
            .order_by(
                # Revoke an expired lease before admitting fresh work. Otherwise
                # an old worker could still run beside a newly claimed pending job.
                case((stale_running, 0), else_=1),
                IngestionJob.created_at,
                IngestionJob.id,
            )
            .with_for_update(skip_locked=True)
            .limit(1)
            .scalar_subquery()
        )
        statement = (
            update(IngestionJob)
            .where(IngestionJob.id == candidate)
            .values(
                status=JobStatus.RUNNING.value,
                started_at=claimed_at,
                heartbeat_at=claimed_at,
                finished_at=None,
                error=None,
                updated_at=claimed_at,
            )
            .returning(IngestionJob)
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def heartbeat(
        self,
        job_id: UUID,
        *,
        lease_started_at: datetime,
        heartbeat_at: datetime,
    ) -> bool:
        """Extend a live lease using compare-and-swap ownership."""
        result = await self._session.execute(
            update(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.status == JobStatus.RUNNING.value,
                IngestionJob.started_at == lease_started_at,
            )
            .values(heartbeat_at=heartbeat_at, updated_at=heartbeat_at)
        )
        return bool(cast(CursorResult[Any], result).rowcount)

    async def mark_succeeded(
        self,
        job_id: UUID,
        *,
        lease_started_at: datetime,
        stats: SyncStats,
        finished_at: datetime,
    ) -> bool:
        """Complete the current lease; stale workers fail the CAS update."""
        result = await self._session.execute(
            update(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.status == JobStatus.RUNNING.value,
                IngestionJob.started_at == lease_started_at,
            )
            .values(
                status=JobStatus.SUCCEEDED.value,
                stats=stats.model_dump(),
                error=None,
                heartbeat_at=finished_at,
                finished_at=finished_at,
                updated_at=finished_at,
            )
        )
        return bool(cast(CursorResult[Any], result).rowcount)

    async def mark_failed(
        self,
        job_id: UUID,
        *,
        lease_started_at: datetime,
        error: str,
        stats: SyncStats,
        finished_at: datetime,
    ) -> bool:
        """Persist a bounded failure description and release the current lease."""
        result = await self._session.execute(
            update(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.status == JobStatus.RUNNING.value,
                IngestionJob.started_at == lease_started_at,
            )
            .values(
                status=JobStatus.FAILED.value,
                stats=stats.model_dump(),
                error=error,
                heartbeat_at=finished_at,
                finished_at=finished_at,
                updated_at=finished_at,
            )
        )
        return bool(cast(CursorResult[Any], result).rowcount)

    async def retry_failed(self, job_id: UUID, *, queued_at: datetime) -> bool:
        """Explicitly requeue a failed job without changing its idempotency key."""
        result = await self._session.execute(
            update(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.status == JobStatus.FAILED.value,
            )
            .values(
                status=JobStatus.PENDING.value,
                stats=SyncStats().model_dump(),
                error=None,
                started_at=None,
                heartbeat_at=None,
                finished_at=None,
                updated_at=queued_at,
            )
        )
        return bool(cast(CursorResult[Any], result).rowcount)

    async def recover_stale(self, *, stale_before: datetime, recovered_at: datetime) -> list[UUID]:
        """Return abandoned running jobs to pending for explicit recovery sweeps."""
        statement = (
            update(IngestionJob)
            .where(
                IngestionJob.status == JobStatus.RUNNING.value,
                or_(
                    IngestionJob.heartbeat_at.is_(None),
                    IngestionJob.heartbeat_at < stale_before,
                ),
            )
            .values(
                status=JobStatus.PENDING.value,
                started_at=None,
                heartbeat_at=None,
                finished_at=None,
                updated_at=recovered_at,
            )
            .returning(IngestionJob.id)
        )
        result = await self._session.scalars(statement)
        return list(result)
