"""Application service for non-blocking ingestion job submission and polling."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import DatabaseError, NotFoundError, RateLimitError, ValidationError
from app.db.models.ingestion import IngestionJob
from app.repositories.ingestion_job_repository import IngestionJobRepository
from app.repositories.source_repository import SourceRepository
from app.schemas.ingestion import IngestionJobOptions


class IngestionQueueService:
    """Validate a source and persist durable work without running the sync inline."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        max_outstanding_jobs: int = 100,
        retry_after_seconds: int = 5,
    ) -> None:
        if max_outstanding_jobs < 1 or retry_after_seconds < 1:
            raise ValueError("queue capacity and retry delay must be positive")
        self._session_factory = session_factory
        self._max_outstanding_jobs = max_outstanding_jobs
        self._retry_after_seconds = retry_after_seconds

    async def enqueue(
        self,
        source_id: UUID,
        *,
        options: IngestionJobOptions,
        idempotency_key: str | None = None,
        requested_by: str | None = None,
    ) -> IngestionJob:
        """Atomically enqueue one enabled source and return its durable job row."""
        key = idempotency_key or f"source-sync:{source_id}:{uuid4()}"
        serialized_options = options.model_dump(mode="json")
        try:
            async with self._session_factory.begin() as session:
                source = await SourceRepository(session).get(source_id)
                if source is None:
                    raise NotFoundError("Knowledge source was not found")
                if not source.enabled:
                    raise ValidationError("Knowledge source is disabled")

                repository = IngestionJobRepository(session)
                await repository.acquire_queue_lock()
                existing = await repository.get_by_idempotency_key(key)
                if existing is not None:
                    self._validate_idempotent_replay(
                        existing,
                        source_id=source_id,
                        options=serialized_options,
                    )
                    return existing
                if await repository.outstanding_count() >= self._max_outstanding_jobs:
                    raise RateLimitError(
                        "Ingestion queue capacity exceeded",
                        details={
                            "queue": "ingestion",
                            "limit": self._max_outstanding_jobs,
                        },
                        retry_after_seconds=self._retry_after_seconds,
                    )

                job = await repository.enqueue(
                    source_id=source_id,
                    idempotency_key=key,
                    options=serialized_options,
                    requested_by=requested_by,
                )
                self._validate_idempotent_replay(
                    job,
                    source_id=source_id,
                    options=serialized_options,
                )
                return job
        except (NotFoundError, RateLimitError, ValidationError):
            raise
        except SQLAlchemyError as exc:
            raise DatabaseError(
                "Unable to enqueue ingestion job",
                details={"operation": "ingestion_job_enqueue"},
            ) from exc

    async def get(self, job_id: UUID) -> IngestionJob:
        """Return one job for polling or raise the public not-found error."""
        try:
            async with self._session_factory() as session:
                job = await IngestionJobRepository(session).get(job_id)
                if job is None:
                    raise NotFoundError("Ingestion job was not found")
                return job
        except NotFoundError:
            raise
        except SQLAlchemyError as exc:
            raise DatabaseError(
                "Unable to read ingestion job",
                details={"operation": "ingestion_job_get"},
            ) from exc

    @staticmethod
    def _validate_idempotent_replay(
        job: IngestionJob,
        *,
        source_id: UUID,
        options: dict[str, Any],
    ) -> None:
        if job.source_id != source_id:
            raise ValidationError(
                "Idempotency key was already used for another source",
                details={"idempotency_key": job.idempotency_key},
            )
        if dict(job.options or {}) != options:
            raise ValidationError(
                "Idempotency key was already used with different options",
                details={"idempotency_key": job.idempotency_key},
            )
