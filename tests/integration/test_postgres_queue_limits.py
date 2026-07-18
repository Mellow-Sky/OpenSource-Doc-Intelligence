"""Real PostgreSQL concurrency checks for durable queue admission and claims."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.exceptions import RateLimitError
from app.db.models.evaluation import EvaluationRun
from app.db.models.ingestion import IngestionJob
from app.db.models.source_document import Source
from app.repositories.evaluation_repository import EvaluationRepository
from app.repositories.ingestion_job_repository import IngestionJobRepository
from app.schemas.ingestion import IngestionJobOptions
from app.services.ingestion_queue_service import IngestionQueueService

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is required for PostgreSQL queue integration tests",
    ),
]


@pytest.mark.asyncio
async def test_postgres_ingestion_admission_gate_is_atomic_across_sessions() -> None:
    """Two API replicas cannot both pass a one-slot global queue capacity check."""

    assert TEST_DATABASE_URL is not None
    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    source_id = uuid4()
    async with factory.begin() as session:
        session.add(
            Source(
                id=source_id,
                name=f"queue-limit-source-{source_id}",
                source_type="github_repository",
                enabled=True,
            )
        )
    async with factory() as session:
        baseline = await IngestionJobRepository(session).outstanding_count()
    service = IngestionQueueService(factory, max_outstanding_jobs=baseline + 1)

    async def enqueue(key: str) -> IngestionJob:
        return await service.enqueue(
            source_id,
            options=IngestionJobOptions(),
            idempotency_key=key,
        )

    outcomes = await asyncio.gather(
        enqueue(f"queue-a-{source_id}"),
        enqueue(f"queue-b-{source_id}"),
        return_exceptions=True,
    )
    try:
        assert sum(isinstance(item, IngestionJob) for item in outcomes) == 1
        assert sum(isinstance(item, RateLimitError) for item in outcomes) == 1
    finally:
        async with factory.begin() as session:
            await session.execute(delete(IngestionJob).where(IngestionJob.source_id == source_id))
            await session.execute(delete(Source).where(Source.id == source_id))
        await engine.dispose()


@pytest.mark.asyncio
async def test_postgres_evaluation_running_gate_is_atomic_across_workers() -> None:
    """Two worker replicas respect the configured global run concurrency."""

    assert TEST_DATABASE_URL is not None
    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    marker = f"queue-claim-{uuid4()}"
    run_ids: list[UUID] = []
    queued_at = datetime.now(UTC) - timedelta(minutes=1)
    async with factory.begin() as session:
        repository = EvaluationRepository(session)
        for suffix in ("a", "b"):
            run = await repository.create_run(
                dataset_name=f"{marker}-{suffix}",
                config_snapshot={},
                queued_at=queued_at,
            )
            run_ids.append(run.id)

    async def claim() -> EvaluationRun | None:
        async with factory.begin() as session:
            return await EvaluationRepository(session).claim_next(
                claimed_at=datetime.now(UTC),
                stale_before=datetime.now(UTC) - timedelta(hours=1),
                max_running=1,
            )

    outcomes = await asyncio.gather(claim(), claim())
    try:
        assert sum(item is not None and item.id in run_ids for item in outcomes) == 1
        assert sum(item is None for item in outcomes) == 1
    finally:
        async with factory.begin() as session:
            await session.execute(delete(EvaluationRun).where(EvaluationRun.id.in_(run_ids)))
        await engine.dispose()
