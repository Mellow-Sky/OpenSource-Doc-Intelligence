"""Business-rule tests for atomic ingestion queue submission."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import DatabaseError, NotFoundError, RateLimitError, ValidationError
from app.db.models.ingestion import IngestionJob
from app.db.models.source_document import Source
from app.ingestion.incremental import SyncStats
from app.schemas.ingestion import IngestionJobOptions
from app.services import ingestion_queue_service as module
from app.services.ingestion_queue_service import IngestionQueueService


@dataclass
class _State:
    source: Source | None
    job: IngestionJob | None = None
    enqueue_calls: list[dict[str, Any]] = field(default_factory=list)
    database_failure: bool = False
    outstanding_count: int = 0
    queue_lock_calls: int = 0


class _FakeSession:
    def __init__(self, state: _State) -> None:
        self.state = state


class _Context:
    def __init__(self, state: _State) -> None:
        self.state = state

    async def __aenter__(self) -> AsyncSession:
        return cast(AsyncSession, _FakeSession(self.state))

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        return None


class _Factory:
    def __init__(self, state: _State) -> None:
        self.state = state

    def __call__(self) -> _Context:
        return _Context(self.state)

    def begin(self) -> _Context:
        return _Context(self.state)


class _SourceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.state = cast(_FakeSession, session).state

    async def get(self, source_id: UUID) -> Source | None:
        source = self.state.source
        return source if source is not None and source.id == source_id else None


class _JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.state = cast(_FakeSession, session).state

    async def acquire_queue_lock(self) -> None:
        self.state.queue_lock_calls += 1

    async def get_by_idempotency_key(self, idempotency_key: str) -> IngestionJob | None:
        job = self.state.job
        if job is not None and job.idempotency_key == idempotency_key:
            return job
        return None

    async def outstanding_count(self) -> int:
        return self.state.outstanding_count

    async def enqueue(
        self,
        *,
        source_id: UUID,
        idempotency_key: str,
        options: dict[str, Any] | None = None,
        requested_by: str | None = None,
    ) -> IngestionJob:
        if self.state.database_failure:
            raise SQLAlchemyError("database unavailable with internal connection details")
        self.state.enqueue_calls.append(
            {
                "source_id": source_id,
                "idempotency_key": idempotency_key,
                "options": options,
                "requested_by": requested_by,
            }
        )
        if self.state.job is None:
            self.state.job = _job(
                source_id,
                idempotency_key=idempotency_key,
                options=options or {},
                requested_by=requested_by,
            )
        return self.state.job

    async def get(self, job_id: UUID) -> IngestionJob | None:
        if self.state.database_failure:
            raise SQLAlchemyError("database unavailable")
        job = self.state.job
        return job if job is not None and job.id == job_id else None


@pytest.fixture(autouse=True)
def repositories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "SourceRepository", _SourceRepository)
    monkeypatch.setattr(module, "IngestionJobRepository", _JobRepository)


def _source(*, enabled: bool = True) -> Source:
    return Source(
        id=uuid4(),
        name=f"source-{uuid4()}",
        source_type="github_repository",
        repository="kubernetes/website",
        branch="main",
        enabled=enabled,
        config={},
    )


def _job(
    source_id: UUID,
    *,
    idempotency_key: str,
    options: dict[str, Any],
    requested_by: str | None = None,
) -> IngestionJob:
    now = datetime(2026, 7, 17, 11, 0, tzinfo=UTC)
    return IngestionJob(
        id=uuid4(),
        source_id=source_id,
        idempotency_key=idempotency_key,
        status="pending",
        requested_by=requested_by,
        options=options,
        stats=SyncStats().model_dump(),
        created_at=now,
        updated_at=now,
    )


def _service(state: _State, *, limit: int = 100) -> IngestionQueueService:
    factory = cast(async_sessionmaker[AsyncSession], _Factory(state))
    return IngestionQueueService(
        factory,
        max_outstanding_jobs=limit,
        retry_after_seconds=7,
    )


@pytest.mark.asyncio
async def test_enqueue_generates_unique_key_and_persists_only_worker_options() -> None:
    source = _source()
    state = _State(source=source)

    job = await _service(state).enqueue(
        source.id,
        options=IngestionJobOptions(dry_run=True, allow_delete_missing=False),
        requested_by="operator@example.test",
    )

    assert job.status == "pending"
    assert job.idempotency_key.startswith(f"source-sync:{source.id}:")
    assert state.enqueue_calls == [
        {
            "source_id": source.id,
            "idempotency_key": job.idempotency_key,
            "options": {"dry_run": True, "allow_delete_missing": False},
            "requested_by": "operator@example.test",
        }
    ]


@pytest.mark.asyncio
async def test_same_idempotency_key_replays_existing_job() -> None:
    source = _source()
    existing = _job(
        source.id,
        idempotency_key="same-request",
        options={"dry_run": False, "allow_delete_missing": True},
    )
    state = _State(source=source, job=existing)
    service = _service(state)

    first = await service.enqueue(
        source.id,
        options=IngestionJobOptions(),
        idempotency_key="same-request",
    )
    second = await service.enqueue(
        source.id,
        options=IngestionJobOptions(),
        idempotency_key="same-request",
    )

    assert first.id == existing.id
    assert second.id == existing.id


@pytest.mark.asyncio
async def test_queue_capacity_rejects_new_work_but_never_idempotent_replay() -> None:
    source = _source()
    state = _State(source=source, outstanding_count=1)

    with pytest.raises(RateLimitError) as exc_info:
        await _service(state, limit=1).enqueue(
            source.id,
            options=IngestionJobOptions(),
            idempotency_key="new-request",
        )

    assert exc_info.value.retry_after_seconds == 7
    assert exc_info.value.details == {"queue": "ingestion", "limit": 1}
    assert state.enqueue_calls == []

    existing = _job(
        source.id,
        idempotency_key="existing-request",
        options={"dry_run": False, "allow_delete_missing": True},
    )
    replay_state = _State(source=source, job=existing, outstanding_count=1)
    replay = await _service(replay_state, limit=1).enqueue(
        source.id,
        options=IngestionJobOptions(),
        idempotency_key="existing-request",
    )

    assert replay.id == existing.id
    assert replay_state.enqueue_calls == []


@pytest.mark.asyncio
async def test_idempotency_key_cannot_cross_sources_or_change_options() -> None:
    source = _source()
    another_source_id = uuid4()
    cross_source = _job(
        another_source_id,
        idempotency_key="collision",
        options={"dry_run": False, "allow_delete_missing": True},
    )

    with pytest.raises(ValidationError, match="another source"):
        await _service(_State(source=source, job=cross_source)).enqueue(
            source.id,
            options=IngestionJobOptions(),
            idempotency_key="collision",
        )

    changed_options = _job(
        source.id,
        idempotency_key="collision",
        options={"dry_run": True, "allow_delete_missing": True},
    )
    with pytest.raises(ValidationError, match="different options"):
        await _service(_State(source=source, job=changed_options)).enqueue(
            source.id,
            options=IngestionJobOptions(),
            idempotency_key="collision",
        )


@pytest.mark.asyncio
async def test_enqueue_rejects_missing_or_disabled_source() -> None:
    missing_id = uuid4()
    with pytest.raises(NotFoundError, match="source was not found"):
        await _service(_State(source=None)).enqueue(
            missing_id,
            options=IngestionJobOptions(),
        )

    disabled = _source(enabled=False)
    with pytest.raises(ValidationError, match="disabled"):
        await _service(_State(source=disabled)).enqueue(
            disabled.id,
            options=IngestionJobOptions(),
        )


@pytest.mark.asyncio
async def test_poll_missing_job_and_database_errors_are_controlled() -> None:
    source = _source()
    service = _service(_State(source=source))
    with pytest.raises(NotFoundError, match="job was not found"):
        await service.get(uuid4())

    failing = _service(_State(source=source, database_failure=True))
    with pytest.raises(DatabaseError) as exc_info:
        await failing.enqueue(source.id, options=IngestionJobOptions())

    assert exc_info.value.message == "Unable to enqueue ingestion job"
    assert "connection details" not in exc_info.value.message
