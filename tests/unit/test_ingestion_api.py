"""HTTP contract tests for durable, non-blocking ingestion submission."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from app.api.exception_handlers import RequestIDMiddleware, install_exception_handlers
from app.api.routes.ingestion import get_ingestion_queue_service, router
from app.core.config import Settings, get_settings
from app.core.exceptions import NotFoundError
from app.db.models.ingestion import IngestionJob
from app.ingestion.incremental import SyncStats
from app.schemas.ingestion import IngestionJobOptions
from app.services.ingestion_queue_service import IngestionQueueService


class _FakeQueueService:
    def __init__(self, job: IngestionJob) -> None:
        self.job = job
        self.enqueue_calls: list[dict[str, Any]] = []
        self.missing = False

    async def enqueue(
        self,
        source_id: UUID,
        *,
        options: IngestionJobOptions,
        idempotency_key: str | None = None,
        requested_by: str | None = None,
    ) -> IngestionJob:
        self.enqueue_calls.append(
            {
                "source_id": source_id,
                "options": options,
                "idempotency_key": idempotency_key,
                "requested_by": requested_by,
            }
        )
        return self.job

    async def get(self, job_id: UUID) -> IngestionJob:
        if self.missing or job_id != self.job.id:
            raise NotFoundError("Ingestion job was not found")
        return self.job


def _job(source_id: UUID) -> IngestionJob:
    now = datetime(2026, 7, 17, 10, 30, tzinfo=UTC)
    return IngestionJob(
        id=uuid4(),
        source_id=source_id,
        idempotency_key="client-operation-42",
        status="pending",
        requested_by="release-bot",
        options={"dry_run": False, "allow_delete_missing": True},
        stats=SyncStats().model_dump(),
        error=None,
        started_at=None,
        finished_at=None,
        heartbeat_at=None,
        created_at=now,
        updated_at=now,
    )


def _app(service: _FakeQueueService) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    install_exception_handlers(app)
    app.include_router(router)
    settings = Settings(_env_file=None, app_env="test", admin_api_key="admin-secret")

    async def settings_override() -> Settings:
        return settings

    async def service_override() -> IngestionQueueService:
        return cast(IngestionQueueService, service)

    app.dependency_overrides[get_settings] = settings_override
    app.dependency_overrides[get_ingestion_queue_service] = service_override
    return app


@pytest.mark.asyncio
async def test_create_job_returns_202_and_only_enqueues() -> None:
    source_id = uuid4()
    service = _FakeQueueService(_job(source_id))
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ingestion/jobs",
            headers={"x-admin-api-key": "admin-secret"},
            json={
                "source_id": str(source_id),
                "idempotency_key": "client-operation-42",
                "requested_by": "release-bot",
                "options": {"dry_run": False, "allow_delete_missing": True},
            },
        )

    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    assert response.json()["stats"] == SyncStats().model_dump()
    assert service.enqueue_calls == [
        {
            "source_id": source_id,
            "options": IngestionJobOptions(),
            "idempotency_key": "client-operation-42",
            "requested_by": "release-bot",
        }
    ]


@pytest.mark.asyncio
async def test_source_sync_accepts_an_empty_body_and_uses_safe_defaults() -> None:
    source_id = uuid4()
    service = _FakeQueueService(_job(source_id))
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/api/v1/ingestion/sources/{source_id}/sync",
            headers={"x-admin-api-key": "admin-secret"},
        )

    assert response.status_code == 202
    assert len(service.enqueue_calls) == 1
    assert service.enqueue_calls[0]["options"] == IngestionJobOptions()
    assert service.enqueue_calls[0]["idempotency_key"] is None


@pytest.mark.asyncio
async def test_job_polling_returns_full_durable_state() -> None:
    source_id = uuid4()
    job = _job(source_id)
    service = _FakeQueueService(job)
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/api/v1/ingestion/jobs/{job.id}",
            headers={"x-admin-api-key": "admin-secret"},
        )

    assert response.status_code == 200
    assert response.json()["id"] == str(job.id)
    assert response.json()["source_id"] == str(source_id)
    assert response.json()["created_at"] == "2026-07-17T10:30:00Z"


@pytest.mark.asyncio
async def test_missing_job_uses_uniform_error_contract() -> None:
    source_id = uuid4()
    service = _FakeQueueService(_job(source_id))
    service.missing = True
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/api/v1/ingestion/jobs/{uuid4()}",
            headers={"x-admin-api-key": "admin-secret", "x-request-id": "request-123"},
        )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "NOT_FOUND",
            "message": "Ingestion job was not found",
            "request_id": "request-123",
            "details": {},
        }
    }


@pytest.mark.asyncio
async def test_admin_authentication_failure_uses_uniform_error_contract() -> None:
    source_id = uuid4()
    service = _FakeQueueService(_job(source_id))
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ingestion/jobs",
            headers={"x-request-id": "auth-request"},
            json={"source_id": str(source_id)},
        )

    assert response.status_code == 401
    assert response.json()["error"] == {
        "code": "AUTHENTICATION_FAILED",
        "message": "Invalid admin API key",
        "request_id": "auth-request",
        "details": {},
    }


@pytest.mark.asyncio
async def test_unknown_worker_option_is_rejected_before_enqueue() -> None:
    source_id = uuid4()
    service = _FakeQueueService(_job(source_id))
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ingestion/jobs",
            headers={"x-admin-api-key": "admin-secret"},
            json={"source_id": str(source_id), "options": {"dangerous": True}},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_FAILED"
    assert service.enqueue_calls == []
