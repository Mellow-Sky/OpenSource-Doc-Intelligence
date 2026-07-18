"""Administrative endpoints for durable, non-blocking source synchronization."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_container
from app.container import AppContainer
from app.core.security import require_admin_api_key
from app.db.models.ingestion import IngestionJob
from app.ingestion.incremental import JobStatus, SyncStats
from app.schemas.ingestion import (
    IngestionJobCreateRequest,
    IngestionJobOptions,
    IngestionJobResponse,
    SourceSyncRequest,
)
from app.services.ingestion_queue_service import IngestionQueueService

router = APIRouter(
    prefix="/api/v1/ingestion",
    tags=["ingestion"],
    dependencies=[Depends(require_admin_api_key)],
)


def get_ingestion_queue_service(
    container: Annotated[AppContainer, Depends(get_container)],
) -> IngestionQueueService:
    """Build a request-scoped queue service around the process database."""
    return IngestionQueueService(
        container.database.session_factory,
        max_outstanding_jobs=container.settings.ingestion_max_outstanding_jobs,
        retry_after_seconds=container.settings.queue_retry_after_seconds,
    )


@router.post(
    "/jobs",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_ingestion_job(
    payload: IngestionJobCreateRequest,
    service: Annotated[IngestionQueueService, Depends(get_ingestion_queue_service)],
) -> IngestionJobResponse:
    """Validate and enqueue a source sync; no loader or model call runs in this request."""
    job = await service.enqueue(
        payload.source_id,
        options=payload.options,
        idempotency_key=payload.idempotency_key,
        requested_by=payload.requested_by,
    )
    return _job_response(job)


@router.get("/jobs/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_job(
    job_id: UUID,
    service: Annotated[IngestionQueueService, Depends(get_ingestion_queue_service)],
) -> IngestionJobResponse:
    """Poll the durable lifecycle and synchronization counters of one job."""
    return _job_response(await service.get(job_id))


@router.post(
    "/sources/{source_id}/sync",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def sync_source(
    source_id: UUID,
    service: Annotated[IngestionQueueService, Depends(get_ingestion_queue_service)],
    payload: SourceSyncRequest | None = None,
) -> IngestionJobResponse:
    """Enqueue the source-scoped synchronization shortcut and return immediately."""
    request = payload or SourceSyncRequest()
    job = await service.enqueue(
        source_id,
        options=request.options,
        idempotency_key=request.idempotency_key,
        requested_by=request.requested_by,
    )
    return _job_response(job)


def _job_response(job: IngestionJob) -> IngestionJobResponse:
    return IngestionJobResponse(
        id=job.id,
        source_id=job.source_id,
        idempotency_key=job.idempotency_key,
        status=JobStatus(job.status),
        requested_by=job.requested_by,
        options=IngestionJobOptions.model_validate(job.options or {}),
        stats=SyncStats.model_validate(job.stats or {}),
        error=job.error,
        started_at=job.started_at,
        finished_at=job.finished_at,
        heartbeat_at=job.heartbeat_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
