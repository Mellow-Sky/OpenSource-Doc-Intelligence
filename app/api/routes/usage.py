"""Usage, cost, and persisted stage-latency reporting endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db_session
from app.core.exceptions import ValidationError
from app.core.security import require_api_key
from app.repositories.usage_repository import UsageFilters, UsageRepository
from app.schemas.usage import UsageSummaryResponse
from app.services.usage_service import UsageService

router = APIRouter(
    prefix="/api/v1",
    tags=["usage"],
    dependencies=[Depends(require_api_key)],
)


def get_usage_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> UsageService:
    """Build the usage service around the request-scoped database session."""
    return UsageService(UsageRepository(session))


@router.get("/usage/summary", response_model=UsageSummaryResponse)
async def usage_summary(
    service: Annotated[UsageService, Depends(get_usage_service)],
    request_id: UUID | None = None,
    operation: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    model: Annotated[str | None, Query(min_length=1, max_length=255)] = None,
    provider: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    created_from: datetime | None = None,
    created_until: datetime | None = None,
) -> UsageSummaryResponse:
    """Summarize provider usage plus authoritative request/retrieval timings."""
    try:
        filters = UsageFilters(
            request_id=request_id,
            operation=_strip(operation),
            model=_strip(model),
            provider=_strip(provider),
            created_from=created_from,
            created_until=created_until,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return await service.summarize(filters)


def _strip(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise ValidationError("Usage filters must not be blank")
    return stripped
