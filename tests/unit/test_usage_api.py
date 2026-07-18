"""HTTP contract tests for usage aggregation and validation."""

from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from app.api.exception_handlers import RequestIDMiddleware, install_exception_handlers
from app.api.routes.usage import get_usage_service, router
from app.core.security import require_api_key
from app.repositories.usage_repository import UsageFilters
from app.schemas.usage import (
    PerformanceResponse,
    RequestPerformanceResponse,
    RetrievalPerformanceResponse,
    UsageBreakdownResponse,
    UsageFilterResponse,
    UsageSummaryResponse,
    UsageTotalsResponse,
)


class FakeUsageService:
    def __init__(self, response: UsageSummaryResponse) -> None:
        self.response = response
        self.filters: UsageFilters | None = None

    async def summarize(self, filters: UsageFilters) -> UsageSummaryResponse:
        self.filters = filters
        return self.response


def _response(request_id: UUID) -> UsageSummaryResponse:
    totals = UsageTotalsResponse(
        operation_count=2,
        request_count=1,
        prompt_tokens=30,
        completion_tokens=10,
        total_tokens=40,
        input_text_count=3,
        input_character_count=240,
        estimated_cost_usd=None,
        priced_operation_count=1,
        unpriced_operation_count=1,
        cost_complete=False,
        average_latency_ms=12.5,
    )
    return UsageSummaryResponse(
        filters=UsageFilterResponse(request_id=request_id, operation="answer_generation"),
        total=totals,
        by_operation=[
            UsageBreakdownResponse(
                key="answer_generation",
                **totals.model_dump(),
            )
        ],
        by_model=[],
        by_provider=[],
        performance=PerformanceResponse(
            requests=RequestPerformanceResponse(
                request_count=1,
                average_latency_ms=100,
                estimated_cost_usd=None,
                priced_request_count=0,
                unpriced_request_count=1,
                cost_complete=False,
            ),
            retrieval=RetrievalPerformanceResponse(
                run_count=1,
                average_keyword_latency_ms=4,
                average_vector_latency_ms=8,
                average_rerank_latency_ms=3,
                average_total_latency_ms=15,
            ),
        ),
    )


def _app(service: FakeUsageService) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    install_exception_handlers(app)
    app.include_router(router)

    async def override_service() -> FakeUsageService:
        return service

    async def allow_api_key() -> None:
        return None

    app.dependency_overrides[get_usage_service] = override_service
    app.dependency_overrides[require_api_key] = allow_api_key
    return app


@pytest.mark.asyncio
async def test_usage_summary_returns_typed_breakdowns_and_filters() -> None:
    request_id = uuid4()
    service = FakeUsageService(_response(request_id))
    transport = httpx.ASGITransport(app=_app(service))
    params: dict[str, Any] = {
        "request_id": str(request_id),
        "operation": " answer_generation ",
        "created_from": "2026-01-01T00:00:00Z",
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/usage/summary", params=params)

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"]["request_count"] == 1
    assert payload["total"]["input_text_count"] == 3
    assert payload["total"]["input_character_count"] == 240
    assert payload["total"]["estimated_cost_usd"] is None
    assert payload["total"]["cost_complete"] is False
    assert payload["performance"]["retrieval"]["average_vector_latency_ms"] == 8
    assert service.filters is not None
    assert service.filters.operation == "answer_generation"
    assert service.filters.created_from is not None


@pytest.mark.asyncio
async def test_usage_summary_rejects_naive_time_with_uniform_error() -> None:
    request_id = uuid4()
    service = FakeUsageService(_response(request_id))
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/usage/summary",
            params={"created_from": "2026-01-01T00:00:00"},
        )

    assert response.status_code == 422
    payload = response.json()["error"]
    assert payload["code"] == "VALIDATION_FAILED"
    assert payload["request_id"] == response.headers["x-request-id"]
    assert "timezone-aware" in payload["message"]
    assert service.filters is None
