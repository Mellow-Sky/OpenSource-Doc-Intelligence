"""Typed API contracts for token, cost, and latency aggregates."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class UsageFilterResponse(BaseModel):
    """Filters applied to provider operations and the shared time window."""

    request_id: UUID | None = None
    operation: str | None = None
    model: str | None = None
    provider: str | None = None
    created_from: datetime | None = None
    created_until: datetime | None = None


class UsageTotalsResponse(BaseModel):
    """Token and cost totals with explicit priced/unpriced cardinalities."""

    operation_count: int = Field(ge=0)
    request_count: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    input_text_count: int = Field(default=0, ge=0)
    input_character_count: int = Field(default=0, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    priced_operation_count: int = Field(ge=0)
    unpriced_operation_count: int = Field(ge=0)
    cost_complete: bool
    average_latency_ms: float = Field(ge=0)


class UsageBreakdownResponse(UsageTotalsResponse):
    """One grouped provider-usage aggregate."""

    key: str


class RequestPerformanceResponse(BaseModel):
    """Request-level latency and all-or-null cost from assistant messages."""

    request_count: int = Field(ge=0)
    average_latency_ms: float = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    priced_request_count: int = Field(ge=0)
    unpriced_request_count: int = Field(ge=0)
    cost_complete: bool


class RetrievalPerformanceResponse(BaseModel):
    """Persisted retrieval-stage latency averages."""

    run_count: int = Field(ge=0)
    average_keyword_latency_ms: float = Field(ge=0)
    average_vector_latency_ms: float = Field(ge=0)
    average_rerank_latency_ms: float = Field(ge=0)
    average_total_latency_ms: float = Field(ge=0)


class PerformanceResponse(BaseModel):
    """Request and retrieval performance for the selected time window."""

    requests: RequestPerformanceResponse
    retrieval: RetrievalPerformanceResponse


class UsageSummaryResponse(BaseModel):
    """Complete usage view returned by ``GET /api/v1/usage/summary``."""

    filters: UsageFilterResponse
    total: UsageTotalsResponse
    by_operation: list[UsageBreakdownResponse]
    by_model: list[UsageBreakdownResponse]
    by_provider: list[UsageBreakdownResponse]
    performance: PerformanceResponse
