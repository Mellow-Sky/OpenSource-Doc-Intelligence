"""Application service for provider usage and end-to-end performance reporting."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from app.domain.usage import EmbeddingBatchUsage
from app.repositories.usage_repository import (
    PerformanceSummary,
    UsageFilters,
    UsageGroupSummary,
    UsageRecordCreate,
    UsageRepository,
    UsageSummary,
)
from app.schemas.usage import (
    PerformanceResponse,
    RequestPerformanceResponse,
    RetrievalPerformanceResponse,
    UsageBreakdownResponse,
    UsageFilterResponse,
    UsageSummaryResponse,
    UsageTotalsResponse,
)
from app.services.pricing_service import PricingCatalog


class UsageService:
    """Build a stable API response without leaking ORM or Decimal objects."""

    def __init__(
        self,
        repository: UsageRepository,
        pricing_catalog: PricingCatalog | None = None,
    ) -> None:
        self._repository = repository
        self._pricing = pricing_catalog or PricingCatalog()

    async def record_embedding_batches(
        self,
        *,
        request_id: UUID,
        operation: str,
        batches: Sequence[EmbeddingBatchUsage],
        created_at: datetime | None = None,
    ) -> None:
        """Persist measured embedding batches in one repository call."""

        if not operation.strip():
            raise ValueError("operation must not be blank")
        if not batches:
            return
        timestamp = created_at or datetime.now(UTC)
        await self._repository.add_many(
            [
                UsageRecordCreate(
                    request_id=request_id,
                    operation=operation,
                    model=batch.model,
                    provider=batch.provider,
                    prompt_tokens=batch.prompt_tokens,
                    completion_tokens=0,
                    total_tokens=batch.prompt_tokens,
                    input_text_count=batch.input_text_count,
                    input_character_count=batch.input_character_count,
                    estimated_cost=(
                        self._pricing.estimate(
                            provider=batch.provider,
                            model=batch.model,
                            prompt_tokens=batch.prompt_tokens,
                            completion_tokens=0,
                        )
                        # Embedding adapters use zero when the provider did not report usage.
                        # A non-empty batch cannot truly consume zero input tokens, so pricing
                        # it as zero would falsely mark the operation as fully priced.
                        if batch.prompt_tokens > 0
                        else None
                    ),
                    latency_ms=batch.latency_ms,
                    created_at=timestamp,
                )
                for batch in batches
            ]
        )

    async def summarize(self, filters: UsageFilters) -> UsageSummaryResponse:
        """Return totals, bounded breakdowns, and persisted timing aggregates."""
        total = await self._repository.summarize(filters=filters)
        by_operation = await self._repository.summarize_by("operation", filters=filters)
        by_model = await self._repository.summarize_by("model", filters=filters)
        by_provider = await self._repository.summarize_by("provider", filters=filters)
        performance = await self._repository.summarize_performance(
            created_from=filters.created_from,
            created_until=filters.created_until,
        )
        return UsageSummaryResponse(
            filters=UsageFilterResponse(
                request_id=filters.request_id,
                operation=filters.operation,
                model=filters.model,
                provider=filters.provider,
                created_from=filters.created_from,
                created_until=filters.created_until,
            ),
            total=_totals(total),
            by_operation=_breakdowns(by_operation),
            by_model=_breakdowns(by_model),
            by_provider=_breakdowns(by_provider),
            performance=_performance(performance),
        )


def _totals(summary: UsageSummary) -> UsageTotalsResponse:
    return UsageTotalsResponse(
        operation_count=summary.operation_count,
        request_count=summary.request_count,
        prompt_tokens=summary.prompt_tokens,
        completion_tokens=summary.completion_tokens,
        total_tokens=summary.total_tokens,
        input_text_count=summary.input_text_count,
        input_character_count=summary.input_character_count,
        estimated_cost_usd=_cost(summary.estimated_cost),
        priced_operation_count=summary.priced_operation_count,
        unpriced_operation_count=summary.unpriced_operation_count,
        cost_complete=summary.unpriced_operation_count == 0,
        average_latency_ms=summary.average_latency_ms,
    )


def _breakdowns(groups: list[UsageGroupSummary]) -> list[UsageBreakdownResponse]:
    return [
        UsageBreakdownResponse(key=group.key, **_totals(group.summary).model_dump())
        for group in groups
    ]


def _performance(summary: PerformanceSummary) -> PerformanceResponse:
    requests = summary.requests
    retrieval = summary.retrieval
    return PerformanceResponse(
        requests=RequestPerformanceResponse(
            request_count=requests.request_count,
            average_latency_ms=requests.average_latency_ms,
            estimated_cost_usd=_cost(requests.estimated_cost),
            priced_request_count=requests.priced_request_count,
            unpriced_request_count=requests.unpriced_request_count,
            cost_complete=requests.unpriced_request_count == 0,
        ),
        retrieval=RetrievalPerformanceResponse(
            run_count=retrieval.run_count,
            average_keyword_latency_ms=retrieval.average_keyword_latency_ms,
            average_vector_latency_ms=retrieval.average_vector_latency_ms,
            average_rerank_latency_ms=retrieval.average_rerank_latency_ms,
            average_total_latency_ms=retrieval.average_total_latency_ms,
        ),
    )


def _cost(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None
