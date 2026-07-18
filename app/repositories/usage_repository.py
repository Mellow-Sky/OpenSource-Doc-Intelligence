"""Batch persistence and aggregate reads for provider token and cost usage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from sqlalchemy import case, func, insert, select
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import ColumnElement

from app.db.models.conversation import Message
from app.db.models.retrieval import RetrievalRun
from app.db.models.usage import UsageRecord
from app.repositories.batching import database_batches

MAX_USAGE_PAGE_SIZE = 1000


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageRecordCreate:
    """One priced or explicitly unpriced model/provider operation."""

    request_id: UUID
    operation: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    input_text_count: int = 0
    input_character_count: int = 0
    estimated_cost: Decimal | None = None
    latency_ms: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.operation.strip() or not self.model.strip() or not self.provider.strip():
            raise ValueError("operation, model, and provider must not be blank")
        counts = (
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
            self.input_text_count,
            self.input_character_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("usage counts must be non-negative")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens plus completion_tokens")
        if self.estimated_cost is not None and self.estimated_cost < 0:
            raise ValueError("estimated_cost must be non-negative")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        _utc(self.created_at)


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageFilters:
    """Optional parameterized filters for usage queries."""

    request_id: UUID | None = None
    operation: str | None = None
    model: str | None = None
    provider: str | None = None
    created_from: datetime | None = None
    created_until: datetime | None = None

    def __post_init__(self) -> None:
        start = _utc(self.created_from) if self.created_from is not None else None
        end = _utc(self.created_until) if self.created_until is not None else None
        if start is not None and end is not None and end <= start:
            raise ValueError("created_until must be after created_from")


@dataclass(frozen=True, slots=True)
class UsageSummary:
    """Aggregate usage where total cost stays unknown if any operation is unpriced."""

    operation_count: int
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_text_count: int
    input_character_count: int
    estimated_cost: Decimal | None
    priced_operation_count: int
    unpriced_operation_count: int
    average_latency_ms: float


@dataclass(frozen=True, slots=True)
class UsageGroupSummary:
    """A usage aggregate for one operation, provider, or model value."""

    key: str
    summary: UsageSummary


@dataclass(frozen=True, slots=True)
class RequestPerformanceSummary:
    """Request-level totals persisted on generated assistant messages."""

    request_count: int
    average_latency_ms: float
    estimated_cost: Decimal | None
    priced_request_count: int
    unpriced_request_count: int


@dataclass(frozen=True, slots=True)
class RetrievalPerformanceSummary:
    """Average retrieval-stage timings persisted for every retrieval run."""

    run_count: int
    average_keyword_latency_ms: float
    average_vector_latency_ms: float
    average_rerank_latency_ms: float
    average_total_latency_ms: float


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    """Request and retrieval timing aggregates for one UTC time window."""

    requests: RequestPerformanceSummary
    retrieval: RetrievalPerformanceSummary


class UsageRepository:
    """Store provider accounting records without taking transaction ownership."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_many(self, records: Sequence[UsageRecordCreate]) -> list[UsageRecord]:
        """Insert bounded operation batches without committing the caller transaction."""
        if not records:
            return []
        persisted: list[UsageRecord] = []
        for batch in database_batches(records):
            statement = (
                insert(UsageRecord)
                .values(
                    [
                        {
                            "id": record.id,
                            "request_id": record.request_id,
                            "operation": record.operation,
                            "model": record.model,
                            "provider": record.provider,
                            "prompt_tokens": record.prompt_tokens,
                            "completion_tokens": record.completion_tokens,
                            "total_tokens": record.total_tokens,
                            "input_text_count": record.input_text_count,
                            "input_character_count": record.input_character_count,
                            "estimated_cost": record.estimated_cost,
                            "latency_ms": record.latency_ms,
                            "created_at": _utc(record.created_at),
                        }
                        for record in batch
                    ]
                )
                .returning(UsageRecord)
            )
            persisted.extend(await self._session.scalars(statement))
        return persisted

    async def list_records(
        self,
        *,
        filters: UsageFilters | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UsageRecord]:
        """Return a deterministic usage page using only bound filter values."""
        _validate_page(limit, offset)
        predicates = _predicates(filters or UsageFilters())
        records = await self._session.scalars(
            select(UsageRecord)
            .where(*predicates)
            .order_by(UsageRecord.created_at.desc(), UsageRecord.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(records)

    async def summarize(self, *, filters: UsageFilters | None = None) -> UsageSummary:
        """Aggregate tokens, latency, and all-or-null cost for matching operations."""
        predicates = _predicates(filters or UsageFilters())
        row = (await self._session.execute(_usage_summary_columns().where(*predicates))).one()
        return _usage_summary_from_row(row)

    async def summarize_by(
        self,
        dimension: Literal["operation", "model", "provider"],
        *,
        filters: UsageFilters | None = None,
    ) -> list[UsageGroupSummary]:
        """Group usage by a bounded, caller-selected dimension without dynamic SQL text."""
        predicates = _predicates(filters or UsageFilters())
        grouping_columns = {
            "operation": UsageRecord.operation,
            "model": UsageRecord.model,
            "provider": UsageRecord.provider,
        }
        grouping = grouping_columns[dimension]
        rows = (
            await self._session.execute(
                _usage_summary_columns(grouping.label("group_key"))
                .where(*predicates)
                .group_by(grouping)
                .order_by(grouping)
            )
        ).all()
        return [
            UsageGroupSummary(key=str(row.group_key), summary=_usage_summary_from_row(row))
            for row in rows
        ]

    async def summarize_performance(
        self,
        *,
        created_from: datetime | None = None,
        created_until: datetime | None = None,
    ) -> PerformanceSummary:
        """Aggregate request and retrieval timings using their authoritative audit tables."""
        time_range = UsageFilters(created_from=created_from, created_until=created_until)
        message_predicates: list[ColumnElement[bool]] = [Message.role == "assistant"]
        retrieval_predicates: list[ColumnElement[bool]] = []
        if time_range.created_from is not None:
            start = _utc(time_range.created_from)
            message_predicates.append(Message.created_at >= start)
            retrieval_predicates.append(RetrievalRun.created_at >= start)
        if time_range.created_until is not None:
            end = _utc(time_range.created_until)
            message_predicates.append(Message.created_at < end)
            retrieval_predicates.append(RetrievalRun.created_at < end)

        request_row = (
            await self._session.execute(
                select(
                    func.count(Message.id).label("request_count"),
                    func.coalesce(func.avg(Message.latency_ms), 0.0).label("average_latency_ms"),
                    case(
                        (
                            func.count(Message.cost) == func.count(Message.id),
                            func.sum(Message.cost),
                        ),
                        else_=None,
                    ).label("estimated_cost"),
                    func.count(Message.cost).label("priced_request_count"),
                ).where(*message_predicates)
            )
        ).one()
        retrieval_row = (
            await self._session.execute(
                select(
                    func.count(RetrievalRun.id).label("run_count"),
                    func.coalesce(func.avg(RetrievalRun.keyword_latency_ms), 0.0).label(
                        "average_keyword_latency_ms"
                    ),
                    func.coalesce(func.avg(RetrievalRun.vector_latency_ms), 0.0).label(
                        "average_vector_latency_ms"
                    ),
                    func.coalesce(func.avg(RetrievalRun.rerank_latency_ms), 0.0).label(
                        "average_rerank_latency_ms"
                    ),
                    func.coalesce(func.avg(RetrievalRun.total_latency_ms), 0.0).label(
                        "average_total_latency_ms"
                    ),
                ).where(*retrieval_predicates)
            )
        ).one()
        request_count = int(request_row.request_count)
        priced_request_count = int(request_row.priced_request_count)
        return PerformanceSummary(
            requests=RequestPerformanceSummary(
                request_count=request_count,
                average_latency_ms=float(request_row.average_latency_ms),
                estimated_cost=cast(Decimal | None, request_row.estimated_cost),
                priced_request_count=priced_request_count,
                unpriced_request_count=request_count - priced_request_count,
            ),
            retrieval=RetrievalPerformanceSummary(
                run_count=int(retrieval_row.run_count),
                average_keyword_latency_ms=float(retrieval_row.average_keyword_latency_ms),
                average_vector_latency_ms=float(retrieval_row.average_vector_latency_ms),
                average_rerank_latency_ms=float(retrieval_row.average_rerank_latency_ms),
                average_total_latency_ms=float(retrieval_row.average_total_latency_ms),
            ),
        )


def _usage_summary_columns(*columns: ColumnElement[Any]) -> Select[Any]:
    return select(
        *columns,
        func.count(UsageRecord.id).label("operation_count"),
        func.count(func.distinct(UsageRecord.request_id)).label("request_count"),
        func.coalesce(func.sum(UsageRecord.prompt_tokens), 0).label("prompt_tokens"),
        func.coalesce(func.sum(UsageRecord.completion_tokens), 0).label("completion_tokens"),
        func.coalesce(func.sum(UsageRecord.total_tokens), 0).label("total_tokens"),
        func.coalesce(func.sum(UsageRecord.input_text_count), 0).label("input_text_count"),
        func.coalesce(func.sum(UsageRecord.input_character_count), 0).label(
            "input_character_count"
        ),
        case(
            (
                func.count(UsageRecord.estimated_cost) == func.count(UsageRecord.id),
                func.sum(UsageRecord.estimated_cost),
            ),
            else_=None,
        ).label("estimated_cost"),
        func.count(UsageRecord.estimated_cost).label("priced_operation_count"),
        func.coalesce(func.avg(UsageRecord.latency_ms), 0.0).label("average_latency_ms"),
    )


def _usage_summary_from_row(row: Row[Any]) -> UsageSummary:
    operation_count = int(row.operation_count)
    priced_operation_count = int(row.priced_operation_count)
    return UsageSummary(
        operation_count=operation_count,
        request_count=int(row.request_count),
        prompt_tokens=int(row.prompt_tokens),
        completion_tokens=int(row.completion_tokens),
        total_tokens=int(row.total_tokens),
        input_text_count=int(row.input_text_count),
        input_character_count=int(row.input_character_count),
        estimated_cost=cast(Decimal | None, row.estimated_cost),
        priced_operation_count=priced_operation_count,
        unpriced_operation_count=operation_count - priced_operation_count,
        average_latency_ms=float(row.average_latency_ms),
    )


def _predicates(filters: UsageFilters) -> list[ColumnElement[bool]]:
    predicates: list[ColumnElement[bool]] = []
    if filters.request_id is not None:
        predicates.append(UsageRecord.request_id == filters.request_id)
    if filters.operation is not None:
        predicates.append(UsageRecord.operation == filters.operation)
    if filters.model is not None:
        predicates.append(UsageRecord.model == filters.model)
    if filters.provider is not None:
        predicates.append(UsageRecord.provider == filters.provider)
    if filters.created_from is not None:
        predicates.append(UsageRecord.created_at >= _utc(filters.created_from))
    if filters.created_until is not None:
        predicates.append(UsageRecord.created_at < _utc(filters.created_until))
    return predicates


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _validate_page(limit: int, offset: int) -> None:
    if not 1 <= limit <= MAX_USAGE_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_USAGE_PAGE_SIZE}")
    if offset < 0:
        raise ValueError("offset must be non-negative")
