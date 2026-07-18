"""Usage persistence and aggregate semantics against a real SQL database."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.db.models.usage import UsageRecord
from app.repositories.usage_repository import UsageFilters, UsageRecordCreate, UsageRepository


class AsyncSessionBridge:
    """Exercise the async repository contract using real synchronous SQL execution."""

    def __init__(self, session: Session) -> None:
        self._session = session

    async def execute(self, statement: Any) -> Any:
        return self._session.execute(statement)

    async def scalars(self, statement: Any) -> Any:
        return self._session.scalars(statement)


class _BatchRecordingSession:
    def __init__(self) -> None:
        self.calls = 0
        self.parameter_counts: list[int] = []

    async def scalars(self, statement: Any) -> list[UsageRecord]:
        self.calls += 1
        self.parameter_counts.append(len(statement.compile().params))
        return [UsageRecord(operation=f"batch-{self.calls}")]


@pytest.mark.asyncio
async def test_usage_summary_counts_distinct_requests_and_preserves_unknown_cost() -> None:
    engine = create_engine("sqlite://")
    UsageRecord.__table__.create(engine)

    now = datetime(2026, 1, 2, tzinfo=UTC)
    first_request = uuid4()
    second_request = uuid4()
    with Session(engine) as session:
        repository = UsageRepository(cast(AsyncSession, AsyncSessionBridge(session)))
        await repository.add_many(
            [
                UsageRecordCreate(
                    request_id=first_request,
                    operation="query_rewrite",
                    model="rewrite-model",
                    provider="remote",
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                    input_text_count=2,
                    input_character_count=120,
                    estimated_cost=Decimal("0.001"),
                    latency_ms=10,
                    created_at=now,
                ),
                UsageRecordCreate(
                    request_id=first_request,
                    operation="answer_generation",
                    model="answer-model",
                    provider="remote",
                    prompt_tokens=20,
                    completion_tokens=8,
                    total_tokens=28,
                    input_text_count=1,
                    input_character_count=80,
                    estimated_cost=None,
                    latency_ms=30,
                    created_at=now,
                ),
                UsageRecordCreate(
                    request_id=second_request,
                    operation="answer_generation",
                    model="answer-model",
                    provider="remote",
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                    input_text_count=3,
                    input_character_count=50,
                    estimated_cost=Decimal("0.003"),
                    latency_ms=20,
                    created_at=now + timedelta(seconds=1),
                ),
            ]
        )
        session.commit()
        summary = await repository.summarize()
        by_operation = await repository.summarize_by("operation")
        filtered = await repository.summarize(filters=UsageFilters(request_id=first_request))

    assert summary.operation_count == 3
    assert summary.request_count == 2
    assert summary.prompt_tokens == 35
    assert summary.completion_tokens == 15
    assert summary.total_tokens == 50
    assert summary.input_text_count == 6
    assert summary.input_character_count == 250
    assert summary.average_latency_ms == pytest.approx(20)
    assert summary.estimated_cost is None
    assert summary.priced_operation_count == 2
    assert summary.unpriced_operation_count == 1

    groups = {group.key: group.summary for group in by_operation}
    assert groups["query_rewrite"].estimated_cost == Decimal("0.00100000")
    assert groups["query_rewrite"].input_text_count == 2
    assert groups["answer_generation"].input_character_count == 130
    assert groups["query_rewrite"].unpriced_operation_count == 0
    assert groups["answer_generation"].estimated_cost is None
    assert groups["answer_generation"].unpriced_operation_count == 1
    assert filtered.operation_count == 2
    assert filtered.request_count == 1

    engine.dispose()


def test_usage_filters_require_aware_ordered_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        UsageFilters(created_from=datetime(2026, 1, 1))

    with pytest.raises(ValueError, match="must be after"):
        UsageFilters(
            created_from=datetime(2026, 1, 2, tzinfo=UTC),
            created_until=datetime(2026, 1, 1, tzinfo=UTC),
        )


@pytest.mark.asyncio
async def test_usage_repository_bounds_501_rows_and_combines_returning_results() -> None:
    session = _BatchRecordingSession()
    repository = UsageRepository(cast(AsyncSession, session))
    request_id = uuid4()
    records = [
        UsageRecordCreate(
            request_id=request_id,
            operation="ingestion_embedding",
            model="embedding-model",
            provider="local",
        )
        for _ in range(501)
    ]

    persisted = await repository.add_many(records)

    assert session.calls == 2
    assert session.parameter_counts[0] == session.parameter_counts[1] * 500
    assert [record.operation for record in persisted] == ["batch-1", "batch-2"]
