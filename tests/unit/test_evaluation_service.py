"""Business-rule tests for evaluation queue admission and backpressure."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import RateLimitError
from app.db.models.evaluation import EvaluationRun
from app.schemas.evaluation import EvaluationRunCreateRequest
from app.services import evaluation_service as module
from app.services.evaluation_service import EvaluationService


@dataclass
class _State:
    outstanding: int = 0
    created: EvaluationRun | None = None
    lock_calls: int = 0


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

    def begin(self) -> _Context:
        return _Context(self.state)

    def __call__(self) -> _Context:
        return _Context(self.state)


class _Repository:
    def __init__(self, session: AsyncSession) -> None:
        self.state = cast(_FakeSession, session).state

    async def acquire_queue_lock(self) -> None:
        self.state.lock_calls += 1

    async def outstanding_count(self) -> int:
        return self.state.outstanding

    async def create_run(
        self,
        *,
        dataset_name: str,
        config_snapshot: dict[str, Any],
    ) -> EvaluationRun:
        run = EvaluationRun(
            id=uuid4(),
            dataset_name=dataset_name,
            config_snapshot=config_snapshot,
            started_at=datetime(2026, 7, 18, tzinfo=UTC),
            status="pending",
            summary={},
        )
        self.state.created = run
        return run

    async def get(self, run_id: UUID) -> EvaluationRun | None:
        run = self.state.created
        return run if run is not None and run.id == run_id else None

    async def result_count(self, _run_id: UUID) -> int:
        return 0


@pytest.fixture(autouse=True)
def repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "EvaluationRepository", _Repository)


def _service(state: _State, *, limit: int) -> EvaluationService:
    factory = cast(async_sessionmaker[AsyncSession], _Factory(state))
    return EvaluationService(
        factory,
        max_outstanding_runs=limit,
        retry_after_seconds=9,
    )


def _request() -> EvaluationRunCreateRequest:
    return EvaluationRunCreateRequest(
        dataset_name="kubernetes_eval",
        dataset_path="kubernetes_eval.jsonl",
        experiment_name="hybrid-rerank",
    )


@pytest.mark.asyncio
async def test_evaluation_queue_admits_only_below_capacity() -> None:
    admitted = _State(outstanding=1)
    view = await _service(admitted, limit=2).enqueue(_request())

    assert view.run is admitted.created
    assert admitted.lock_calls == 1
    assert view.run.config_snapshot["request"]["experiment_name"] == "hybrid-rerank"


@pytest.mark.asyncio
async def test_evaluation_queue_returns_retryable_rate_limit_at_capacity() -> None:
    full = _State(outstanding=2)

    with pytest.raises(RateLimitError) as exc_info:
        await _service(full, limit=2).enqueue(_request())

    assert exc_info.value.retry_after_seconds == 9
    assert exc_info.value.details == {"queue": "evaluation", "limit": 2}
    assert full.created is None
