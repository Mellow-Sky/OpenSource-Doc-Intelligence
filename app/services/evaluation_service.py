"""Application service for durable, non-blocking evaluation submission and polling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import DatabaseError, NotFoundError, RateLimitError
from app.db.models.evaluation import EvaluationRun
from app.repositories.evaluation_repository import EvaluationRepository
from app.schemas.evaluation import EvaluationRunCreateRequest
from evaluation.models import EvaluationRunReport


class EvaluationLeaseLostError(RuntimeError):
    """Raised when a worker tries to complete a lease that it no longer owns."""


@dataclass(frozen=True, slots=True)
class EvaluationRunView:
    """One run plus an aggregate that avoids loading large result JSON payloads."""

    run: EvaluationRun
    result_count: int


class EvaluationService:
    """Own queue transaction boundaries while leaving execution to a worker."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        max_outstanding_runs: int = 10,
        retry_after_seconds: int = 5,
    ) -> None:
        if max_outstanding_runs < 1 or retry_after_seconds < 1:
            raise ValueError("queue capacity and retry delay must be positive")
        self._session_factory = session_factory
        self._max_outstanding_runs = max_outstanding_runs
        self._retry_after_seconds = retry_after_seconds

    async def enqueue(self, request: EvaluationRunCreateRequest) -> EvaluationRunView:
        """Persist a pending run and return immediately without executing any case."""
        snapshot = {
            "request": {
                "dataset_path": request.dataset_path,
                "experiment_name": request.experiment_name,
            },
            "overrides": request.config_snapshot,
        }
        try:
            async with self._session_factory.begin() as session:
                repository = EvaluationRepository(session)
                await repository.acquire_queue_lock()
                if await repository.outstanding_count() >= self._max_outstanding_runs:
                    raise RateLimitError(
                        "Evaluation queue capacity exceeded",
                        details={
                            "queue": "evaluation",
                            "limit": self._max_outstanding_runs,
                        },
                        retry_after_seconds=self._retry_after_seconds,
                    )
                run = await repository.create_run(
                    dataset_name=request.dataset_name,
                    config_snapshot=snapshot,
                )
                return EvaluationRunView(run=run, result_count=0)
        except RateLimitError:
            raise
        except SQLAlchemyError as exc:
            raise DatabaseError(
                "Unable to enqueue evaluation run",
                details={"operation": "evaluation_run_enqueue"},
            ) from exc

    async def get(self, run_id: UUID) -> EvaluationRunView:
        """Read one run and its result count for API polling."""
        try:
            async with self._session_factory() as session:
                repository = EvaluationRepository(session)
                run = await repository.get(run_id)
                if run is None:
                    raise NotFoundError("Evaluation run was not found")
                return EvaluationRunView(
                    run=run,
                    result_count=await repository.result_count(run_id),
                )
        except NotFoundError:
            raise
        except SQLAlchemyError as exc:
            raise DatabaseError(
                "Unable to read evaluation run",
                details={"operation": "evaluation_run_get"},
            ) from exc

    async def complete(
        self,
        run: EvaluationRun,
        report: EvaluationRunReport,
        *,
        report_path: Path,
        lease_started_at: datetime,
    ) -> None:
        """Atomically upsert all cases, insert linked outputs, and finish one lease."""
        try:
            async with self._session_factory.begin() as session:
                repository = EvaluationRepository(session)
                await repository.add_report_results(
                    run.id,
                    dataset_name=run.dataset_name,
                    records=report.results,
                )
                updated = await repository.mark_succeeded(
                    run.id,
                    summary=report.summary,
                    report_path=str(report_path),
                    lease_started_at=lease_started_at,
                )
                if not updated:
                    raise EvaluationLeaseLostError(
                        "evaluation run completion compare-and-swap was rejected"
                    )
        except EvaluationLeaseLostError:
            raise
        except SQLAlchemyError as exc:
            raise DatabaseError(
                "Unable to persist evaluation results",
                details={"operation": "evaluation_run_complete"},
            ) from exc
