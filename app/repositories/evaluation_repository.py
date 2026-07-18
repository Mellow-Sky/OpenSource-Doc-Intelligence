"""Durable evaluation run queue and batched per-case result persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import Table, case, func, insert, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.evaluation import (
    EvaluationCase as EvaluationCaseModel,
)
from app.db.models.evaluation import (
    EvaluationResult,
    EvaluationRun,
)
from app.domain.evaluation import EvaluationCase, EvaluationRunStatus
from app.repositories.queue_gate import EVALUATION_QUEUE, acquire_queue_advisory_lock
from evaluation.models import EvaluationResultRecord

MAX_RESULT_BATCH_SIZE = 1_000
MAX_CASE_BATCH_SIZE = 1_000


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationCaseUpsert:
    """Complete dataset row used to create or refresh a stable case identity."""

    external_id: str
    dataset_name: str
    question: str
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    reference_answer: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    expected_citations: list[str] = field(default_factory=list)
    answerable: bool
    difficulty: str
    category: str
    source_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    human_reviewed: bool = False

    @classmethod
    def from_case(cls, dataset_name: str, case: EvaluationCase) -> EvaluationCaseUpsert:
        """Convert the report's validated case snapshot without dropping provenance."""
        return cls(
            external_id=case.id,
            dataset_name=dataset_name,
            question=case.question,
            conversation_history=[
                turn.model_dump(mode="json") for turn in case.conversation_history
            ],
            reference_answer=case.reference_answer,
            relevant_chunk_ids=list(case.relevant_chunk_ids),
            expected_citations=list(case.expected_citations),
            answerable=case.answerable,
            difficulty=case.difficulty.value,
            category=case.category,
            source_type=case.source_type,
            metadata=dict(case.metadata),
            human_reviewed=case.human_reviewed,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationResultCreate:
    """Complete serializable output for one evaluation case."""

    case_external_id: str
    question: str
    generated_answer: str
    predicted_answerable: bool
    retrieved_results: list[dict[str, Any]] = field(default_factory=list)
    citations: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    evaluation_case_id: UUID | None = None
    id: UUID = field(default_factory=uuid4)

    @classmethod
    def from_record(
        cls,
        record: EvaluationResultRecord,
        *,
        evaluation_case_id: UUID | None = None,
    ) -> EvaluationResultCreate:
        """Preserve a runner result, including judge and reference-case provenance."""
        metrics: dict[str, Any] = dict(record.metrics)
        metrics["_evaluation"] = {
            "case": record.case.model_dump(mode="json"),
            "rewritten_query": record.rewritten_query,
            "judge": record.judge.model_dump(mode="json") if record.judge is not None else None,
            "judge_provider": record.judge_provider,
            "judge_model": record.judge_model,
        }
        usage: dict[str, Any] = dict(record.usage)
        usage["judge_usage"] = {
            **record.judge_usage,
            "provider": record.judge_provider,
            "model": record.judge_model,
            "estimated_cost_usd": record.judge_estimated_cost_usd,
        }
        return cls(
            evaluation_case_id=evaluation_case_id,
            case_external_id=record.case.id,
            question=record.case.question,
            generated_answer=record.generated_answer,
            predicted_answerable=record.predicted_answerable,
            retrieved_results=[
                evidence.model_dump(mode="json") for evidence in record.retrieved_evidence
            ],
            citations=record.citations.model_dump(mode="json"),
            metrics=metrics,
            latency=dict(record.latency_ms),
            usage=usage,
            error=record.error,
        )


class EvaluationRepository:
    """Persist evaluation runs without committing or rolling back caller transactions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_run(
        self,
        *,
        dataset_name: str,
        config_snapshot: Mapping[str, Any],
        queued_at: datetime | None = None,
        run_id: UUID | None = None,
    ) -> EvaluationRun:
        """Add one pending run; ``started_at`` records queue time until it is claimed."""
        timestamp = _utc(queued_at or datetime.now(UTC))
        run = EvaluationRun(
            id=run_id or uuid4(),
            dataset_name=dataset_name,
            config_snapshot=dict(config_snapshot),
            started_at=timestamp,
            heartbeat_at=None,
            status=EvaluationRunStatus.PENDING.value,
            summary={},
        )
        self._session.add(run)
        await self._session.flush()
        return run

    async def get(self, run_id: UUID) -> EvaluationRun | None:
        """Return one run by primary key."""
        return await self._session.get(EvaluationRun, run_id)

    async def result_count(self, run_id: UUID) -> int:
        """Count persisted case results without loading their JSON payloads."""
        value = await self._session.scalar(
            select(func.count(EvaluationResult.id)).where(
                EvaluationResult.evaluation_run_id == run_id
            )
        )
        return int(value or 0)

    async def acquire_queue_lock(self) -> None:
        """Serialize global evaluation queue admission and claim decisions."""

        await acquire_queue_advisory_lock(self._session, EVALUATION_QUEUE)

    async def outstanding_count(self) -> int:
        """Count pending and running evaluations that consume queue capacity."""

        value = await self._session.scalar(
            select(func.count(EvaluationRun.id)).where(
                EvaluationRun.status.in_(
                    (
                        EvaluationRunStatus.PENDING.value,
                        EvaluationRunStatus.RUNNING.value,
                    )
                )
            )
        )
        return int(value or 0)

    async def claim_next(
        self,
        *,
        claimed_at: datetime | None = None,
        stale_before: datetime | None = None,
        max_running: int | None = None,
    ) -> EvaluationRun | None:
        """Atomically claim a pending or explicitly stale running job with SKIP LOCKED."""
        timestamp = _utc(claimed_at or datetime.now(UTC))
        if max_running is not None:
            if max_running < 1:
                raise ValueError("max_running must be positive")
            await self.acquire_queue_lock()
            live_running = EvaluationRun.status == EvaluationRunStatus.RUNNING.value
            if stale_before is not None:
                live_running = live_running & (
                    func.coalesce(EvaluationRun.heartbeat_at, EvaluationRun.started_at)
                    >= _utc(stale_before)
                )
            running_count = await self._session.scalar(
                select(func.count(EvaluationRun.id)).where(live_running)
            )
            if int(running_count or 0) >= max_running:
                return None
        pending = EvaluationRun.status == EvaluationRunStatus.PENDING.value
        eligible = pending
        stale_running = None
        if stale_before is not None:
            stale_running = (EvaluationRun.status == EvaluationRunStatus.RUNNING.value) & (
                func.coalesce(EvaluationRun.heartbeat_at, EvaluationRun.started_at)
                < _utc(stale_before)
            )
            eligible = or_(
                eligible,
                stale_running,
            )
        candidate_query = select(EvaluationRun.id).where(eligible)
        if stale_running is not None:
            candidate_query = candidate_query.order_by(case((stale_running, 0), else_=1))
        candidate = (
            candidate_query.order_by(EvaluationRun.started_at, EvaluationRun.id)
            .with_for_update(skip_locked=True)
            .limit(1)
            .scalar_subquery()
        )
        result = await self._session.execute(
            update(EvaluationRun)
            .where(EvaluationRun.id == candidate)
            .values(
                status=EvaluationRunStatus.RUNNING.value,
                started_at=timestamp,
                heartbeat_at=timestamp,
                finished_at=None,
                summary={},
                report_path=None,
            )
            .returning(EvaluationRun)
        )
        return result.scalar_one_or_none()

    async def heartbeat(
        self,
        run_id: UUID,
        *,
        lease_started_at: datetime,
        heartbeat_at: datetime | None = None,
    ) -> bool:
        """Refresh a running lease only when this worker still owns its claim token."""

        timestamp = _utc(heartbeat_at or datetime.now(UTC))
        result = await self._session.execute(
            update(EvaluationRun)
            .where(
                EvaluationRun.id == run_id,
                EvaluationRun.status == EvaluationRunStatus.RUNNING.value,
                EvaluationRun.started_at == _utc(lease_started_at),
            )
            .values(heartbeat_at=timestamp)
        )
        return bool(cast(CursorResult[Any], result).rowcount)

    async def upsert_cases(
        self,
        records: Sequence[EvaluationCaseUpsert],
    ) -> dict[str, UUID]:
        """Upsert a bounded dataset batch and return external-id to database-id mapping."""
        if not records:
            return {}
        if len(records) > MAX_CASE_BATCH_SIZE:
            raise ValueError(f"evaluation case batch cannot exceed {MAX_CASE_BATCH_SIZE}")
        identities = [(record.dataset_name, record.external_id) for record in records]
        if any(not dataset_name or not external_id for dataset_name, external_id in identities):
            raise ValueError("evaluation case dataset_name and external_id cannot be blank")
        if len({dataset_name for dataset_name, _ in identities}) != 1:
            raise ValueError("evaluation case batch must belong to one dataset")
        if len(set(identities)) != len(identities):
            raise ValueError("evaluation case batch contains duplicate dataset identities")

        values = [
            {
                "id": uuid4(),
                "external_id": record.external_id,
                "dataset_name": record.dataset_name,
                "question": record.question,
                "conversation_history": record.conversation_history,
                "reference_answer": record.reference_answer,
                "relevant_chunk_ids": record.relevant_chunk_ids,
                "expected_citations": record.expected_citations,
                "answerable": record.answerable,
                "difficulty": record.difficulty,
                "category": record.category,
                "source_type": record.source_type,
                "metadata": record.metadata,
                "human_reviewed": record.human_reviewed,
            }
            for record in records
        ]
        dialect_name = self._session.get_bind().dialect.name
        table = cast(Table, EvaluationCaseModel.__table__)
        statement: Any
        if dialect_name == "sqlite":
            statement = sqlite_insert(table).values(values)
        else:
            statement = postgresql_insert(table).values(values)
        excluded = statement.excluded
        statement = statement.on_conflict_do_update(
            index_elements=[
                table.c.dataset_name,
                table.c.external_id,
            ],
            set_={
                "question": excluded.question,
                "conversation_history": excluded.conversation_history,
                "reference_answer": excluded.reference_answer,
                "relevant_chunk_ids": excluded.relevant_chunk_ids,
                "expected_citations": excluded.expected_citations,
                "answerable": excluded.answerable,
                "difficulty": excluded.difficulty,
                "category": excluded.category,
                "source_type": excluded.source_type,
                "metadata": excluded.metadata,
                "human_reviewed": excluded.human_reviewed,
                "updated_at": func.now(),
            },
        ).returning(
            table.c.external_id,
            table.c.id,
        )
        rows = (await self._session.execute(statement)).all()
        mapping = {str(external_id): cast(UUID, case_id) for external_id, case_id in rows}
        if len(mapping) != len(records):
            raise RuntimeError("evaluation case upsert returned an incomplete identity mapping")
        return mapping

    async def add_report_results(
        self,
        run_id: UUID,
        *,
        dataset_name: str,
        records: Sequence[EvaluationResultRecord],
    ) -> list[EvaluationResult]:
        """Persist case snapshots and linked outputs with two batched SQL statements."""
        if not records:
            return []
        if len(records) > MAX_RESULT_BATCH_SIZE:
            raise ValueError(f"evaluation result batch cannot exceed {MAX_RESULT_BATCH_SIZE}")
        case_ids = await self.upsert_cases(
            [EvaluationCaseUpsert.from_case(dataset_name, record.case) for record in records]
        )
        return await self.add_results(
            run_id,
            [
                EvaluationResultCreate.from_record(
                    record,
                    evaluation_case_id=case_ids[record.case.id],
                )
                for record in records
            ],
        )

    async def add_results(
        self,
        run_id: UUID,
        records: Sequence[EvaluationResultCreate],
    ) -> list[EvaluationResult]:
        """Insert a bounded result batch in one statement and without committing."""
        if not records:
            return []
        if len(records) > MAX_RESULT_BATCH_SIZE:
            raise ValueError(f"evaluation result batch cannot exceed {MAX_RESULT_BATCH_SIZE}")
        statement = (
            insert(EvaluationResult)
            .values(
                [
                    {
                        "id": record.id,
                        "evaluation_run_id": run_id,
                        "evaluation_case_id": record.evaluation_case_id,
                        "case_external_id": record.case_external_id,
                        "question": record.question,
                        "generated_answer": record.generated_answer,
                        "predicted_answerable": record.predicted_answerable,
                        "retrieved_results": record.retrieved_results,
                        "citations": record.citations,
                        "metrics": record.metrics,
                        "latency": record.latency,
                        "usage": record.usage,
                        "error": record.error,
                    }
                    for record in records
                ]
            )
            .returning(EvaluationResult)
        )
        return list(await self._session.scalars(statement))

    async def mark_succeeded(
        self,
        run_id: UUID,
        *,
        summary: Mapping[str, Any],
        report_path: str,
        lease_started_at: datetime,
        finished_at: datetime | None = None,
    ) -> bool:
        """Complete a running run through a compare-and-swap update."""
        return await self._finish(
            run_id,
            status=EvaluationRunStatus.SUCCEEDED,
            summary=summary,
            report_path=report_path,
            lease_started_at=lease_started_at,
            finished_at=finished_at,
        )

    async def mark_failed(
        self,
        run_id: UUID,
        *,
        error: str,
        lease_started_at: datetime,
        summary: Mapping[str, Any] | None = None,
        finished_at: datetime | None = None,
    ) -> bool:
        """Record a bounded public failure reason for a pending or running run."""
        failure_summary = dict(summary or {})
        failure_summary["error"] = error[:4_000]
        timestamp = _utc(finished_at or datetime.now(UTC))
        result = await self._session.execute(
            update(EvaluationRun)
            .where(
                EvaluationRun.id == run_id,
                EvaluationRun.status == EvaluationRunStatus.RUNNING.value,
                EvaluationRun.started_at == _utc(lease_started_at),
            )
            .values(
                status=EvaluationRunStatus.FAILED.value,
                summary=failure_summary,
                report_path=None,
                finished_at=timestamp,
                heartbeat_at=timestamp,
            )
        )
        return bool(cast(CursorResult[Any], result).rowcount)

    async def _finish(
        self,
        run_id: UUID,
        *,
        status: EvaluationRunStatus,
        summary: Mapping[str, Any],
        report_path: str,
        lease_started_at: datetime,
        finished_at: datetime | None,
    ) -> bool:
        timestamp = _utc(finished_at or datetime.now(UTC))
        result = await self._session.execute(
            update(EvaluationRun)
            .where(
                EvaluationRun.id == run_id,
                EvaluationRun.status == EvaluationRunStatus.RUNNING.value,
                EvaluationRun.started_at == _utc(lease_started_at),
            )
            .values(
                status=status.value,
                summary=dict(summary),
                report_path=report_path,
                finished_at=timestamp,
                heartbeat_at=timestamp,
            )
        )
        return bool(cast(CursorResult[Any], result).rowcount)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("evaluation timestamps must be timezone-aware")
    return value.astimezone(UTC)
