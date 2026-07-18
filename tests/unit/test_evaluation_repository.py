"""Evaluation persistence tests using real SQL statements and transaction control."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from app.db.models.evaluation import EvaluationCase as EvaluationCaseModel
from app.db.models.evaluation import EvaluationResult, EvaluationRun
from app.domain.evaluation import Difficulty, EvaluationCase
from app.repositories import evaluation_repository as repository_module
from app.repositories.evaluation_repository import (
    MAX_RESULT_BATCH_SIZE,
    EvaluationRepository,
    EvaluationResultCreate,
)
from evaluation.models import (
    EvaluationCitationSummary,
    EvaluationResultRecord,
)


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(_type: JSONB, _compiler: Any, **_kwargs: Any) -> str:
    """Allow repository SQL to run against an isolated SQLite unit-test database."""
    return "JSON"


class AsyncSessionBridge:
    """Exercise the async repository contract through a real synchronous session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, instance: object) -> None:
        self._session.add(instance)

    async def flush(self) -> None:
        self._session.flush()

    async def get(self, entity: type[Any], identity: object) -> Any:
        return self._session.get(entity, identity)

    async def execute(self, statement: Any) -> Any:
        return self._session.execute(statement)

    async def scalar(self, statement: Any) -> Any:
        return self._session.scalar(statement)

    async def scalars(self, statement: Any) -> Any:
        return self._session.scalars(statement)

    def get_bind(self) -> Any:
        return self._session.get_bind()


def _repository(session: Session) -> EvaluationRepository:
    return EvaluationRepository(cast(AsyncSession, AsyncSessionBridge(session)))


@pytest.mark.asyncio
async def test_repository_leaves_transaction_ownership_to_caller() -> None:
    engine = create_engine("sqlite://")
    EvaluationRun.__table__.create(engine)
    run_id = uuid4()

    with Session(engine) as session:
        await _repository(session).create_run(
            run_id=run_id,
            dataset_name="rollback-test",
            config_snapshot={"mode": "hybrid"},
            queued_at=datetime(2026, 7, 18, tzinfo=UTC),
        )
        session.rollback()

    with Session(engine) as session:
        assert await _repository(session).get(run_id) is None
    engine.dispose()


@pytest.mark.asyncio
async def test_run_queue_lifecycle_and_result_batch_are_persisted() -> None:
    engine = create_engine("sqlite://")
    EvaluationCaseModel.__table__.create(engine)
    EvaluationRun.__table__.create(engine)
    EvaluationResult.__table__.create(engine)
    queued_at = datetime(2026, 7, 18, 1, tzinfo=UTC)
    claimed_at = datetime(2026, 7, 18, 2, tzinfo=UTC)
    finished_at = datetime(2026, 7, 18, 3, tzinfo=UTC)

    with Session(engine) as session:
        repository = _repository(session)
        run = await repository.create_run(
            dataset_name="kubernetes-eval",
            config_snapshot={"retrieval_mode": "hybrid"},
            queued_at=queued_at,
        )
        run_id = run.id
        session.commit()

        claimed = await repository.claim_next(claimed_at=claimed_at)
        assert claimed is not None
        assert claimed.id == run_id
        assert claimed.status == "running"
        # SQLite drops timezone metadata; PostgreSQL preserves the UTC-aware value.
        assert claimed.started_at.replace(tzinfo=UTC) == claimed_at
        assert claimed.heartbeat_at is not None

        heartbeat_at = datetime(2026, 7, 18, 2, 30, tzinfo=UTC)
        assert await repository.heartbeat(
            run_id,
            lease_started_at=claimed_at,
            heartbeat_at=heartbeat_at,
        )

        records = [
            EvaluationResultCreate(
                case_external_id="k8s-001",
                question="How do I roll back a Deployment?",
                generated_answer="Use rollout undo. [1]",
                predicted_answerable=True,
                retrieved_results=[{"chunk_id": "chunk-1", "rank": 1}],
                citations={"citation_ids": ["chunk-1"], "validity": [True]},
                metrics={"recall_at_1": 1.0, "token_f1": 0.8},
                latency={"total_ms": 25.0},
                usage={"total_tokens": 42},
            )
        ]
        inserted = await repository.add_results(run_id, records)
        assert [item.case_external_id for item in inserted] == ["k8s-001"]
        assert inserted[0].citations["validity"] == [True]
        assert await repository.result_count(run_id) == 1

        assert await repository.mark_succeeded(
            run_id,
            summary={"recall_at_1": 1.0},
            report_path="evaluation/reports/run/report.json",
            lease_started_at=claimed_at,
            finished_at=finished_at,
        )
        assert not await repository.mark_succeeded(
            run_id,
            summary={},
            report_path="evaluation/reports/duplicate/report.json",
            lease_started_at=claimed_at,
            finished_at=finished_at,
        )
        session.commit()

    with Session(engine) as session:
        persisted = await _repository(session).get(run_id)
        assert persisted is not None
        assert persisted.status == "succeeded"
        assert persisted.summary == {"recall_at_1": 1.0}
        assert persisted.finished_at is not None
        assert persisted.finished_at.replace(tzinfo=UTC) == finished_at
    engine.dispose()


def _result_record(
    *,
    external_id: str = "k8s-001",
    reference_answer: str = "Use rollout undo.",
) -> EvaluationResultRecord:
    case = EvaluationCase(
        id=external_id,
        question="How do I roll back a Deployment?",
        conversation_history=[
            {"role": "user", "content": "A rollout failed."},
        ],
        reference_answer=reference_answer,
        relevant_chunk_ids=["chunk-1"],
        expected_citations=["chunk-1"],
        answerable=True,
        difficulty=Difficulty.MEDIUM,
        category="how_to",
        source_type="official_documentation",
        metadata={"kubernetes_version": "1.34"},
        human_reviewed=True,
    )
    return EvaluationResultRecord(
        case=case,
        generated_answer="Use rollout undo. [1]",
        rewritten_query=case.question,
        predicted_answerable=True,
        retrieved_evidence=[],
        metrics={"recall_at_1": 1.0},
        citations=EvaluationCitationSummary(),
    )


@pytest.mark.asyncio
async def test_report_results_batch_upserts_complete_cases_and_links_foreign_keys() -> None:
    engine = create_engine("sqlite://")
    EvaluationCaseModel.__table__.create(engine)
    EvaluationRun.__table__.create(engine)
    EvaluationResult.__table__.create(engine)
    claimed_at = datetime(2026, 7, 18, 2, tzinfo=UTC)

    with Session(engine) as session:
        repository = _repository(session)
        run = await repository.create_run(
            dataset_name="kubernetes-eval",
            config_snapshot={},
            queued_at=datetime(2026, 7, 18, 1, tzinfo=UTC),
        )
        claimed = await repository.claim_next(claimed_at=claimed_at)
        assert claimed is not None

        inserted = await repository.add_report_results(
            run.id,
            dataset_name=run.dataset_name,
            records=[_result_record()],
        )
        case = session.scalar(select(EvaluationCaseModel))
        assert case is not None
        assert case.external_id == "k8s-001"
        assert case.reference_answer == "Use rollout undo."
        assert case.conversation_history == [{"role": "user", "content": "A rollout failed."}]
        assert case.relevant_chunk_ids == ["chunk-1"]
        assert case.expected_citations == ["chunk-1"]
        assert case.source_type == "official_documentation"
        assert case.human_reviewed is True
        assert case.metadata_ == {"kubernetes_version": "1.34"}
        assert inserted[0].evaluation_case_id == case.id
        assert inserted[0].metrics["_evaluation"]["case"]["id"] == "k8s-001"

        mapping = await repository.upsert_cases(
            [
                repository_module.EvaluationCaseUpsert.from_case(
                    run.dataset_name,
                    _result_record(reference_answer="Updated reference.").case,
                )
            ]
        )
        assert mapping == {"k8s-001": case.id}
        assert session.scalar(select(EvaluationCaseModel.reference_answer)) == "Updated reference."
        assert len(session.scalars(select(EvaluationCaseModel)).all()) == 1

    engine.dispose()


@pytest.mark.asyncio
async def test_report_persistence_uses_two_batch_inserts_instead_of_per_case_sql() -> None:
    engine = create_engine("sqlite://")
    EvaluationCaseModel.__table__.create(engine)
    EvaluationRun.__table__.create(engine)
    EvaluationResult.__table__.create(engine)
    statements: list[str] = []

    def capture_statement(
        _connection: Any,
        _cursor: Any,
        statement: str,
        _parameters: Any,
        _context: Any,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture_statement)
    with Session(engine) as session:
        repository = _repository(session)
        run = await repository.create_run(
            dataset_name="kubernetes-eval",
            config_snapshot={},
            queued_at=datetime(2026, 7, 18, 1, tzinfo=UTC),
        )
        statements.clear()
        await repository.add_report_results(
            run.id,
            dataset_name=run.dataset_name,
            records=[
                _result_record(external_id="k8s-001"),
                _result_record(external_id="k8s-002"),
                _result_record(external_id="k8s-003"),
            ],
        )

    inserts = [
        statement
        for statement in statements
        if statement.startswith("INSERT INTO evaluation_cases")
        or statement.startswith("INSERT INTO evaluation_results")
    ]
    assert len(inserts) == 2
    assert inserts[0].count("VALUES") == 1
    assert inserts[1].count("VALUES") == 1
    engine.dispose()


@pytest.mark.asyncio
async def test_stale_reclaim_rejects_the_original_worker_lease() -> None:
    engine = create_engine("sqlite://")
    EvaluationRun.__table__.create(engine)
    run_id = uuid4()
    first_claim = datetime(2026, 7, 18, 1, tzinfo=UTC)
    first_heartbeat = datetime(2026, 7, 18, 1, 5, tzinfo=UTC)
    second_claim = datetime(2026, 7, 18, 2, tzinfo=UTC)

    with Session(engine) as session:
        repository = _repository(session)
        await repository.create_run(
            run_id=run_id,
            dataset_name="lease-test",
            config_snapshot={},
            queued_at=datetime(2026, 7, 18, tzinfo=UTC),
        )
        claimed = await repository.claim_next(claimed_at=first_claim)
        assert claimed is not None
        assert await repository.heartbeat(
            run_id,
            lease_started_at=first_claim,
            heartbeat_at=first_heartbeat,
        )

        # A stale cutoff before the latest heartbeat cannot steal the lease.
        assert (
            await repository.claim_next(
                claimed_at=second_claim,
                stale_before=datetime(2026, 7, 18, 1, 4, tzinfo=UTC),
            )
            is None
        )
        reclaimed = await repository.claim_next(
            claimed_at=second_claim,
            stale_before=datetime(2026, 7, 18, 1, 6, tzinfo=UTC),
        )
        assert reclaimed is not None
        assert reclaimed.started_at.replace(tzinfo=UTC) == second_claim

        assert not await repository.mark_failed(
            run_id,
            error="late original worker",
            lease_started_at=first_claim,
        )
        assert not await repository.mark_succeeded(
            run_id,
            summary={"worker": "old"},
            report_path="evaluation/reports/old/report.json",
            lease_started_at=first_claim,
        )
        assert await repository.mark_succeeded(
            run_id,
            summary={"worker": "new"},
            report_path="evaluation/reports/new/report.json",
            lease_started_at=second_claim,
        )

    engine.dispose()


@pytest.mark.asyncio
async def test_result_batch_is_bounded_before_any_database_call() -> None:
    record = EvaluationResultCreate(
        case_external_id="case",
        question="question",
        generated_answer="answer",
        predicted_answerable=True,
    )
    session = cast(AsyncSession, object())
    repository = EvaluationRepository(session)

    with pytest.raises(ValueError, match="cannot exceed"):
        await repository.add_results(uuid4(), [record] * (MAX_RESULT_BATCH_SIZE + 1))


@pytest.mark.asyncio
async def test_global_running_limit_blocks_another_claim_until_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_op_queue_lock(_session: AsyncSession, _queue_name: str) -> None:
        return None

    monkeypatch.setattr(repository_module, "acquire_queue_advisory_lock", no_op_queue_lock)
    engine = create_engine("sqlite://")
    EvaluationRun.__table__.create(engine)
    first_claim = datetime(2026, 7, 18, 1, tzinfo=UTC)
    second_claim = datetime(2026, 7, 18, 2, tzinfo=UTC)

    with Session(engine) as session:
        repository = _repository(session)
        await repository.create_run(
            dataset_name="first",
            config_snapshot={},
            queued_at=datetime(2026, 7, 18, 0, 0, tzinfo=UTC),
        )
        await repository.create_run(
            dataset_name="second",
            config_snapshot={},
            queued_at=datetime(2026, 7, 18, 0, 1, tzinfo=UTC),
        )
        first = await repository.claim_next(
            claimed_at=first_claim,
            stale_before=datetime(2026, 7, 18, 0, 30, tzinfo=UTC),
            max_running=1,
        )
        assert first is not None
        assert (
            await repository.claim_next(
                claimed_at=second_claim,
                stale_before=datetime(2026, 7, 18, 0, 30, tzinfo=UTC),
                max_running=1,
            )
            is None
        )
        assert await repository.mark_succeeded(
            first.id,
            summary={},
            report_path="evaluation/reports/first/report.json",
            lease_started_at=first_claim,
        )
        following = await repository.claim_next(
            claimed_at=second_claim,
            stale_before=datetime(2026, 7, 18, 0, 30, tzinfo=UTC),
            max_running=1,
        )
        assert following is not None
        assert following.dataset_name == "second"

    engine.dispose()
