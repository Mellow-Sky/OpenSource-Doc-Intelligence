"""Real-SQL tests for service transactions and worker case/result linkage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from app.container import AppContainer
from app.core.config import Settings
from app.db.models.evaluation import EvaluationCase as EvaluationCaseModel
from app.db.models.evaluation import EvaluationResult, EvaluationRun
from app.db.session import Database
from app.domain.evaluation import Difficulty, EvaluationCase
from app.evaluation_worker import EvaluationWorker
from app.repositories import evaluation_repository as repository_module
from app.repositories.evaluation_repository import EvaluationRepository
from app.services.evaluation_service import EvaluationLeaseLostError, EvaluationService
from evaluation.models import (
    EvaluationCitationSummary,
    EvaluationResultRecord,
    EvaluationRunReport,
)


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(_type: JSONB, _compiler: Any, **_kwargs: Any) -> str:
    return "JSON"


@dataclass
class _Database:
    session_factory: async_sessionmaker[AsyncSession]

    async def close(self) -> None:
        return None


class _AsyncSessionBridge:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, instance: object) -> None:
        self._session.add(instance)

    async def flush(self) -> None:
        self._session.flush()

    async def get(self, entity: type[Any], identity: object) -> Any:
        return self._session.get(entity, identity)

    async def execute(self, statement: Any) -> Any:
        return _ResultBridge(self._session.execute(statement))

    async def scalar(self, statement: Any) -> Any:
        return self._session.scalar(statement)

    async def scalars(self, statement: Any) -> Any:
        return self._session.scalars(statement)

    def get_bind(self) -> Any:
        return self._session.get_bind()


class _ResultBridge:
    """Restore timezone metadata SQLite discards from DateTime columns."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def scalar_one_or_none(self) -> Any:
        value = self._result.scalar_one_or_none()
        if (
            isinstance(value, EvaluationRun)
            and value.started_at is not None
            and value.started_at.tzinfo is None
        ):
            value.started_at = value.started_at.replace(tzinfo=UTC)
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)


class _SessionContext:
    def __init__(self, engine: Any) -> None:
        self._session = Session(engine, expire_on_commit=False)

    async def __aenter__(self) -> AsyncSession:
        return cast(AsyncSession, _AsyncSessionBridge(self._session))

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        if exc_type is None:
            self._session.commit()
        else:
            self._session.rollback()
        self._session.close()


class _SessionFactory:
    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def begin(self) -> _SessionContext:
        return _SessionContext(self._engine)

    def __call__(self) -> _SessionContext:
        return _SessionContext(self._engine)


def _database(path: Path) -> tuple[Any, _Database]:
    engine = create_engine(f"sqlite:///{path}")
    EvaluationCaseModel.__table__.create(engine)
    EvaluationRun.__table__.create(engine)
    EvaluationResult.__table__.create(engine)
    factory = cast(async_sessionmaker[AsyncSession], _SessionFactory(engine))
    return engine, _Database(factory)


def _report(run_id: str) -> EvaluationRunReport:
    case = EvaluationCase(
        id="k8s-001",
        question="How do I roll back a Deployment?",
        conversation_history=[],
        reference_answer="Use kubectl rollout undo.",
        relevant_chunk_ids=["chunk-1"],
        expected_citations=["chunk-1"],
        answerable=True,
        category="how_to",
        difficulty=Difficulty.MEDIUM,
        source_type="official_documentation",
        metadata={"version": "1.34"},
        human_reviewed=True,
    )
    result = EvaluationResultRecord(
        case=case,
        generated_answer="Use kubectl rollout undo. [1]",
        rewritten_query=case.question,
        predicted_answerable=True,
        retrieved_evidence=[],
        metrics={"recall_at_1": 1.0},
        citations=EvaluationCitationSummary(),
    )
    started_at = datetime(2026, 7, 18, 1, tzinfo=UTC)
    return EvaluationRunReport(
        run_id=run_id,
        experiment_name="hybrid",
        dataset_name="kubernetes-eval",
        dataset_path="evaluation/datasets/kubernetes_eval.jsonl",
        dataset_fingerprint="f" * 64,
        dataset_size=1,
        started_at=started_at,
        finished_at=datetime(2026, 7, 18, 1, 1, tzinfo=UTC),
        elapsed_seconds=60,
        config_snapshot={},
        summary={"recall_at_1": 1.0},
        category_metrics={},
        difficulty_metrics={},
        answerability_groups={},
        results=[result],
    )


@pytest.mark.asyncio
async def test_service_rolls_back_case_and_result_when_lease_is_lost(tmp_path: Path) -> None:
    engine, database = _database(tmp_path / "service.sqlite")
    claimed_at = datetime(2026, 7, 18, 2, tzinfo=UTC)
    async with database.session_factory.begin() as session:
        repository = EvaluationRepository(session)
        run = await repository.create_run(
            dataset_name="kubernetes-eval",
            config_snapshot={},
            queued_at=datetime(2026, 7, 18, 1, tzinfo=UTC),
        )
        claimed = await repository.claim_next(claimed_at=claimed_at)
        assert claimed is not None

    service = EvaluationService(database.session_factory)
    with pytest.raises(EvaluationLeaseLostError):
        await service.complete(
            run,
            _report(str(run.id)),
            report_path=tmp_path / "report.json",
            lease_started_at=datetime(2026, 7, 18, 3, tzinfo=UTC),
        )

    with Session(engine) as session:
        assert session.scalar(select(func.count(EvaluationCaseModel.id))) == 0
        assert session.scalar(select(func.count(EvaluationResult.id))) == 0
    engine.dispose()


@pytest.mark.asyncio
async def test_worker_persists_one_stable_case_and_links_result_fk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_op_queue_lock(_session: AsyncSession, _queue_name: str) -> None:
        return None

    monkeypatch.setattr(repository_module, "acquire_queue_advisory_lock", no_op_queue_lock)
    engine, database = _database(tmp_path / "worker.sqlite")
    settings = Settings(
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'worker.sqlite'}",
    )
    async with database.session_factory.begin() as session:
        run = await EvaluationRepository(session).create_run(
            dataset_name="kubernetes-eval",
            config_snapshot={},
            queued_at=datetime(2026, 7, 18, 1, tzinfo=UTC),
        )

    container = AppContainer(
        settings=settings,
        database=cast(Database, database),
    )
    worker = EvaluationWorker(container, report_root=tmp_path / "reports")

    async def execute_report(
        _run: EvaluationRun,
        *,
        lease_started_at: datetime,
        output_directory: Path,
    ) -> EvaluationRunReport:
        assert lease_started_at.tzinfo is not None
        assert output_directory.is_relative_to(tmp_path / "reports")
        return _report(str(_run.id))

    monkeypatch.setattr(worker, "_execute_with_heartbeat", execute_report)
    assert await worker.run_once() is True

    with Session(engine) as session:
        case = session.scalar(select(EvaluationCaseModel))
        result = session.scalar(select(EvaluationResult))
        persisted_run = session.get(EvaluationRun, run.id)
        assert case is not None
        assert result is not None
        assert persisted_run is not None
        assert result.evaluation_case_id == case.id
        assert result.case_external_id == case.external_id == "k8s-001"
        assert case.reference_answer == "Use kubectl rollout undo."
        assert case.source_type == "official_documentation"
        assert case.human_reviewed is True
        assert persisted_run.status == "succeeded"
        assert persisted_run.summary == {"recall_at_1": 1.0}
    engine.dispose()
