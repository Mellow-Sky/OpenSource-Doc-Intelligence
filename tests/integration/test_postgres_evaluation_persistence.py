"""PostgreSQL coverage for the production case upsert and result FK path."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models.evaluation import EvaluationCase as EvaluationCaseModel
from app.db.models.evaluation import EvaluationResult, EvaluationRun
from app.domain.evaluation import Difficulty, EvaluationCase
from app.repositories.evaluation_repository import EvaluationRepository
from evaluation.models import EvaluationCitationSummary, EvaluationResultRecord

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is required for PostgreSQL evaluation persistence tests",
    ),
]


def _record(external_id: str, reference_answer: str) -> EvaluationResultRecord:
    case = EvaluationCase(
        id=external_id,
        question="How do I roll back a Deployment?",
        reference_answer=reference_answer,
        relevant_chunk_ids=["chunk-1"],
        expected_citations=["chunk-1"],
        answerable=True,
        category="how_to",
        difficulty=Difficulty.MEDIUM,
        source_type="official_documentation",
        metadata={"version": "1.34"},
        human_reviewed=True,
    )
    return EvaluationResultRecord(
        case=case,
        generated_answer="Use kubectl rollout undo. [1]",
        rewritten_query=case.question,
        predicted_answerable=True,
        retrieved_evidence=[],
        metrics={"recall_at_1": 1.0},
        citations=EvaluationCitationSummary(),
    )


@pytest.mark.asyncio
async def test_postgres_case_upsert_is_stable_and_every_result_has_the_case_fk() -> None:
    assert TEST_DATABASE_URL is not None
    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    marker = str(uuid4())
    dataset_name = f"evaluation-persistence-{marker}"
    external_id = f"case-{marker}"
    run_ids = []
    try:
        async with factory.begin() as session:
            repository = EvaluationRepository(session)
            for index, reference in enumerate(("first", "updated"), start=1):
                run = await repository.create_run(
                    dataset_name=dataset_name,
                    config_snapshot={},
                    queued_at=datetime(2026, 7, 18, index, tzinfo=UTC),
                )
                run_ids.append(run.id)
                await repository.add_report_results(
                    run.id,
                    dataset_name=dataset_name,
                    records=[_record(external_id, reference)],
                )

        async with factory() as session:
            case = await session.scalar(
                select(EvaluationCaseModel).where(
                    EvaluationCaseModel.dataset_name == dataset_name,
                    EvaluationCaseModel.external_id == external_id,
                )
            )
            assert case is not None
            assert case.reference_answer == "updated"
            assert case.source_type == "official_documentation"
            assert case.human_reviewed is True
            assert (
                await session.scalar(
                    select(func.count(EvaluationCaseModel.id)).where(
                        EvaluationCaseModel.dataset_name == dataset_name
                    )
                )
                == 1
            )
            case_ids = list(
                await session.scalars(
                    select(EvaluationResult.evaluation_case_id).where(
                        EvaluationResult.evaluation_run_id.in_(run_ids)
                    )
                )
            )
            assert case_ids == [case.id, case.id]
    finally:
        async with factory.begin() as session:
            await session.execute(delete(EvaluationRun).where(EvaluationRun.id.in_(run_ids)))
            await session.execute(
                delete(EvaluationCaseModel).where(EvaluationCaseModel.dataset_name == dataset_name)
            )
        await engine.dispose()
