"""Audit persistence for retrieval runs, ranked candidates, and citations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.retrieval import AnswerCitation, RetrievalResult, RetrievalRun
from app.domain.citations import Citation
from app.domain.retrieval import RetrievalCandidate


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalRunCreate:
    """Complete trace-level fields for a retrieval execution."""

    message_id: UUID
    query: str
    rewritten_query: str
    filters: dict[str, Any] = field(default_factory=dict)
    keyword_latency_ms: float = 0.0
    vector_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    retrieved_count: int = 0
    reranked_count: int = 0
    no_answer_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.query.strip() or not self.rewritten_query.strip():
            raise ValueError("retrieval queries must not be blank")
        timings = (
            self.keyword_latency_ms,
            self.vector_latency_ms,
            self.rerank_latency_ms,
            self.total_latency_ms,
        )
        if any(value < 0 for value in timings):
            raise ValueError("retrieval timings must be non-negative")
        if self.retrieved_count < 0 or self.reranked_count < 0:
            raise ValueError("retrieval counts must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalResultCreate:
    """One candidate's complete channel, fusion, and reranker rank trace."""

    retrieval_run_id: UUID
    chunk_id: UUID
    keyword_rank: int | None = None
    vector_rank: int | None = None
    fused_rank: int | None = None
    rerank_rank: int | None = None
    keyword_score: float | None = None
    vector_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    selected_for_context: bool = False
    id: UUID = field(default_factory=uuid4)

    @classmethod
    def from_candidate(
        cls,
        retrieval_run_id: UUID,
        candidate: RetrievalCandidate,
    ) -> RetrievalResultCreate:
        """Create a persistence command without discarding any ranking channel."""
        return cls(
            retrieval_run_id=retrieval_run_id,
            chunk_id=candidate.chunk_id,
            keyword_rank=candidate.keyword_rank,
            vector_rank=candidate.vector_rank,
            fused_rank=candidate.fused_rank,
            rerank_rank=candidate.rerank_rank,
            keyword_score=candidate.keyword_score,
            vector_score=candidate.vector_score,
            fused_score=candidate.fused_score,
            rerank_score=candidate.rerank_score,
            selected_for_context=candidate.selected_for_context,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerCitationCreate:
    """One answer claim and the exact chunk cited to support it."""

    message_id: UUID
    chunk_id: UUID
    citation_number: int
    quoted_text: str
    claim_text: str
    citation_valid: bool | None = None
    validation_score: float | None = None
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.citation_number < 1:
            raise ValueError("citation_number must be positive")
        if not self.quoted_text:
            raise ValueError("quoted_text must not be empty")
        if self.validation_score is not None and not 0 <= self.validation_score <= 1:
            raise ValueError("validation_score must be between 0 and 1")

    @classmethod
    def from_citation(
        cls,
        message_id: UUID,
        citation: Citation,
        *,
        claim_text: str,
    ) -> AnswerCitationCreate:
        """Convert a validated public citation into its audit record."""
        return cls(
            message_id=message_id,
            chunk_id=citation.chunk_id,
            citation_number=citation.number,
            quoted_text=citation.quoted_text,
            claim_text=claim_text,
            citation_valid=citation.valid,
            validation_score=citation.validation_score,
        )


@dataclass(frozen=True, slots=True)
class CompleteRetrievalRun:
    """A retrieval run with preloaded results and message citations."""

    run: RetrievalRun
    results: list[RetrievalResult]
    citations: list[AnswerCitation]


class RetrievalRunRepository:
    """Write a complete retrieval audit trail without committing the unit of work."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_run(
        self,
        record: RetrievalRunCreate,
        *,
        created_at: datetime | None = None,
    ) -> RetrievalRun:
        """Insert one run with every configured filter, timing, count, and score."""
        timestamp = _utc(created_at)
        statement = (
            insert(RetrievalRun)
            .values(
                id=record.id,
                message_id=record.message_id,
                query=record.query,
                rewritten_query=record.rewritten_query,
                filters=dict(record.filters),
                keyword_latency_ms=record.keyword_latency_ms,
                vector_latency_ms=record.vector_latency_ms,
                rerank_latency_ms=record.rerank_latency_ms,
                total_latency_ms=record.total_latency_ms,
                retrieved_count=record.retrieved_count,
                reranked_count=record.reranked_count,
                no_answer_score=record.no_answer_score,
                metadata=dict(record.metadata),
                created_at=timestamp,
                updated_at=timestamp,
            )
            .returning(RetrievalRun)
        )
        return (await self._session.scalars(statement)).one()

    async def add_results(
        self,
        records: Sequence[RetrievalResultCreate],
    ) -> list[RetrievalResult]:
        """Insert ranked candidates in one SQL statement."""
        if not records:
            return []
        statement = (
            insert(RetrievalResult)
            .values(
                [
                    {
                        "id": record.id,
                        "retrieval_run_id": record.retrieval_run_id,
                        "chunk_id": record.chunk_id,
                        "keyword_rank": record.keyword_rank,
                        "vector_rank": record.vector_rank,
                        "fused_rank": record.fused_rank,
                        "rerank_rank": record.rerank_rank,
                        "keyword_score": record.keyword_score,
                        "vector_score": record.vector_score,
                        "fused_score": record.fused_score,
                        "rerank_score": record.rerank_score,
                        "selected_for_context": record.selected_for_context,
                    }
                    for record in records
                ]
            )
            .returning(RetrievalResult)
        )
        return list(await self._session.scalars(statement))

    async def add_citations(
        self,
        records: Sequence[AnswerCitationCreate],
    ) -> list[AnswerCitation]:
        """Insert answer citations in one SQL statement."""
        if not records:
            return []
        statement = (
            insert(AnswerCitation)
            .values(
                [
                    {
                        "id": record.id,
                        "message_id": record.message_id,
                        "chunk_id": record.chunk_id,
                        "citation_number": record.citation_number,
                        "quoted_text": record.quoted_text,
                        "claim_text": record.claim_text,
                        "citation_valid": record.citation_valid,
                        "validation_score": record.validation_score,
                    }
                    for record in records
                ]
            )
            .returning(AnswerCitation)
        )
        return list(await self._session.scalars(statement))

    async def get_complete(self, retrieval_run_id: UUID) -> CompleteRetrievalRun | None:
        """Load a run, all ranks, and all answer citations with a fixed query count."""
        run = cast(
            RetrievalRun | None,
            await self._session.scalar(
                select(RetrievalRun)
                .options(selectinload(RetrievalRun.results))
                .where(RetrievalRun.id == retrieval_run_id)
            ),
        )
        if run is None:
            return None
        results = sorted(run.results, key=_result_sort_key)
        citations = list(
            await self._session.scalars(
                select(AnswerCitation)
                .where(AnswerCitation.message_id == run.message_id)
                .order_by(AnswerCitation.citation_number, AnswerCitation.id)
            )
        )
        return CompleteRetrievalRun(run=run, results=results, citations=citations)


def _result_sort_key(result: RetrievalResult) -> tuple[int, int, int, str]:
    missing = 2**31 - 1
    return (
        result.rerank_rank if result.rerank_rank is not None else missing,
        result.fused_rank if result.fused_rank is not None else missing,
        result.keyword_rank or result.vector_rank or missing,
        str(result.id),
    )


def _utc(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return timestamp.astimezone(UTC)
