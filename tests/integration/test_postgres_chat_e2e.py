from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import delete, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.db.models.conversation import Conversation, Message
from app.db.models.retrieval import AnswerCitation, RetrievalResult, RetrievalRun
from app.db.models.source_document import Chunk, Document, Source
from app.db.models.usage import UsageRecord
from app.domain.chat import ChatResult
from app.domain.evaluation import Difficulty, EvaluationCase
from app.domain.retrieval import QueryFilters
from app.providers.base import (
    EmbeddingProvider,
    EmbeddingResponse,
    GenerationResponse,
    LLMProvider,
    RerankerProvider,
    RerankResponse,
    TokenUsage,
)
from app.providers.testing import DeterministicEmbeddingProvider, DeterministicRerankerProvider
from app.repositories.chunk_repository import ChunkRepository, PendingEmbeddingChunk
from app.services.chat_service import ChatService
from app.services.indexing_service import IndexingService
from evaluation.adapters import response_from_chat
from evaluation.models import EvaluationResponse
from evaluation.reporting import write_report
from evaluation.runner import EvaluationRunner

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
EMBEDDING_DIMENSION = 1024

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is required for PostgreSQL chat E2E tests",
    ),
]


class _FixtureEmbeddingProvider(EmbeddingProvider):
    """Give deterministic vectors a test-unique model name and real usage counters."""

    def __init__(self, marker: str) -> None:
        self._delegate = DeterministicEmbeddingProvider(EMBEDDING_DIMENSION)
        self._model = f"e2e-embedding-{marker}"

    @property
    def name(self) -> str:
        return "e2e-deterministic"

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    async def healthcheck(self) -> None:
        return None

    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        response = await self._delegate.embed(texts)
        return EmbeddingResponse(
            vectors=response.vectors,
            model=self.model,
            dimension=self.dimension,
            usage=TokenUsage(prompt_tokens=sum(max(1, len(item.split())) for item in texts)),
        )


class _FixtureRerankerProvider(RerankerProvider):
    """Run the deterministic overlap scorer under a test-unique model name."""

    def __init__(self, marker: str) -> None:
        self._delegate = DeterministicRerankerProvider()
        self._model = f"e2e-reranker-{marker}"

    @property
    def name(self) -> str:
        return "e2e-deterministic"

    @property
    def model(self) -> str:
        return self._model

    async def healthcheck(self) -> None:
        return None

    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse:
        response = await self._delegate.rerank(query, documents)
        return RerankResponse(scores=response.scores, model=self.model)


class _GroundedLLM(LLMProvider):
    """Return one evidence-backed answer and strict JSON for gray-zone judging."""

    def __init__(self, marker: str) -> None:
        self._model = f"e2e-llm-{marker}"

    @property
    def name(self) -> str:
        return "e2e-deterministic"

    @property
    def model(self) -> str:
        return self._model

    async def healthcheck(self) -> None:
        return None

    async def generate(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        del max_tokens, temperature
        prompt_tokens = sum(max(1, len(item["content"].split())) for item in messages)
        if response_format is not None:
            content = json.dumps(
                {
                    "sufficient": True,
                    "score": 0.98,
                    "reason": "The fixture directly supplies the rollback command.",
                }
            )
        else:
            content = (
                "Use kubectl rollout undo deployment/nginx to roll back the Deployment nginx "
                "to its previous revision. [1]"
            )
        return GenerationResponse(
            text=content,
            model=self.model,
            usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=len(content.split())),
            finish_reason="stop",
        )

    def stream(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        del messages, max_tokens, temperature
        return _one_delta("unused")


class _ForgedCitationLLM(_GroundedLLM):
    """Emit one supported marker and one marker absent from supplied context."""

    def __init__(self, marker: str) -> None:
        super().__init__(marker)
        self._model = f"e2e-forged-citation-llm-{marker}"

    async def generate(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        if response_format is not None:
            return await super().generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
        content = (
            "Use kubectl rollout undo deployment/nginx for the previous revision [1]. "
            "A source that was not provided also guarantees success [999]."
        )
        return GenerationResponse(
            text=content,
            model=self.model,
            usage=TokenUsage(
                prompt_tokens=sum(max(1, len(item["content"].split())) for item in messages),
                completion_tokens=len(content.split()),
            ),
            finish_reason="stop",
        )


async def _one_delta(value: str) -> AsyncIterator[str]:
    yield value


class _ScopedChunkRepository(ChunkRepository):
    """Restrict the indexing pass to the row created by this test."""

    def __init__(self, session: AsyncSession, chunk_id: UUID) -> None:
        super().__init__(session)
        self._fixture_session = session
        self._chunk_id = chunk_id

    async def list_needing_embedding(
        self,
        *,
        model: str,
        dimension: int,
        limit: int,
    ) -> list[PendingEmbeddingChunk]:
        row = (
            await self._fixture_session.execute(
                select(Chunk.id, Chunk.content_hash, Chunk.contextualized_content).where(
                    Chunk.id == self._chunk_id,
                    Chunk.deleted_at.is_(None),
                    or_(
                        Chunk.embedding.is_(None),
                        Chunk.embedding_model.is_distinct_from(model),
                        Chunk.embedding_dimension.is_distinct_from(dimension),
                    ),
                )
            )
        ).one_or_none()
        if row is None or limit < 1:
            return []
        return [
            PendingEmbeddingChunk(
                id=row.id,
                content_hash=row.content_hash,
                contextualized_content=row.contextualized_content,
            )
        ]


class _SourceFilteredExecutor:
    """Exercise EvaluationRunner through the real ChatService while isolating retrieval."""

    def __init__(
        self,
        service: ChatService,
        source_id: UUID,
        completed_results: list[ChatResult],
    ) -> None:
        self._service = service
        self._filters = QueryFilters(source_ids=[source_id])
        self._completed_results = completed_results

    async def execute(self, case: EvaluationCase) -> EvaluationResponse:
        result = await self._service.complete(
            case.question,
            filters=self._filters,
            history_override=case.conversation_history,
        )
        self._completed_results.append(result)
        return response_from_chat(result)


def _settings(
    database_url: str,
    embedding: _FixtureEmbeddingProvider,
    reranker: _FixtureRerankerProvider,
) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        database_url=database_url,
        embedding_provider=embedding.name,
        embedding_model=embedding.model,
        embedding_dimension=embedding.dimension,
        embedding_batch_size=4,
        reranker_provider=reranker.name,
        reranker_model=reranker.model,
        keyword_top_k=5,
        vector_top_k=5,
        rerank_top_k=3,
        enable_query_rewrite=False,
        enable_reranker=True,
        enable_citation_validation=False,
        no_answer_topic_overlap_threshold=0.2,
        no_answer_gray_zone_lower=0.1,
        no_answer_gray_zone_upper=0.5,
        evidence_sufficiency_threshold=0.6,
        citation_coverage_threshold=0.5,
        max_context_tokens=1200,
    )


async def _assert_migrated(session_factory: async_sessionmaker[AsyncSession]) -> None:
    async with session_factory() as session:
        revision = await session.scalar(text("SELECT version_num FROM alembic_version"))
        extensions = set(
            await session.scalars(
                text("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')")
            )
        )
        chunks_table = await session.scalar(text("SELECT to_regclass('chunks')::text"))
    assert revision == _alembic_head(), (
        "TEST_DATABASE_URL must be upgraded with alembic upgrade head"
    )
    assert chunks_table == "chunks"
    assert extensions == {"vector", "pg_trgm"}


def _alembic_head() -> str:
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    heads = ScriptDirectory.from_config(config).get_heads()
    assert len(heads) == 1, f"expected one Alembic head, found {heads}"
    return heads[0]


async def _seed_knowledge_base(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    marker: str,
) -> tuple[UUID, UUID]:
    source_id = uuid4()
    document_id = uuid4()
    chunk_id = uuid4()
    content = (
        f"Integration marker {marker}. Deployment rollback: use kubectl rollout undo "
        "deployment/nginx to roll back the Deployment nginx to its previous revision."
    )
    async with session_factory.begin() as session:
        session.add_all(
            [
                Source(
                    id=source_id,
                    name=f"e2e-source-{marker}",
                    source_type="github_repository",
                    enabled=True,
                ),
                Document(
                    id=document_id,
                    source_id=source_id,
                    external_id=f"fixtures/{marker}/deployment.md",
                    document_type="official_documentation",
                    title=f"Deployment rollback {marker}",
                    canonical_url=(
                        "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/"
                    ),
                    source_version="v1.34",
                    content_hash=marker.ljust(64, "0")[:64],
                    metadata_={"kind": "Deployment", "version": "v1.34"},
                    status="active",
                ),
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    chunk_index=0,
                    document_title=f"Deployment rollback {marker}",
                    heading_path=["Deployments", "Rolling Back"],
                    content=content,
                    contextualized_content=f"Deployment rollback Rolling Back {content}",
                    token_count=len(content.split()),
                    content_hash=marker[::-1].ljust(64, "f")[:64],
                    start_offset=0,
                    end_offset=len(content),
                    start_line=1,
                    end_line=2,
                    metadata_={"kind": "Deployment", "version": "v1.34"},
                ),
            ]
        )
    return source_id, chunk_id


async def _cleanup(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    source_id: UUID,
    marker: str,
    model_names: Sequence[str],
) -> None:
    async with session_factory.begin() as session:
        await session.execute(delete(UsageRecord).where(UsageRecord.model.in_(model_names)))
        await session.execute(delete(Conversation).where(Conversation.title.contains(marker)))
        await session.execute(delete(Source).where(Source.id == source_id))


@pytest.mark.asyncio
async def test_postgres_index_chat_refusal_persistence_and_evaluation_report(
    tmp_path: Path,
) -> None:
    """Run the paid-provider-free vertical path against a migrated PostgreSQL database."""
    assert TEST_DATABASE_URL is not None
    marker = f"e2e{uuid4().hex}"
    engine = create_async_engine(TEST_DATABASE_URL)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    embedding = _FixtureEmbeddingProvider(marker)
    reranker = _FixtureRerankerProvider(marker)
    llm = _GroundedLLM(marker)
    forged_llm = _ForgedCitationLLM(marker)
    source_id: UUID | None = None
    completed_results: list[ChatResult] = []

    try:
        await _assert_migrated(session_factory)
        source_id, chunk_id = await _seed_knowledge_base(session_factory, marker=marker)

        def repository_factory(session: AsyncSession) -> ChunkRepository:
            return _ScopedChunkRepository(session, chunk_id)

        indexing = IndexingService(
            session_factory=session_factory,
            provider=embedding,
            dimension=embedding.dimension,
            batch_size=4,
            repository_factory=repository_factory,
        )
        indexing_stats = await indexing.run_once(limit=1)
        assert indexing_stats.selected == 1
        assert indexing_stats.indexed == 1
        assert indexing_stats.stale == 0

        async with session_factory() as session:
            indexed_chunk = await session.get(Chunk, chunk_id)
            assert indexed_chunk is not None
            assert indexed_chunk.embedding is not None
            assert len(indexed_chunk.embedding) == EMBEDDING_DIMENSION
            assert indexed_chunk.embedding_model == embedding.model
            assert indexed_chunk.embedding_dimension == EMBEDDING_DIMENSION
            assert indexed_chunk.search_vector is not None
            indexing_usage = list(
                await session.scalars(
                    select(UsageRecord).where(UsageRecord.request_id == indexing_stats.request_id)
                )
            )
            assert len(indexing_usage) == 1
            assert indexing_usage[0].operation == "indexing_embedding"
            assert indexing_usage[0].input_text_count == 1
            assert indexing_usage[0].input_character_count > 0
            assert indexing_usage[0].prompt_tokens > 0
            assert indexing_usage[0].estimated_cost is None
            assert indexing_usage[0].latency_ms >= 0

        settings = _settings(TEST_DATABASE_URL, embedding, reranker)
        chat = ChatService(
            session_factory=session_factory,
            settings=settings,
            embedding_provider=embedding,
            reranker_provider=reranker,
            llm_provider=llm,
        )
        filters = QueryFilters(source_ids=[source_id])
        answer_query = (
            f"kubectl rollout undo deployment/nginx rollback Deployment previous revision {marker}"
        )
        outside_query = f"{marker} quantum banana astrophysics weather stock price chemistry"

        answer_result = await chat.complete(answer_query, filters=filters)
        completed_results.append(answer_result)
        assert answer_result.answerable is True
        assert answer_result.citations
        assert answer_result.citations[0].chunk_id == chunk_id
        assert answer_result.citations[0].valid is True
        assert answer_result.usage.total_tokens > 0
        assert answer_result.latency.total_ms > 0
        assert answer_result.retrieval.keyword_count == 1
        assert answer_result.retrieval.vector_count == 1
        assert answer_result.retrieval.reranked_count == 1

        refusal_result = await chat.complete(outside_query, filters=filters)
        completed_results.append(refusal_result)
        assert refusal_result.answerable is False
        assert refusal_result.no_answer.reason == "topic_mismatch"
        assert refusal_result.citations == []
        assert "does not contain enough" in refusal_result.answer

        forged_chat = ChatService(
            session_factory=session_factory,
            settings=settings,
            embedding_provider=embedding,
            reranker_provider=reranker,
            llm_provider=forged_llm,
        )
        forged_result = await forged_chat.complete(answer_query, filters=filters)
        completed_results.append(forged_result)
        assert forged_result.answerable is False
        assert forged_result.no_answer.reason == "citation_validation_failed"
        assert forged_result.citations == []
        assert "[999]" not in forged_result.answer
        assert forged_result.citation_report is not None
        assert forged_result.citation_report.invalid_citation_numbers == [999]

        async with session_factory() as session:
            assistant = await session.get(Message, answer_result.message_id)
            assert assistant is not None
            assert assistant.token_usage["total_tokens"] == answer_result.usage.total_tokens
            assert assistant.latency_ms is not None and assistant.latency_ms >= 0

            answer_run = await session.scalar(
                select(RetrievalRun).where(RetrievalRun.message_id == answer_result.message_id)
            )
            assert answer_run is not None
            assert answer_run.metadata_["answerable"] is True
            assert answer_run.keyword_latency_ms >= 0
            assert answer_run.vector_latency_ms >= 0
            assert answer_run.rerank_latency_ms >= 0
            assert answer_run.total_latency_ms >= 0

            traced = list(
                await session.scalars(
                    select(RetrievalResult).where(RetrievalResult.retrieval_run_id == answer_run.id)
                )
            )
            assert len(traced) == 1
            assert traced[0].chunk_id == chunk_id
            assert traced[0].selected_for_context is True

            citation = await session.scalar(
                select(AnswerCitation).where(AnswerCitation.message_id == answer_result.message_id)
            )
            assert citation is not None
            assert citation.chunk_id == chunk_id
            assert citation.citation_valid is True
            assert citation.quoted_text

            usage = list(
                await session.scalars(
                    select(UsageRecord).where(UsageRecord.request_id == answer_result.request_id)
                )
            )
            operations = {item.operation for item in usage}
            assert {"query_embedding", "rerank", "answer_generation"} <= operations
            assert sum(item.total_tokens for item in usage) > 0
            assert all(item.latency_ms >= 0 for item in usage)

            refusal_citations = list(
                await session.scalars(
                    select(AnswerCitation).where(
                        AnswerCitation.message_id == refusal_result.message_id
                    )
                )
            )
            assert refusal_citations == []

        cases = [
            EvaluationCase(
                id=f"{marker}-answerable",
                question=answer_query,
                reference_answer=(
                    "Use kubectl rollout undo deployment/nginx to roll back the Deployment "
                    "nginx to its previous revision."
                ),
                relevant_chunk_ids=[str(chunk_id)],
                expected_citations=[str(chunk_id)],
                answerable=True,
                category="how_to",
                difficulty=Difficulty.EASY,
                source_type="official_documentation",
                human_reviewed=True,
            ),
            EvaluationCase(
                id=f"{marker}-unanswerable",
                question=outside_query,
                reference_answer="Outside the fixture knowledge base.",
                answerable=False,
                category="out_of_scope",
                difficulty=Difficulty.EASY,
                human_reviewed=True,
            ),
        ]
        runner = EvaluationRunner(
            _SourceFilteredExecutor(chat, source_id, completed_results),
            concurrency=1,
        )
        report = await runner.run(
            cases,
            experiment_name="postgres-chat-e2e",
            dataset_name="postgres-fixture",
            dataset_path=tmp_path / "fixture.jsonl",
            config_snapshot={
                "retrieval_mode": "hybrid",
                "embedding_model": embedding.model,
                "reranker_model": reranker.model,
                "llm_model": llm.model,
            },
        )
        assert report.dataset_size == 2
        assert report.summary["recall_at_1"] == 0.5
        assert report.summary["no_answer_accuracy"] == 1.0
        assert report.summary["total_tokens"] > 0
        assert report.results[0].metrics["citation_precision"] == 1.0
        assert report.results[0].metrics["expected_citation_recall"] == 1.0
        assert report.results[1].predicted_answerable is False
        assert len(completed_results) == 5

        report_paths = write_report(report, tmp_path / "actual-report")
        serialized = json.loads(report_paths["json"].read_text(encoding="utf-8"))
        assert serialized["run_id"] == report.run_id
        assert serialized["summary"]["no_answer_f1"] == 1.0
        assert "postgres-chat-e2e" in report_paths["markdown"].read_text(encoding="utf-8")
        assert len(report_paths["jsonl"].read_text(encoding="utf-8").splitlines()) == 2
        assert report_paths["csv"].is_file()
    finally:
        if source_id is not None:
            await _cleanup(
                session_factory,
                source_id=source_id,
                marker=marker,
                model_names=[embedding.model, reranker.model, llm.model, forged_llm.model],
            )
        await engine.dispose()
