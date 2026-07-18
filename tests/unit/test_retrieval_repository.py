from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import RetrievalError
from app.domain.retrieval import QueryFilters, RetrievalCandidate
from app.repositories.retrieval_repository import (
    RetrievalRepository,
    build_keyword_statement,
    build_vector_statement,
)
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.vector_retriever import VectorRetriever


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> _FakeResult:
        return self

    def all(self) -> list[dict[str, Any]]:
        return self._rows


class _RecordingSession:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        error: SQLAlchemyError | None = None,
    ) -> None:
        self.rows = rows
        self.error = error
        self.statement: Any = None
        self.parameters: dict[str, Any] = {}

    async def execute(self, statement: Any, parameters: dict[str, Any]) -> _FakeResult:
        self.statement = statement
        self.parameters = parameters
        if self.error is not None:
            raise self.error
        return _FakeResult(self.rows)


def _row(*, keyword_score: float = 0.8, vector_score: float = 0.9) -> dict[str, Any]:
    return {
        "chunk_id": uuid4(),
        "document_id": uuid4(),
        "document_title": "Deployments",
        "document_type": "official_documentation",
        "heading_path": ["Workloads", "Deployments"],
        "content": "A Deployment supports a rolling update.",
        "canonical_url": "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
        "document_metadata": {"kind": "Deployment", "version": "v1.34"},
        "chunk_metadata": {"api_group": "apps"},
        "source_id": uuid4(),
        "source_name": "kubernetes-docs",
        "source_type": "github_repository",
        "start_offset": 10,
        "end_offset": 52,
        "keyword_score": keyword_score,
        "vector_score": vector_score,
    }


def test_keyword_statement_compiles_to_weighted_parameterized_active_search() -> None:
    malicious = "x') OR true; DROP TABLE chunks; --"
    filters = QueryFilters(
        source_ids=[uuid4()],
        document_types=["official_documentation"],
        versions=["v1.34"],
        api_groups=["apps"],
        kinds=["Deployment"],
        issue_states=["open"],
        release_versions=["v1.34.0"],
        metadata={
            "api_versions": ["v1"],
            "tenant'; DROP TABLE sources; --": malicious,
        },
    )

    compiled = build_keyword_statement(filters=filters, limit=30).compile(
        dialect=postgresql.dialect()
    )
    sql = str(compiled)

    assert "websearch_to_tsquery" in sql
    assert "plainto_tsquery" in sql
    assert "ts_rank_cd" in sql
    assert "similarity(chunks.content" in sql
    assert "chunks.search_vector @@" in sql
    assert "chunks.deleted_at IS NULL" in sql
    assert "documents.deleted_at IS NULL" in sql
    assert "documents.status =" in sql
    assert "sources.enabled IS true" in sql
    assert "documents.source_id IN" in sql
    assert "documents.metadata @>" in sql
    assert malicious not in sql
    assert any(malicious in str(value) for value in compiled.params.values())
    assert any(value == {"version": "v1"} for value in compiled.params.values())


def test_vector_statement_compiles_to_filtered_hnsw_friendly_cosine_query() -> None:
    filters = QueryFilters(kinds=["Deployment"], api_groups=["apps"])

    compiled = build_vector_statement(
        filters=filters,
        limit=12,
        embedding_dimension=1024,
        embedding_model="BAAI/bge-m3",
    ).compile(dialect=postgresql.dialect())
    sql = str(compiled)

    assert "chunks.embedding <=>" in sql
    assert "ORDER BY (chunks.embedding <=>" in sql
    assert "chunks.embedding IS NOT NULL" in sql
    assert "chunks.embedding_dimension =" in sql
    assert "chunks.embedding_model =" in sql
    assert "documents.metadata @>" in sql
    assert "query_embedding" in compiled.params
    assert "BAAI/bge-m3" in compiled.params.values()


@pytest.mark.asyncio
async def test_keyword_search_executes_once_and_maps_complete_candidate() -> None:
    session = _RecordingSession([_row(keyword_score=0.73)])
    repository = RetrievalRepository(cast(AsyncSession, session))
    malicious_query = "Deployment'; DROP TABLE chunks; --"

    candidates = await repository.keyword_search(
        malicious_query,
        filters=QueryFilters(kinds=["Deployment"]),
        limit=7,
    )

    assert session.parameters == {"search_query": malicious_query}
    compiled_sql = str(session.statement.compile(dialect=postgresql.dialect()))
    assert malicious_query not in compiled_sql
    assert len(candidates) == 1
    assert candidates[0].keyword_rank == 1
    assert candidates[0].keyword_score == pytest.approx(0.73)
    assert candidates[0].heading_path == ["Workloads", "Deployments"]
    assert candidates[0].metadata["api_group"] == "apps"
    assert candidates[0].metadata["source_name"] == "kubernetes-docs"


@pytest.mark.asyncio
async def test_vector_search_executes_once_and_maps_rank_and_similarity() -> None:
    session = _RecordingSession([_row(vector_score=0.98), _row(vector_score=0.64)])
    repository = RetrievalRepository(cast(AsyncSession, session))

    candidates = await repository.vector_search(
        [0.2, 0.4, 0.6],
        filters=QueryFilters(document_types=["official_documentation"]),
        limit=2,
        embedding_model="fixture-model",
    )

    assert session.parameters == {"query_embedding": [0.2, 0.4, 0.6]}
    assert [candidate.vector_rank for candidate in candidates] == [1, 2]
    assert [candidate.vector_score for candidate in candidates] == pytest.approx([0.98, 0.64])


@pytest.mark.asyncio
async def test_repository_validates_inputs_before_database_access() -> None:
    session = _RecordingSession([])
    repository = RetrievalRepository(cast(AsyncSession, session))

    with pytest.raises(ValueError, match="blank"):
        await repository.keyword_search("   ")
    with pytest.raises(ValueError, match="between"):
        await repository.keyword_search("deployment", limit=201)
    with pytest.raises(ValueError, match="empty"):
        await repository.vector_search([])
    with pytest.raises(ValueError, match="finite"):
        await repository.vector_search([float("nan")])

    assert session.statement is None


@pytest.mark.asyncio
async def test_repository_wraps_database_errors_without_query_details() -> None:
    session = _RecordingSession([], error=SQLAlchemyError("database included secret query"))
    repository = RetrievalRepository(cast(AsyncSession, session))

    with pytest.raises(RetrievalError, match="keyword retrieval failed") as captured:
        await repository.keyword_search("sensitive user query")

    assert "sensitive user query" not in str(captured.value)


class _KeywordRepositoryStub:
    def __init__(self) -> None:
        self.call: tuple[str, QueryFilters | None, int] | None = None

    async def keyword_search(
        self,
        query: str,
        *,
        filters: QueryFilters | None = None,
        limit: int = 30,
    ) -> list[RetrievalCandidate]:
        self.call = (query, filters, limit)
        return []


class _VectorRepositoryStub:
    def __init__(self) -> None:
        self.call: tuple[Sequence[float], QueryFilters | None, int, str | None] | None = None

    async def vector_search(
        self,
        embedding: Sequence[float],
        *,
        filters: QueryFilters | None = None,
        limit: int = 30,
        embedding_model: str | None = None,
    ) -> list[RetrievalCandidate]:
        self.call = (embedding, filters, limit, embedding_model)
        return []


@pytest.mark.asyncio
async def test_retriever_adapters_apply_defaults_and_dimension_validation() -> None:
    keyword_repository = _KeywordRepositoryStub()
    keyword = KeywordRetriever(keyword_repository, default_top_k=17, max_query_length=20)
    filters = QueryFilters(kinds=["Pod"])

    await keyword.retrieve("  Pod status  ", filters=filters)

    assert keyword_repository.call == ("Pod status", filters, 17)
    with pytest.raises(ValueError, match="maximum length"):
        await keyword.retrieve("x" * 21)

    vector_repository = _VectorRepositoryStub()
    vector = VectorRetriever(
        vector_repository,
        embedding_dimension=3,
        embedding_model="fixture-model",
        default_top_k=11,
    )

    await vector.retrieve([0.1, 0.2, 0.3], filters=filters)

    assert vector_repository.call == ([0.1, 0.2, 0.3], filters, 11, "fixture-model")
    with pytest.raises(ValueError, match="dimension mismatch"):
        await vector.retrieve([0.1, 0.2])


def test_statement_builders_reject_unbounded_limits_and_dimensions() -> None:
    with pytest.raises(ValueError, match="between"):
        build_keyword_statement(filters=QueryFilters(), limit=0)
    with pytest.raises(ValueError, match="between"):
        build_vector_statement(
            filters=QueryFilters(),
            limit=999,
            embedding_dimension=1024,
        )
    with pytest.raises(ValueError, match="positive"):
        build_vector_statement(filters=QueryFilters(), limit=10, embedding_dimension=0)
