"""Hybrid retrieval orchestration covers success and channel degradation."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

import pytest

from app.core.exceptions import RetrievalError
from app.domain.retrieval import (
    FusionMode,
    QueryFilters,
    RetrievalCandidate,
    RetrievalMode,
)
from app.providers.base import (
    EmbeddingProvider,
    EmbeddingResponse,
    RerankerProvider,
    RerankResponse,
)
from app.retrieval.hybrid_fusion import HybridFusion
from app.retrieval.query_preprocessor import QueryPreprocessor
from app.services.retrieval_service import RetrievalService


def _candidate(
    number: int,
    *,
    keyword_rank: int | None = None,
    vector_rank: int | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=UUID(int=number),
        document_id=UUID(int=100 + number),
        document_title=f"Document {number}",
        document_type="official_documentation",
        content=f"evidence {number}",
        keyword_rank=keyword_rank,
        keyword_score=1 / keyword_rank if keyword_rank else None,
        vector_rank=vector_rank,
        vector_score=1 / vector_rank if vector_rank else None,
    )


class FakeKeywordRetriever:
    def __init__(self, candidates: list[RetrievalCandidate], *, fail: bool = False) -> None:
        self.candidates = candidates
        self.fail = fail
        self.calls = 0

    async def retrieve(
        self,
        query: str,
        *,
        filters: QueryFilters | None = None,
        limit: int | None = None,
    ) -> list[RetrievalCandidate]:
        self.calls += 1
        if self.fail:
            raise RetrievalError("keyword unavailable")
        return self.candidates[:limit]


class FakeVectorRetriever:
    def __init__(self, candidates: list[RetrievalCandidate], *, fail: bool = False) -> None:
        self.candidates = candidates
        self.fail = fail
        self.calls = 0

    async def retrieve(
        self,
        embedding: Sequence[float],
        *,
        filters: QueryFilters | None = None,
        limit: int | None = None,
    ) -> list[RetrievalCandidate]:
        self.calls += 1
        assert embedding == [1.0, 0.0]
        if self.fail:
            raise RetrievalError("vector unavailable")
        return self.candidates[:limit]


class FakeEmbedding(EmbeddingProvider):
    name = "fake"
    model = "fake-embedding"
    dimension = 2

    async def healthcheck(self) -> None:
        return None

    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        return EmbeddingResponse([[1.0, 0.0]], self.model, self.dimension)


class FakeReranker(RerankerProvider):
    name = "fake"

    async def healthcheck(self) -> None:
        return None

    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse:
        return RerankResponse(
            scores=[float(document.rsplit(maxsplit=1)[-1]) for document in documents],
            model="fake-reranker",
        )


def _service(
    keyword: FakeKeywordRetriever,
    vector: FakeVectorRetriever,
    *,
    embedding: EmbeddingProvider | None = None,
    reranker: RerankerProvider | None = None,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    enable_reranker: bool = True,
) -> RetrievalService:
    return RetrievalService(
        keyword_retriever=keyword,
        vector_retriever=vector,
        embedding_provider=embedding,
        reranker_provider=reranker,
        preprocessor=QueryPreprocessor(),
        fusion=HybridFusion(mode=FusionMode.RRF),
        retrieval_mode=mode,
        keyword_top_k=30,
        vector_top_k=30,
        rerank_top_k=2,
        embedding_timeout_seconds=1,
        reranker_timeout_seconds=1,
        enable_reranker=enable_reranker,
    )


@pytest.mark.asyncio
async def test_hybrid_retrieval_retains_channel_ranks_and_reranks() -> None:
    keyword = FakeKeywordRetriever([_candidate(1, keyword_rank=1)])
    vector = FakeVectorRetriever([_candidate(2, vector_rank=1), _candidate(1, vector_rank=2)])
    service = _service(
        keyword,
        vector,
        embedding=FakeEmbedding(),
        reranker=FakeReranker(),
    )

    outcome = await service.retrieve("Deployment 如何回滚?", top_k=2)

    assert [item.chunk_id.int for item in outcome.candidates] == [2, 1]
    by_id = {item.chunk_id.int: item for item in outcome.candidates}
    assert by_id[1].keyword_rank == 1
    assert by_id[1].vector_rank == 2
    assert outcome.keyword_count == 1
    assert outcome.vector_count == 2
    assert outcome.reranked_count == 2
    assert outcome.degraded_channels == []
    assert outcome.query.language == "mixed"


@pytest.mark.asyncio
async def test_hybrid_retrieval_degrades_when_embedding_provider_is_missing() -> None:
    keyword = FakeKeywordRetriever([_candidate(1, keyword_rank=1)])
    vector = FakeVectorRetriever([])
    service = _service(
        keyword,
        vector,
        embedding=None,
        reranker=None,
        enable_reranker=False,
    )

    outcome = await service.retrieve("Deployment rollback")

    assert outcome.degraded_channels == ["vector"]
    assert outcome.keyword_count == 1
    assert outcome.vector_count == 0
    assert vector.calls == 0
    assert outcome.reranker_degraded is False
    assert outcome.reranked_count == 0


@pytest.mark.asyncio
async def test_hybrid_retrieval_fails_only_when_every_channel_fails() -> None:
    service = _service(
        FakeKeywordRetriever([], fail=True),
        FakeVectorRetriever([], fail=True),
        embedding=FakeEmbedding(),
        reranker=FakeReranker(),
    )

    with pytest.raises(RetrievalError, match="All configured retrieval channels failed"):
        await service.retrieve("Deployment rollback")


@pytest.mark.asyncio
async def test_keyword_mode_does_not_call_embedding_or_vector_retrieval() -> None:
    keyword = FakeKeywordRetriever([_candidate(1, keyword_rank=1)])
    vector = FakeVectorRetriever([])
    service = _service(
        keyword,
        vector,
        embedding=None,
        reranker=None,
        mode=RetrievalMode.KEYWORD,
        enable_reranker=False,
    )

    outcome = await service.retrieve("Deployment", mode=RetrievalMode.KEYWORD)

    assert outcome.keyword_count == 1
    assert vector.calls == 0
