"""Reranking preserves provenance and degrades to fused order."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import pytest

from app.core.exceptions import ProviderError
from app.domain.retrieval import RetrievalCandidate
from app.providers.base import RerankerProvider, RerankResponse
from app.retrieval.reranker import rerank_candidates


def _candidate(rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_title=f"Doc {rank}",
        document_type="official_documentation",
        content=f"content {rank}",
        fused_rank=rank,
        fused_score=1 / rank,
    )


class ScoringReranker(RerankerProvider):
    name = "scoring"

    async def healthcheck(self) -> None:
        return None

    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse:
        return RerankResponse(scores=[0.1, 0.9, 0.5], model="test")


class FailingReranker(RerankerProvider):
    name = "failing"

    async def healthcheck(self) -> None:
        raise ProviderError("unavailable")

    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse:
        raise ProviderError("unavailable")


@pytest.mark.asyncio
async def test_reranker_sorts_scores_and_limits_top_n() -> None:
    candidates = [_candidate(1), _candidate(2), _candidate(3)]

    outcome = await rerank_candidates(
        query="query",
        candidates=candidates,
        provider=ScoringReranker(),
        top_n=2,
        timeout_seconds=1,
    )

    assert [item.document_title for item in outcome.candidates] == ["Doc 2", "Doc 3"]
    assert [item.rerank_rank for item in outcome.candidates] == [1, 2]
    assert outcome.degraded is False


@pytest.mark.asyncio
async def test_reranker_failure_degrades_to_fused_order() -> None:
    candidates = [_candidate(2), _candidate(1), _candidate(3)]

    outcome = await rerank_candidates(
        query="query",
        candidates=candidates,
        provider=FailingReranker(),
        top_n=2,
        timeout_seconds=1,
    )

    assert [item.fused_rank for item in outcome.candidates] == [1, 2]
    assert all(item.rerank_rank is None for item in outcome.candidates)
    assert all(item.rerank_score is None for item in outcome.candidates)
    assert outcome.degraded is True
    assert outcome.reason == "ProviderError"


@pytest.mark.asyncio
async def test_disabled_reranker_preserves_only_fusion_provenance() -> None:
    outcome = await rerank_candidates(
        query="query",
        candidates=[_candidate(2), _candidate(1)],
        provider=ScoringReranker(),
        top_n=2,
        timeout_seconds=1,
        enabled=False,
    )

    assert [item.fused_rank for item in outcome.candidates] == [1, 2]
    assert all(item.rerank_rank is None for item in outcome.candidates)
    assert all(item.rerank_score is None for item in outcome.candidates)
    assert outcome.degraded is False
    assert outcome.reason == "disabled"


@pytest.mark.asyncio
async def test_reranker_score_threshold_filters_only_successfully_scored_results() -> None:
    outcome = await rerank_candidates(
        query="query",
        candidates=[_candidate(1), _candidate(2), _candidate(3)],
        provider=ScoringReranker(),
        top_n=3,
        timeout_seconds=1,
        score_threshold=0.5,
    )

    assert [item.document_title for item in outcome.candidates] == ["Doc 2", "Doc 3"]
    assert len(outcome.all_candidates or []) == 3
