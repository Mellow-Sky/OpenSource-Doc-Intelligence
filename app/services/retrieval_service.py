"""Application orchestration for preprocessing, hybrid recall, fusion, and reranking."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import ProviderError, RetrievalError
from app.domain.retrieval import (
    FusionMode,
    QueryFilters,
    RetrievalCandidate,
    RetrievalMode,
    RetrievalOutcome,
    RetrievalQuery,
    RetrievalTimings,
)
from app.providers.base import EmbeddingProvider, RerankerProvider
from app.repositories.retrieval_repository import RetrievalRepository
from app.retrieval.hybrid_fusion import HybridFusion
from app.retrieval.keyword_retriever import KeywordRetriever, KeywordRetrieverProtocol
from app.retrieval.query_preprocessor import QueryPreprocessor
from app.retrieval.reranker import rerank_candidates
from app.retrieval.vector_retriever import VectorRetriever, VectorRetrieverProtocol


@dataclass(frozen=True, slots=True)
class _ChannelResult:
    candidates: list[RetrievalCandidate] = field(default_factory=list)
    latency_ms: float = 0.0
    error: RetrievalError | ProviderError | TimeoutError | None = None
    embedding_prompt_tokens: int = 0
    embedding_model: str | None = None


class RetrievalService:
    """Run independent recall channels concurrently and retain ranking provenance."""

    def __init__(
        self,
        *,
        keyword_retriever: KeywordRetrieverProtocol,
        vector_retriever: VectorRetrieverProtocol,
        embedding_provider: EmbeddingProvider | None,
        reranker_provider: RerankerProvider | None,
        preprocessor: QueryPreprocessor,
        fusion: HybridFusion,
        retrieval_mode: RetrievalMode,
        keyword_top_k: int,
        vector_top_k: int,
        rerank_top_k: int,
        embedding_timeout_seconds: float,
        reranker_timeout_seconds: float,
        enable_reranker: bool,
        reranker_score_threshold: float | None = None,
    ) -> None:
        self._keyword_retriever = keyword_retriever
        self._vector_retriever = vector_retriever
        self._embedding_provider = embedding_provider
        self._reranker_provider = reranker_provider
        self._preprocessor = preprocessor
        self._fusion = fusion
        self._retrieval_mode = retrieval_mode
        self._keyword_top_k = keyword_top_k
        self._vector_top_k = vector_top_k
        self._rerank_top_k = rerank_top_k
        self._embedding_timeout_seconds = embedding_timeout_seconds
        self._reranker_timeout_seconds = reranker_timeout_seconds
        self._enable_reranker = enable_reranker
        self._reranker_score_threshold = reranker_score_threshold

    async def retrieve(
        self,
        query: str,
        *,
        rewritten_query: str | None = None,
        filters: QueryFilters | None = None,
        mode: RetrievalMode | None = None,
        top_k: int | None = None,
    ) -> RetrievalOutcome:
        """Retrieve and rerank evidence for an original or independently rewritten query."""
        total_started = time.perf_counter()
        rewrite_started = time.perf_counter()
        analyzed = self._preprocessor.preprocess(rewritten_query or query, filters)
        rewrite_ms = (time.perf_counter() - rewrite_started) * 1000
        resolved_mode = mode or self._retrieval_mode
        final_top_k = top_k or self._rerank_top_k
        retrieval_query = RetrievalQuery(
            original=query,
            rewritten=analyzed.normalized,
            language=analyzed.language,
            filters=analyzed.filters,
            mode=resolved_mode,
            top_k=final_top_k,
        )

        keyword_result, vector_result = await self._recall(retrieval_query)
        degraded_channels = _validate_channel_results(
            resolved_mode,
            keyword_result,
            vector_result,
        )

        fusion_started = time.perf_counter()
        fusion_limit = max(self._keyword_top_k, self._vector_top_k)
        fused = self._fusion.fuse(
            keyword_result.candidates,
            vector_result.candidates,
            retrieval_mode=resolved_mode,
            top_k=fusion_limit,
        )
        fusion_ms = (time.perf_counter() - fusion_started) * 1000

        reranked = await rerank_candidates(
            query=analyzed.normalized,
            candidates=fused,
            provider=self._reranker_provider,
            top_n=final_top_k,
            timeout_seconds=self._reranker_timeout_seconds,
            enabled=self._enable_reranker,
            score_threshold=self._reranker_score_threshold,
        )
        actual_reranked_count = (
            len(reranked.candidates)
            if self._enable_reranker and not reranked.degraded and fused
            else 0
        )
        timings = RetrievalTimings(
            rewrite_ms=rewrite_ms,
            keyword_ms=keyword_result.latency_ms,
            vector_ms=vector_result.latency_ms,
            fusion_ms=fusion_ms,
            rerank_ms=reranked.latency_ms,
            total_ms=(time.perf_counter() - total_started) * 1000,
        )
        return RetrievalOutcome(
            query=retrieval_query,
            candidates=reranked.candidates,
            trace_candidates=reranked.all_candidates or reranked.candidates,
            keyword_count=len(keyword_result.candidates),
            vector_count=len(vector_result.candidates),
            reranked_count=actual_reranked_count,
            timings=timings,
            degraded_channels=degraded_channels,
            reranker_degraded=reranked.degraded,
            reranker_reason=reranked.reason,
            embedding_prompt_tokens=vector_result.embedding_prompt_tokens,
            embedding_model=vector_result.embedding_model,
        )

    async def _recall(
        self,
        query: RetrievalQuery,
    ) -> tuple[_ChannelResult, _ChannelResult]:
        if query.mode is RetrievalMode.HYBRID:
            return await asyncio.gather(
                self._keyword_recall(query),
                self._vector_recall(query),
            )
        if query.mode is RetrievalMode.KEYWORD:
            return await self._keyword_recall(query), _ChannelResult()
        return _ChannelResult(), await self._vector_recall(query)

    async def _keyword_recall(self, query: RetrievalQuery) -> _ChannelResult:
        started = time.perf_counter()
        try:
            candidates = await self._keyword_retriever.retrieve(
                query.rewritten,
                filters=query.filters,
                limit=self._keyword_top_k,
            )
            return _ChannelResult(
                candidates=candidates,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except RetrievalError as exc:
            return _ChannelResult(
                latency_ms=(time.perf_counter() - started) * 1000,
                error=exc,
            )

    async def _vector_recall(self, query: RetrievalQuery) -> _ChannelResult:
        started = time.perf_counter()
        try:
            if self._embedding_provider is None:
                raise ProviderError("Embedding provider is unavailable")
            async with asyncio.timeout(self._embedding_timeout_seconds):
                response = await self._embedding_provider.embed([query.rewritten])
            if (
                len(response.vectors) != 1
                or response.dimension != self._embedding_provider.dimension
                or len(response.vectors[0]) != self._embedding_provider.dimension
            ):
                raise ProviderError("Embedding provider returned an incompatible query vector")
            candidates = await self._vector_retriever.retrieve(
                response.vectors[0],
                filters=query.filters,
                limit=self._vector_top_k,
            )
            return _ChannelResult(
                candidates=candidates,
                latency_ms=(time.perf_counter() - started) * 1000,
                embedding_prompt_tokens=response.usage.prompt_tokens,
                embedding_model=response.model,
            )
        except (ProviderError, RetrievalError, TimeoutError) as exc:
            return _ChannelResult(
                latency_ms=(time.perf_counter() - started) * 1000,
                error=exc,
            )


def build_retrieval_service(
    session: AsyncSession,
    settings: Settings,
    *,
    vector_session: AsyncSession | None = None,
    embedding_provider: EmbeddingProvider | None,
    reranker_provider: RerankerProvider | None,
) -> RetrievalService:
    """Compose request-scoped database adapters with process-scoped model providers."""
    repository = RetrievalRepository(session)
    vector_repository = RetrievalRepository(vector_session or session)
    return RetrievalService(
        keyword_retriever=KeywordRetriever(
            repository,
            default_top_k=settings.keyword_top_k,
            max_query_length=settings.max_query_length,
        ),
        vector_retriever=VectorRetriever(
            vector_repository,
            embedding_dimension=settings.embedding_dimension,
            embedding_model=settings.embedding_model,
            default_top_k=settings.vector_top_k,
        ),
        embedding_provider=embedding_provider,
        reranker_provider=reranker_provider,
        preprocessor=QueryPreprocessor(max_length=settings.max_query_length),
        fusion=HybridFusion(
            mode=FusionMode(settings.fusion_mode),
            rrf_k=settings.rrf_k,
            keyword_weight=settings.keyword_fusion_weight,
            vector_weight=settings.vector_fusion_weight,
        ),
        retrieval_mode=RetrievalMode(settings.retrieval_mode),
        keyword_top_k=settings.keyword_top_k,
        vector_top_k=settings.vector_top_k,
        rerank_top_k=settings.rerank_top_k,
        embedding_timeout_seconds=settings.embedding_timeout_seconds,
        reranker_timeout_seconds=settings.reranker_timeout_seconds,
        enable_reranker=settings.enable_reranker,
        reranker_score_threshold=settings.reranker_score_threshold,
    )


def _validate_channel_results(
    mode: RetrievalMode,
    keyword: _ChannelResult,
    vector: _ChannelResult,
) -> list[str]:
    if mode is RetrievalMode.KEYWORD and keyword.error is not None:
        raise keyword.error
    if mode is RetrievalMode.VECTOR and vector.error is not None:
        raise vector.error
    if mode is RetrievalMode.HYBRID and keyword.error is not None and vector.error is not None:
        raise RetrievalError(
            "All configured retrieval channels failed",
            details={
                "keyword": type(keyword.error).__name__,
                "vector": type(vector.error).__name__,
            },
        )
    degraded: list[str] = []
    if keyword.error is not None:
        degraded.append("keyword")
    if vector.error is not None:
        degraded.append("vector")
    return degraded
