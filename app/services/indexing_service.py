"""Batch embedding backfill that never holds a transaction during inference."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import ProviderError
from app.domain.usage import EmbeddingBatchUsage
from app.providers.base import EmbeddingProvider
from app.repositories.chunk_repository import (
    ChunkEmbeddingUpdate,
    ChunkRepository,
    PendingEmbeddingChunk,
)
from app.repositories.usage_repository import UsageRepository
from app.services.pricing_service import PricingCatalog
from app.services.usage_service import UsageService


@dataclass(frozen=True, slots=True)
class IndexingStats:
    """Counts for one bounded embedding backfill pass."""

    selected: int = 0
    indexed: int = 0
    stale: int = 0
    request_id: UUID = field(default_factory=uuid4)


RepositoryFactory = Callable[[AsyncSession], ChunkRepository]
UsageRepositoryFactory = Callable[[AsyncSession], UsageRepository]


class IndexingService:
    """Backfill missing or model-stale vectors with optimistic database writes."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        provider: EmbeddingProvider,
        dimension: int,
        batch_size: int,
        repository_factory: RepositoryFactory = ChunkRepository,
        usage_repository_factory: UsageRepositoryFactory = UsageRepository,
        pricing_catalog: PricingCatalog | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._provider = provider
        self._dimension = dimension
        self._batch_size = batch_size
        self._repository_factory = repository_factory
        self._usage_repository_factory = usage_repository_factory
        self._pricing = pricing_catalog or PricingCatalog()

    async def run_once(
        self,
        *,
        limit: int = 1000,
        request_id: UUID | None = None,
    ) -> IndexingStats:
        """Index at most ``limit`` active chunks and report concurrent stale writes."""
        if limit < 1:
            raise ValueError("limit must be positive")
        if self._provider.dimension != self._dimension:
            raise ProviderError("Embedding provider and database dimensions do not match")
        resolved_request_id = request_id or uuid4()
        async with self._session_factory() as session:
            pending = await self._repository_factory(session).list_needing_embedding(
                model=self._provider.model,
                dimension=self._dimension,
                limit=limit,
            )
        if not pending:
            return IndexingStats(request_id=resolved_request_id)
        updates, usage_batches = await self._embed(pending)
        written_at = datetime.now(UTC)
        async with self._session_factory() as session, session.begin():
            indexed = await self._repository_factory(session).update_embeddings(
                updates,
                updated_at=written_at,
            )
            await UsageService(
                self._usage_repository_factory(session),
                self._pricing,
            ).record_embedding_batches(
                request_id=resolved_request_id,
                operation="indexing_embedding",
                batches=usage_batches,
                created_at=written_at,
            )
        return IndexingStats(
            selected=len(pending),
            indexed=indexed,
            stale=len(pending) - indexed,
            request_id=resolved_request_id,
        )

    async def _embed(
        self,
        pending: Sequence[PendingEmbeddingChunk],
    ) -> tuple[list[ChunkEmbeddingUpdate], list[EmbeddingBatchUsage]]:
        updates: list[ChunkEmbeddingUpdate] = []
        usage_batches: list[EmbeddingBatchUsage] = []
        for start in range(0, len(pending), self._batch_size):
            batch = pending[start : start + self._batch_size]
            texts = [chunk.contextualized_content for chunk in batch]
            started = time.perf_counter()
            response = await self._provider.embed(texts)
            latency_ms = (time.perf_counter() - started) * 1000
            if response.dimension != self._dimension or len(response.vectors) != len(batch):
                raise ProviderError("Embedding provider returned an incompatible batch")
            for chunk, vector in zip(batch, response.vectors, strict=True):
                if len(vector) != self._dimension:
                    raise ProviderError("Embedding provider returned an unexpected dimension")
                updates.append(
                    ChunkEmbeddingUpdate(
                        id=chunk.id,
                        content_hash=chunk.content_hash,
                        embedding=list(vector),
                        model=response.model,
                        dimension=response.dimension,
                    )
                )
            usage_batches.append(
                EmbeddingBatchUsage(
                    model=response.model,
                    provider=self._provider.name,
                    input_text_count=len(texts),
                    input_character_count=sum(len(text) for text in texts),
                    prompt_tokens=response.usage.prompt_tokens,
                    latency_ms=latency_ms,
                )
            )
        return updates, usage_batches
