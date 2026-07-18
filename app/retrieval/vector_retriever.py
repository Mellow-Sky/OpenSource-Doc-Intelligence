"""Application-facing pgvector cosine retrieval adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.domain.retrieval import QueryFilters, RetrievalCandidate
from app.repositories.retrieval_repository import MAX_RETRIEVAL_LIMIT


@runtime_checkable
class VectorSearchRepository(Protocol):
    """Persistence capability required by the vector retriever."""

    async def vector_search(
        self,
        embedding: Sequence[float],
        *,
        filters: QueryFilters | None = None,
        limit: int = 30,
        embedding_model: str | None = None,
    ) -> list[RetrievalCandidate]:
        """Return ranked vector candidates."""
        ...


@runtime_checkable
class VectorRetrieverProtocol(Protocol):
    """Interface consumed by the hybrid retrieval service."""

    async def retrieve(
        self,
        embedding: Sequence[float],
        *,
        filters: QueryFilters | None = None,
        limit: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Retrieve vector candidates for one query embedding."""
        ...


class VectorRetriever:
    """Validate embedding shape and delegate one HNSW-friendly query."""

    def __init__(
        self,
        repository: VectorSearchRepository,
        *,
        embedding_dimension: int,
        embedding_model: str | None = None,
        default_top_k: int = 30,
    ) -> None:
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        self._repository = repository
        self._embedding_dimension = embedding_dimension
        self._embedding_model = embedding_model
        self._default_top_k = _validate_limit(default_top_k)

    async def retrieve(
        self,
        embedding: Sequence[float],
        *,
        filters: QueryFilters | None = None,
        limit: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Retrieve ranked cosine-similarity candidates."""
        if len(embedding) != self._embedding_dimension:
            raise ValueError(
                "embedding dimension mismatch: "
                f"expected {self._embedding_dimension}, received {len(embedding)}"
            )
        resolved_limit = self._default_top_k if limit is None else _validate_limit(limit)
        return await self._repository.vector_search(
            embedding,
            filters=filters or QueryFilters(),
            limit=resolved_limit,
            embedding_model=self._embedding_model,
        )


def _validate_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_RETRIEVAL_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_RETRIEVAL_LIMIT}")
    return limit
