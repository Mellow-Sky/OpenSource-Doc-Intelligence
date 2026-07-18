"""Application-facing PostgreSQL keyword retrieval adapter."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.retrieval import QueryFilters, RetrievalCandidate
from app.repositories.retrieval_repository import MAX_RETRIEVAL_LIMIT


@runtime_checkable
class KeywordSearchRepository(Protocol):
    """Persistence capability required by the keyword retriever."""

    async def keyword_search(
        self,
        query: str,
        *,
        filters: QueryFilters | None = None,
        limit: int = 30,
    ) -> list[RetrievalCandidate]:
        """Return ranked keyword candidates."""
        ...


@runtime_checkable
class KeywordRetrieverProtocol(Protocol):
    """Interface consumed by the hybrid retrieval service."""

    async def retrieve(
        self,
        query: str,
        *,
        filters: QueryFilters | None = None,
        limit: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Retrieve keyword candidates for one normalized query."""
        ...


class KeywordRetriever:
    """Validate service input and delegate one batch query to PostgreSQL."""

    def __init__(
        self,
        repository: KeywordSearchRepository,
        *,
        default_top_k: int = 30,
        max_query_length: int = 4000,
    ) -> None:
        self._repository = repository
        self._default_top_k = _validate_limit(default_top_k)
        if max_query_length < 1:
            raise ValueError("max_query_length must be positive")
        self._max_query_length = max_query_length

    async def retrieve(
        self,
        query: str,
        *,
        filters: QueryFilters | None = None,
        limit: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Retrieve ranked FTS candidates without mutating the normalized query."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if len(normalized_query) > self._max_query_length:
            raise ValueError(f"query exceeds maximum length {self._max_query_length}")
        resolved_limit = self._default_top_k if limit is None else _validate_limit(limit)
        return await self._repository.keyword_search(
            normalized_query,
            filters=filters or QueryFilters(),
            limit=resolved_limit,
        )


def _validate_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_RETRIEVAL_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_RETRIEVAL_LIMIT}")
    return limit
