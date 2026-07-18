"""Composition root containing infrastructure objects and replaceable providers."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.db.session import Database
from app.ingestion.chunkers import TokenCounter
from app.providers.base import EmbeddingProvider, LLMProvider, RerankerProvider


@dataclass(slots=True)
class AppContainer:
    """Explicit dependencies shared by API routes and background workers."""

    settings: Settings
    database: Database
    embedding_provider: EmbeddingProvider | None = None
    reranker_provider: RerankerProvider | None = None
    llm_provider: LLMProvider | None = None
    context_token_counter: TokenCounter | None = None

    async def close(self) -> None:
        """Gracefully close providers and database connections."""
        providers = (self.embedding_provider, self.reranker_provider, self.llm_provider)
        for provider in providers:
            if provider is not None:
                await provider.close()
        await self.database.close()
