"""Asynchronous model provider interfaces used by application services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-reported or locally estimated token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    vectors: list[list[float]]
    model: str
    dimension: int
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    text: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RerankResponse:
    scores: list[float]
    model: str


class Provider(ABC):
    """Common health contract for external and local model providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a stable provider identifier."""

    @abstractmethod
    async def healthcheck(self) -> None:
        """Raise ProviderError when the provider is unavailable."""

    async def close(self) -> None:
        """Release provider-owned resources; implementations may override."""
        return None


class EmbeddingProvider(Provider):
    @property
    def model(self) -> str:
        """Return the model identifier; simple adapters may reuse their provider name."""
        return self.name

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension produced by this provider."""

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        """Embed a non-empty batch while preserving input order."""


class LLMProvider(Provider):
    @abstractmethod
    async def generate(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        """Generate a complete response from chat messages."""

    @abstractmethod
    def stream(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream response text deltas."""


class RerankerProvider(Provider):
    @property
    def model(self) -> str:
        """Return the model identifier; simple adapters may reuse their provider name."""
        return self.name

    @abstractmethod
    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse:
        """Score a document batch against one query while preserving input order."""
