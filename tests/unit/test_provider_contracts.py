"""Provider ports support deterministic mocks without paid APIs."""

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from app.providers.base import (
    EmbeddingProvider,
    EmbeddingResponse,
    GenerationResponse,
    LLMProvider,
    RerankerProvider,
    RerankResponse,
)


class FakeEmbedding(EmbeddingProvider):
    name = "fake"
    dimension = 2

    async def healthcheck(self) -> None:
        return None

    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        return EmbeddingResponse([[float(len(text)), 1.0] for text in texts], self.name, 2)


class FakeLLM(LLMProvider):
    name = "fake"

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
        return GenerationResponse(text=messages[-1]["content"], model=self.name)

    async def _stream(self, messages: Sequence[dict[str, str]]) -> AsyncIterator[str]:
        yield messages[-1]["content"]

    def stream(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        return self._stream(messages)


class FakeReranker(RerankerProvider):
    name = "fake"

    async def healthcheck(self) -> None:
        return None

    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse:
        return RerankResponse([float(query in document) for document in documents], self.name)


@pytest.mark.asyncio
async def test_provider_ports_are_replaceable() -> None:
    embedding = await FakeEmbedding().embed(["abc"])
    generation = await FakeLLM().generate(
        messages=[{"role": "user", "content": "answer"}], max_tokens=10
    )
    rerank = await FakeReranker().rerank("needle", ["needle here", "other"])

    assert embedding.vectors == [[3.0, 1.0]]
    assert generation.text == "answer"
    assert rerank.scores == [1.0, 0.0]
