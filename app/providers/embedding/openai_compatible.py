"""OpenAI-compatible batched embedding provider."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import httpx
from pydantic import SecretStr

from app.core.exceptions import ProviderError
from app.providers.base import EmbeddingProvider, EmbeddingResponse, TokenUsage
from app.providers.http_client import RetryingJSONClient


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Embed batches through the widely supported ``POST /embeddings`` contract."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | None,
        model: str,
        dimension: int,
        batch_size: int,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        client: httpx.AsyncClient | None = None,
        request_headers: Mapping[str, str] | None = None,
        healthcheck_mode: Literal["catalog", "inference"] = "catalog",
    ) -> None:
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._healthcheck_mode = healthcheck_mode
        self._http = RetryingJSONClient(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_concurrency=max_concurrency,
            client=client,
            request_headers=request_headers,
        )

    @property
    def name(self) -> str:
        return "openai_compatible"

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def healthcheck(self) -> None:
        """Validate the configured model without performing inference by default."""
        if self._healthcheck_mode == "inference":
            await self.embed(["healthcheck"])
            return
        await self._http.healthcheck(model=self._model)

    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        if not texts:
            raise ProviderError("Embedding input batch must not be empty")
        batches = [
            texts[index : index + self._batch_size]
            for index in range(0, len(texts), self._batch_size)
        ]
        responses = await asyncio.gather(*(self._embed_batch(batch) for batch in batches))
        vectors = [vector for response_vectors, _ in responses for vector in response_vectors]
        prompt_tokens = sum(usage.prompt_tokens for _, usage in responses)
        return EmbeddingResponse(
            vectors=vectors,
            model=self._model,
            dimension=self._dimension,
            usage=TokenUsage(prompt_tokens=prompt_tokens),
        )

    async def _embed_batch(self, texts: Sequence[str]) -> tuple[list[list[float]], TokenUsage]:
        payload = self._embedding_payload(texts)
        decoded = await self._http.request_json("POST", self._embedding_resource(), payload=payload)
        data = decoded.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise ProviderError("Embedding provider returned a different vector count")
        indexed: list[tuple[int, list[float]]] = []
        for fallback_index, item in enumerate(data):
            if not isinstance(item, Mapping):
                raise ProviderError("Embedding provider returned an invalid data item")
            raw_vector = item.get("embedding")
            if not isinstance(raw_vector, list) or len(raw_vector) != self._dimension:
                raise ProviderError("Embedding provider returned an unexpected dimension")
            if not all(isinstance(value, (int, float)) for value in raw_vector):
                raise ProviderError("Embedding provider returned non-numeric vector values")
            index = item.get("index", fallback_index)
            if not isinstance(index, int):
                raise ProviderError("Embedding provider returned an invalid vector index")
            indexed.append((index, [float(value) for value in raw_vector]))
        indexed.sort(key=lambda pair: pair[0])
        if [index for index, _ in indexed] != list(range(len(texts))):
            raise ProviderError("Embedding provider returned duplicate or missing vector indexes")
        usage_data = decoded.get("usage")
        prompt_tokens = 0
        if isinstance(usage_data, Mapping):
            value = usage_data.get("prompt_tokens", usage_data.get("total_tokens", 0))
            if isinstance(value, int) and value >= 0:
                prompt_tokens = value
        return [vector for _, vector in indexed], TokenUsage(prompt_tokens=prompt_tokens)

    def _embedding_resource(self) -> str:
        return "embeddings"

    def _embedding_payload(self, texts: Sequence[str]) -> dict[str, Any]:
        return {"model": self._model, "input": list(texts)}

    async def close(self) -> None:
        await self._http.close()
