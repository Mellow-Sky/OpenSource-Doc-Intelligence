"""HTTP reranker provider compatible with common ``POST /rerank`` APIs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import httpx
from pydantic import SecretStr

from app.core.exceptions import ProviderError
from app.providers.base import RerankerProvider, RerankResponse
from app.providers.http_client import RetryingJSONClient


class RemoteRerankerProvider(RerankerProvider):
    """Batch and restore scores by response index for deterministic ordering."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | None,
        model: str,
        batch_size: int,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        client: httpx.AsyncClient | None = None,
        healthcheck_mode: Literal["inference", "endpoint"] = "inference",
        healthcheck_resource: str | None = None,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._healthcheck_mode = healthcheck_mode
        self._healthcheck_resource = healthcheck_resource
        self._http = RetryingJSONClient(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_concurrency=max_concurrency,
            client=client,
        )

    @property
    def name(self) -> str:
        return "remote"

    @property
    def model(self) -> str:
        return self._model

    async def healthcheck(self) -> None:
        """Probe the rerank contract, never an unrelated model-list endpoint."""
        if self._healthcheck_mode == "endpoint":
            if self._healthcheck_resource is None:
                raise ProviderError("Reranker healthcheck endpoint is not configured")
            await self._http.request_json("GET", self._healthcheck_resource)
            return
        await self.rerank("healthcheck", ["healthcheck"])

    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse:
        if not documents:
            return RerankResponse(scores=[], model=self._model)
        batches = [
            documents[index : index + self._batch_size]
            for index in range(0, len(documents), self._batch_size)
        ]
        results = await asyncio.gather(*(self._rerank_batch(query, batch) for batch in batches))
        return RerankResponse(
            scores=[score for batch_scores in results for score in batch_scores],
            model=self._model,
        )

    async def _rerank_batch(self, query: str, documents: Sequence[str]) -> list[float]:
        payload: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": list(documents),
            "top_n": len(documents),
            "return_documents": False,
        }
        decoded = await self._http.request_json("POST", "rerank", payload=payload)
        raw_results = decoded.get("results", decoded.get("data"))
        if not isinstance(raw_results, list) or len(raw_results) != len(documents):
            raise ProviderError("Reranker returned a different score count")
        indexed: list[tuple[int, float]] = []
        for fallback_index, item in enumerate(raw_results):
            if not isinstance(item, Mapping):
                raise ProviderError("Reranker returned an invalid result item")
            index = item.get("index", fallback_index)
            score = item.get("relevance_score", item.get("score"))
            if not isinstance(index, int) or not isinstance(score, (int, float)):
                raise ProviderError("Reranker returned an invalid index or score")
            indexed.append((index, float(score)))
        indexed.sort(key=lambda pair: pair[0])
        if [index for index, _ in indexed] != list(range(len(documents))):
            raise ProviderError("Reranker returned duplicate or missing result indexes")
        return [score for _, score in indexed]

    async def close(self) -> None:
        await self._http.close()
