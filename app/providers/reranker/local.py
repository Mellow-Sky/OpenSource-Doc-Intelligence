"""Lazy local BGE cross-encoder reranker."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from app.core.exceptions import ProviderError
from app.providers.base import RerankerProvider, RerankResponse


class LocalRerankerProvider(RerankerProvider):
    """Run FlagEmbedding model initialization and scoring in worker threads."""

    def __init__(self, *, model: str, batch_size: int, max_concurrency: int = 2) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._model_name = model
        self._batch_size = batch_size
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()
        self._inference_gate = asyncio.Semaphore(max_concurrency)

    @property
    def name(self) -> str:
        return "local"

    @property
    def model(self) -> str:
        return self._model_name

    async def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model)
        return self._model

    def _load_model(self) -> Any:
        try:
            from FlagEmbedding import FlagReranker

            return FlagReranker(self._model_name, use_fp16=False)
        except (ImportError, OSError, RuntimeError) as exc:
            raise ProviderError("Unable to load the configured local reranker model") from exc

    async def healthcheck(self) -> None:
        await self._ensure_loaded()

    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse:
        if not documents:
            return RerankResponse(scores=[], model=self._model_name)
        async with self._inference_gate:
            model = await self._ensure_loaded()
            pairs = [[query, document] for document in documents]
            try:
                raw_scores = await asyncio.to_thread(
                    model.compute_score,
                    pairs,
                    batch_size=self._batch_size,
                    normalize=True,
                )
                if isinstance(raw_scores, (int, float)):
                    raw_scores = [raw_scores]
                scores = [float(score) for score in raw_scores]
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise ProviderError("Local reranker inference failed") from exc
        if len(scores) != len(documents):
            raise ProviderError("Local reranker returned a different score count")
        return RerankResponse(scores=scores, model=self._model_name)
