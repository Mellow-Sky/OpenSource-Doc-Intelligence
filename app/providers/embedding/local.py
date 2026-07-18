"""Lazy local sentence-transformers embedding provider."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from app.core.exceptions import ProviderError
from app.providers.base import EmbeddingProvider, EmbeddingResponse, TokenUsage


class LocalEmbeddingProvider(EmbeddingProvider):
    """Run blocking model loading and inference outside the event loop."""

    def __init__(
        self,
        *,
        model: str,
        dimension: int,
        batch_size: int,
        max_concurrency: int = 2,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._model_name = model
        self._dimension = dimension
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

    @property
    def dimension(self) -> int:
        return self._dimension

    async def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model)
        return self._model

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._model_name)
        except (ImportError, OSError, RuntimeError) as exc:
            raise ProviderError("Unable to load the configured local embedding model") from exc
        actual_dimension = model.get_sentence_embedding_dimension()
        if actual_dimension != self._dimension:
            raise ProviderError(
                "Local embedding model dimension does not match EMBEDDING_DIMENSION",
                details={"configured": self._dimension, "actual": actual_dimension},
            )
        return model

    async def healthcheck(self) -> None:
        await self._ensure_loaded()

    async def get_tokenizer(self) -> Any:
        """Share the loaded sentence-transformers tokenizer with ingestion."""

        model = await self._ensure_loaded()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise ProviderError(
                "Configured local embedding model does not expose a tokenizer",
                details={"model": self._model_name},
            )
        return tokenizer

    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        if not texts:
            raise ProviderError("Embedding input batch must not be empty")
        async with self._inference_gate:
            model = await self._ensure_loaded()
            tokenizer = await self.get_tokenizer()
            try:
                encoded, prompt_tokens = await asyncio.to_thread(
                    self._encode_and_count,
                    model,
                    tokenizer,
                    list(texts),
                )
                vectors = [[float(value) for value in row] for row in encoded.tolist()]
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                raise ProviderError("Local embedding inference failed") from exc
        if len(vectors) != len(texts) or any(len(vector) != self._dimension for vector in vectors):
            raise ProviderError("Local embedding model returned an unexpected output shape")
        return EmbeddingResponse(
            vectors=vectors,
            model=self._model_name,
            dimension=self._dimension,
            usage=TokenUsage(prompt_tokens=prompt_tokens),
        )

    def _encode_and_count(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> tuple[Any, int]:
        """Run blocking inference and token accounting on the same worker thread."""

        encoded = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        prompt_tokens = sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
        return encoded, prompt_tokens
