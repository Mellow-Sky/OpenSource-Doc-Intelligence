"""Deterministic providers enabled only by explicit test configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from hashlib import sha256

from app.providers.base import (
    EmbeddingProvider,
    EmbeddingResponse,
    RerankerProvider,
    RerankResponse,
)

_WORD_RE = re.compile(r"[\w.-]+", re.UNICODE)


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Hash text into a stable normalized vector for offline tests."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    @property
    def name(self) -> str:
        return "deterministic"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return "deterministic-sha256"

    async def healthcheck(self) -> None:
        return None

    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        vectors = [_hash_vector(text, self._dimension) for text in texts]
        return EmbeddingResponse(vectors, "deterministic-sha256", self._dimension)


class DeterministicRerankerProvider(RerankerProvider):
    """Score token overlap for reproducible fallback-path tests."""

    @property
    def name(self) -> str:
        return "deterministic"

    @property
    def model(self) -> str:
        return "deterministic-overlap"

    async def healthcheck(self) -> None:
        return None

    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse:
        query_terms = {term.casefold() for term in _WORD_RE.findall(query)}
        scores = []
        for document in documents:
            document_terms = {term.casefold() for term in _WORD_RE.findall(document)}
            union = query_terms | document_terms
            scores.append(len(query_terms & document_terms) / len(union) if union else 0.0)
        return RerankResponse(scores=scores, model="deterministic-overlap")


def _hash_vector(text: str, dimension: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        digest = sha256(f"{counter}:{text}".encode()).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    vector = values[:dimension]
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]
