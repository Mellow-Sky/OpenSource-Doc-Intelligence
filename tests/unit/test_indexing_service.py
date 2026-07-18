"""Embedding backfill performs inference between short database sessions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pytest

from app.providers.base import EmbeddingProvider, EmbeddingResponse, TokenUsage
from app.repositories.chunk_repository import ChunkEmbeddingUpdate, PendingEmbeddingChunk
from app.repositories.usage_repository import UsageRecordCreate
from app.services.indexing_service import IndexingService
from app.services.pricing_service import PricingCatalog


class FakeEmbedding(EmbeddingProvider):
    name = "fake"
    model = "fake-v2"
    dimension = 2

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def healthcheck(self) -> None:
        return None

    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        self.batch_sizes.append(len(texts))
        return EmbeddingResponse(
            vectors=[[float(len(text)), 1.0] for text in texts],
            model=self.model,
            dimension=self.dimension,
            usage=TokenUsage(prompt_tokens=len(texts)),
        )


class FakeRepository:
    def __init__(self) -> None:
        self.pending = [
            PendingEmbeddingChunk(uuid4(), "a" * 64, "first"),
            PendingEmbeddingChunk(uuid4(), "b" * 64, "second"),
            PendingEmbeddingChunk(uuid4(), "c" * 64, "third"),
        ]
        self.updates: list[ChunkEmbeddingUpdate] = []
        self.requested_model = ""

    async def list_needing_embedding(self, *, model: str, dimension: int, limit: int):
        self.requested_model = model
        assert dimension == 2
        return self.pending[:limit]

    async def update_embeddings(self, records, *, updated_at):
        self.updates = list(records)
        return len(records) - 1  # Simulate one chunk changing during inference.


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def begin(self):
        return self


class FakeUsageRepository:
    def __init__(self) -> None:
        self.add_calls = 0
        self.records: list[UsageRecordCreate] = []

    async def add_many(self, records: Sequence[UsageRecordCreate]) -> list[object]:
        self.add_calls += 1
        self.records.extend(records)
        return []


def _pricing(path: Path) -> PricingCatalog:
    path.write_text(
        """
pricing:
  fake:
    fake-v2:
      input_per_million_tokens: 2
      output_per_million_tokens: 0
""".strip(),
        encoding="utf-8",
    )
    return PricingCatalog.from_file(path)


@pytest.mark.asyncio
async def test_indexing_service_batches_and_detects_optimistic_stale_write(
    tmp_path: Path,
) -> None:
    repository = FakeRepository()
    usage_repository = FakeUsageRepository()
    provider = FakeEmbedding()
    service = IndexingService(
        session_factory=FakeSession,
        provider=provider,
        dimension=2,
        batch_size=2,
        repository_factory=lambda _session: repository,
        usage_repository_factory=lambda _session: usage_repository,
        pricing_catalog=_pricing(tmp_path / "pricing.yaml"),
    )

    stats = await service.run_once(limit=3)

    assert stats.selected == 3
    assert stats.indexed == 2
    assert stats.stale == 1
    assert provider.batch_sizes == [2, 1]
    assert stats.request_id is not None
    assert repository.requested_model == "fake-v2"
    assert [update.content_hash for update in repository.updates] == [
        "a" * 64,
        "b" * 64,
        "c" * 64,
    ]
    assert usage_repository.add_calls == 1
    assert [record.operation for record in usage_repository.records] == [
        "indexing_embedding",
        "indexing_embedding",
    ]
    assert {record.request_id for record in usage_repository.records} == {stats.request_id}
    assert [record.input_text_count for record in usage_repository.records] == [2, 1]
    assert [record.input_character_count for record in usage_repository.records] == [11, 5]
    assert [record.prompt_tokens for record in usage_repository.records] == [2, 1]
    assert all(record.estimated_cost is not None for record in usage_repository.records)
    assert all(record.latency_ms >= 0 for record in usage_repository.records)
