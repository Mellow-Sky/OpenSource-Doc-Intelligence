from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import cast
from uuid import uuid4

import pytest

from app.domain.usage import EmbeddingBatchUsage
from app.repositories.usage_repository import UsageRecordCreate, UsageRepository
from app.services.pricing_service import PricingCatalog
from app.services.usage_service import UsageService


class _RecordingUsageRepository:
    def __init__(self) -> None:
        self.calls = 0
        self.records: list[UsageRecordCreate] = []

    async def add_many(self, records: Sequence[UsageRecordCreate]) -> list[object]:
        self.calls += 1
        self.records.extend(records)
        return []


def test_pricing_catalog_calculates_configured_input_and_output_cost(tmp_path) -> None:
    path = tmp_path / "pricing.yaml"
    path.write_text(
        """
pricing:
  provider-a:
    model-a:
      input_per_million_tokens: 2.5
      output_per_million_tokens: 10
""".strip(),
        encoding="utf-8",
    )

    cost = PricingCatalog.from_file(path).estimate(
        provider="provider-a",
        model="model-a",
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
    )

    assert cost == Decimal("7.5")


def test_unknown_model_cost_is_none_not_fabricated(tmp_path) -> None:
    path = tmp_path / "pricing.yaml"
    path.write_text("pricing: {}", encoding="utf-8")

    assert (
        PricingCatalog.from_file(path).estimate(
            provider="unknown",
            model="unknown",
            prompt_tokens=123,
            completion_tokens=45,
        )
        is None
    )


@pytest.mark.asyncio
async def test_embedding_batches_are_priced_and_persisted_in_one_call(tmp_path) -> None:
    path = tmp_path / "pricing.yaml"
    path.write_text(
        """
pricing:
  local:
    known:
      input_per_million_tokens: 2
      output_per_million_tokens: 0
""".strip(),
        encoding="utf-8",
    )
    repository = _RecordingUsageRepository()
    service = UsageService(
        cast(UsageRepository, repository),
        PricingCatalog.from_file(path),
    )
    request_id = uuid4()

    await service.record_embedding_batches(
        request_id=request_id,
        operation="ingestion_embedding",
        batches=[
            EmbeddingBatchUsage(
                model="known",
                provider="local",
                input_text_count=2,
                input_character_count=100,
                prompt_tokens=10,
                latency_ms=4.5,
            ),
            EmbeddingBatchUsage(
                model="unknown",
                provider="remote",
                input_text_count=1,
                input_character_count=20,
                prompt_tokens=3,
                latency_ms=2.0,
            ),
            EmbeddingBatchUsage(
                model="known",
                provider="local",
                input_text_count=1,
                input_character_count=12,
                prompt_tokens=0,
                latency_ms=1.0,
            ),
        ],
    )

    assert repository.calls == 1
    assert len(repository.records) == 3
    assert repository.records[0].request_id == request_id
    assert repository.records[0].estimated_cost == Decimal("0.00002")
    assert repository.records[1].estimated_cost is None
    assert repository.records[2].estimated_cost is None
    assert repository.records[0].input_text_count == 2
    assert repository.records[0].input_character_count == 100
