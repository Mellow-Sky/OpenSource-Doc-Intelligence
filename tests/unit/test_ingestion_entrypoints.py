"""Composition tests for model-aware tokenization in ingestion entry points."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from app import worker as ingestion_worker
from app.container import AppContainer
from app.core.config import Settings
from app.ingestion.chunkers import RegexTokenCounter, TokenCounter
from app.ingestion.incremental import SyncStats
from app.providers.base import EmbeddingProvider
from app.services.ingestion_service import SyncResult
from app.services.pricing_service import PricingCatalog
from scripts import ingest_kubernetes


class _Provider:
    closed = False

    async def close(self) -> None:
        self.closed = True


class _Database:
    closed = False

    def __init__(self, _settings: Settings) -> None:
        self.session_factory = object()

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_production_worker_composes_configured_token_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The durable worker must pass the resolved model counter into ingestion."""

    settings = Settings(_env_file=None, app_env="test")
    provider = _Provider()
    database = _Database(settings)
    counter = RegexTokenCounter()
    composed_with: list[object] = []
    worker_counters: list[TokenCounter] = []
    once_values: list[bool] = []

    async def create_counter(
        runtime: Settings,
        embedding_provider: EmbeddingProvider | None,
    ) -> TokenCounter:
        assert runtime is settings
        composed_with.append(embedding_provider)
        return counter

    class Worker:
        def __init__(self, _container: object, *, token_counter: TokenCounter) -> None:
            worker_counters.append(token_counter)

        async def run(self, *, once: bool = False) -> None:
            once_values.append(once)

    monkeypatch.setattr(ingestion_worker, "get_settings", lambda: settings)
    monkeypatch.setattr(ingestion_worker, "Database", lambda _settings: database)
    monkeypatch.setattr(
        ingestion_worker,
        "create_embedding_provider",
        lambda _settings: cast(EmbeddingProvider, provider),
    )
    monkeypatch.setattr(ingestion_worker, "create_chunk_token_counter", create_counter)
    monkeypatch.setattr(ingestion_worker, "IngestionWorker", Worker)
    monkeypatch.setattr(ingestion_worker, "_install_signal_handlers", lambda _worker: None)

    await ingestion_worker.run_worker(once=True)

    assert composed_with == [provider]
    assert worker_counters == [counter]
    assert once_values == [True]
    assert provider.closed is True
    assert database.closed is True


@pytest.mark.asyncio
async def test_ingest_cli_composes_configured_token_counter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI must resolve the model tokenizer before constructing the service."""

    source_config = tmp_path / "sources.yaml"
    source_config.write_text("sources: []\n", encoding="utf-8")
    provider = _Provider()
    counter = RegexTokenCounter()
    counter_calls: list[tuple[Settings, object]] = []
    service_counters: list[TokenCounter | None] = []
    service_pricing: list[PricingCatalog | None] = []

    async def create_counter(
        settings: Settings,
        embedding_provider: EmbeddingProvider | None,
    ) -> TokenCounter:
        counter_calls.append((settings, embedding_provider))
        return counter

    class Service:
        def __init__(self, *_args: Any, token_counter: TokenCounter | None = None, **_kwargs: Any):
            service_counters.append(token_counter)
            service_pricing.append(_kwargs.get("pricing_catalog"))

    monkeypatch.setattr(ingest_kubernetes, "Database", _Database)
    monkeypatch.setattr(ingest_kubernetes, "create_chunk_token_counter", create_counter)
    monkeypatch.setattr(ingest_kubernetes, "IngestionService", Service)
    settings = Settings(_env_file=None, app_env="test")
    args = Namespace(
        source_id=None,
        all=False,
        config=source_config,
        dry_run=True,
        no_delete=False,
        cache_dir=tmp_path / "cache",
    )

    status = await ingest_kubernetes.run(
        args,
        settings=settings,
        embedding_provider=cast(EmbeddingProvider, provider),
    )

    assert status == 0
    assert counter_calls == [(settings, provider)]
    assert service_counters == [counter]
    assert isinstance(service_pricing[0], PricingCatalog)
    assert provider.closed is True


def test_worker_composes_configured_embedding_pricing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text("pricing: {}\n", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        app_env="test",
        pricing_config_path=pricing_path,
    )
    composed_pricing: list[PricingCatalog | None] = []

    class Service:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            composed_pricing.append(kwargs.get("pricing_catalog"))

    container = SimpleNamespace(
        database=SimpleNamespace(session_factory=object()),
        embedding_provider=cast(EmbeddingProvider, _Provider()),
        settings=settings,
    )
    monkeypatch.setattr(ingestion_worker, "IngestionService", Service)

    ingestion_worker.IngestionWorker(
        cast(AppContainer, container),
        token_counter=RegexTokenCounter(),
    )

    assert isinstance(composed_pricing[0], PricingCatalog)


@pytest.mark.asyncio
async def test_worker_correlates_embedding_usage_with_ingestion_job_id() -> None:
    source_id = uuid4()
    job_id = uuid4()
    calls: list[tuple[UUID, UUID | None]] = []

    class Service:
        async def sync_source(
            self,
            requested_source_id: UUID,
            *,
            dry_run: bool,
            allow_delete_missing: bool,
            request_id: UUID | None = None,
        ) -> SyncResult:
            assert dry_run is False
            assert allow_delete_missing is True
            calls.append((requested_source_id, request_id))
            return SyncResult(
                source_id=requested_source_id,
                stats=SyncStats(),
                complete_snapshot=True,
                dry_run=False,
                request_id=request_id or uuid4(),
            )

    async def heartbeat(
        _job_id: UUID,
        _lease_started_at: datetime,
        stop: asyncio.Event,
    ) -> bool:
        await stop.wait()
        return True

    worker = object.__new__(ingestion_worker.IngestionWorker)
    worker._service = Service()  # type: ignore[attr-defined]
    worker._heartbeat = heartbeat  # type: ignore[method-assign]

    result = await worker._synchronize_with_heartbeat(
        job_id,
        source_id,
        datetime(2026, 7, 18, tzinfo=UTC),
        {},
    )

    assert calls == [(source_id, job_id)]
    assert result.request_id == job_id
