"""Dependency-aware readiness reports complete and partial availability."""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from app.api.dependencies import get_container
from app.container import AppContainer
from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.db.session import Database
from app.main import create_app
from app.providers.base import EmbeddingProvider, LLMProvider, RerankerProvider


class _ColumnResult:
    def __init__(self, column: dict[str, object] | None) -> None:
        self._column = column

    def mappings(self) -> _ColumnResult:
        return self

    def one_or_none(self) -> dict[str, object] | None:
        return self._column


class _Connection:
    def __init__(
        self,
        *,
        extension_version: str | None = "0.8.0",
        column: dict[str, object] | None = None,
    ) -> None:
        self.extension_version = extension_version
        self.column = column or {"format_type": "vector(1024)", "dimension": 1024}

    async def __aenter__(self) -> _Connection:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def scalar(self, _statement: object) -> str | None:
        return self.extension_version

    async def execute(self, statement: object) -> _ColumnResult:
        if "pg_attribute" in str(statement):
            return _ColumnResult(self.column)
        return _ColumnResult(None)


class _Engine:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def connect(self) -> _Connection:
        return self.connection


class _Database:
    def __init__(self, connection: _Connection) -> None:
        self.engine = _Engine(connection)


class _Provider:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    async def healthcheck(self) -> None:
        if self.failure is not None:
            raise self.failure


def _container(
    *,
    connection: _Connection | None = None,
    reranker_failure: Exception | None = None,
) -> AppContainer:
    settings = Settings(
        _env_file=None,
        app_env="test",
        embedding_provider="deterministic",
        reranker_provider="deterministic",
        llm_provider="deterministic",
    )
    return AppContainer(
        settings=settings,
        database=cast(Database, _Database(connection or _Connection())),
        embedding_provider=cast(EmbeddingProvider, _Provider()),
        reranker_provider=cast(RerankerProvider, _Provider(reranker_failure)),
        llm_provider=cast(LLMProvider, _Provider()),
    )


async def _readiness(container: AppContainer) -> httpx.Response:
    app = create_app(container.settings)

    async def container_override() -> AppContainer:
        return container

    app.dependency_overrides[get_container] = container_override
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/ready")


@pytest.mark.asyncio
async def test_ready_returns_200_when_database_schema_and_providers_are_ready() -> None:
    response = await _readiness(_container())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert set(payload["checks"]) == {
        "postgresql",
        "pgvector",
        "embedding",
        "reranker",
        "llm",
    }
    assert all(check["ready"] for check in payload["checks"].values())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "column",
    [
        None,
        {"format_type": "vector(768)", "dimension": 768},
        {"format_type": "vector", "dimension": -1},
    ],
)
async def test_ready_returns_503_when_pgvector_migration_is_missing_or_wrong(
    column: dict[str, Any] | None,
) -> None:
    connection = _Connection(column={"format_type": "sentinel", "dimension": 0})
    connection.column = column

    response = await _readiness(_container(connection=connection))

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["checks"]["pgvector"]["ready"] is False
    assert payload["checks"]["pgvector"]["latency_ms"] >= 0
    assert payload["checks"]["pgvector"]["detail"] == "RuntimeError"
    assert payload["checks"]["postgresql"]["ready"] is True


@pytest.mark.asyncio
async def test_ready_keeps_successful_checks_when_one_provider_fails() -> None:
    response = await _readiness(_container(reranker_failure=ProviderError("reranker unavailable")))

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["reranker"]["ready"] is False
    assert checks["reranker"]["detail"] == "ProviderError"
    assert all(checks[name]["ready"] for name in ("postgresql", "pgvector", "embedding", "llm"))
