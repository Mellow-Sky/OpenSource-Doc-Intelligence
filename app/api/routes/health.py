"""Liveness and dependency-aware readiness endpoints."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text

from app import __version__
from app.api.dependencies import get_container
from app.container import AppContainer
from app.schemas.common import HealthResponse, ReadinessCheck, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return process liveness without depending on downstream services."""
    return HealthResponse(status="ok", service="opensource-doc-intelligence", version=__version__)


async def _timed_check(check: Callable[[], Awaitable[None]]) -> ReadinessCheck:
    started = time.perf_counter()
    try:
        await check()
    except Exception as exc:
        return ReadinessCheck(
            ready=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            detail=type(exc).__name__,
        )
    return ReadinessCheck(ready=True, latency_ms=(time.perf_counter() - started) * 1000)


@router.get("/ready", response_model=ReadinessResponse)
async def ready(
    response: Response,
    container: AppContainer = Depends(get_container),
) -> ReadinessResponse:
    """Check PostgreSQL, pgvector, and every configured model provider."""

    async def postgres_check() -> None:
        async with container.database.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def pgvector_check() -> None:
        async with container.database.engine.connect() as connection:
            extension_version = await connection.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            if extension_version is None:
                raise RuntimeError("pgvector extension is not installed")
            result = await connection.execute(
                text(
                    """
                    SELECT
                        format_type(attribute.atttypid, attribute.atttypmod) AS format_type,
                        attribute.atttypmod AS dimension
                    FROM pg_attribute AS attribute
                    WHERE attribute.attrelid = to_regclass('chunks')
                      AND attribute.attname = 'embedding'
                      AND attribute.attnum > 0
                      AND NOT attribute.attisdropped
                    """
                )
            )
            column = result.mappings().one_or_none()
            if column is None:
                raise RuntimeError("chunks.embedding is missing; run database migrations")
            expected_dimension = container.settings.database_vector_dimension
            expected_format = f"vector({expected_dimension})"
            if (
                column["format_type"] != expected_format
                or column["dimension"] != expected_dimension
            ):
                raise RuntimeError("chunks.embedding does not match DATABASE_VECTOR_DIMENSION")

    async def embedding_check() -> None:
        if container.embedding_provider is None:
            raise RuntimeError("embedding provider is not configured")
        await container.embedding_provider.healthcheck()

    async def reranker_check() -> None:
        if container.reranker_provider is None:
            raise RuntimeError("reranker provider is not configured")
        await container.reranker_provider.healthcheck()

    async def llm_check() -> None:
        if container.llm_provider is None:
            raise RuntimeError("LLM provider is not configured")
        await container.llm_provider.healthcheck()

    checks = {
        "postgresql": await _timed_check(postgres_check),
        "pgvector": await _timed_check(pgvector_check),
        "embedding": await _timed_check(embedding_check),
        "reranker": await _timed_check(reranker_check),
        "llm": await _timed_check(llm_check),
    }
    is_ready = all(check.ready for check in checks.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if is_ready else "not_ready", checks=checks)
