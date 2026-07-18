"""FastAPI application factory and production entry point."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.exception_handlers import RequestIDMiddleware, install_exception_handlers
from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.health import router as health_router
from app.api.routes.ingestion import router as ingestion_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.usage import router as usage_router
from app.container import AppContainer
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.metrics import PrometheusMiddleware
from app.core.telemetry import configure_telemetry
from app.db.session import Database
from app.ingestion.chunkers import create_llm_token_counter
from app.providers.base import LLMProvider
from app.providers.factory import (
    create_embedding_provider,
    create_llm_provider,
    create_reranker_provider,
)


def _runtime_llm_provider(settings: Settings) -> LLMProvider | None:
    """Leave an intentionally unconfigured LLM visible as a failed readiness check."""
    if settings.llm_provider in {"azure", "azure_openai"} and (
        not settings.llm_base_url
        or settings.llm_api_key is None
        or not settings.llm_api_version
        or not settings.llm_deployment
    ):
        return None
    if settings.llm_provider in {
        "openai",
        "openai_compatible",
        "ollama",
        "vllm",
    } and (not settings.llm_base_url or not settings.llm_model):
        return None
    return create_llm_provider(settings)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an isolated application instance suitable for production and tests."""
    runtime_settings = settings or get_settings()
    configure_logging(runtime_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = AppContainer(
            settings=runtime_settings,
            database=Database(runtime_settings),
        )
        try:
            container.embedding_provider = create_embedding_provider(runtime_settings)
            container.reranker_provider = create_reranker_provider(runtime_settings)
            container.llm_provider = _runtime_llm_provider(runtime_settings)
            container.context_token_counter = await create_llm_token_counter(
                runtime_settings,
                container.llm_provider,
            )
            app.state.container = container
            yield
        finally:
            await container.close()
            telemetry_provider = getattr(app.state, "telemetry_provider", None)
            if telemetry_provider is not None:
                await asyncio.to_thread(telemetry_provider.shutdown)

    app = FastAPI(
        title="OpenSource Doc Intelligence",
        description="Enterprise RAG for open-source project documentation",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if runtime_settings.app_env != "production" else None,
        redoc_url="/redoc" if runtime_settings.app_env != "production" else None,
    )
    app.dependency_overrides[get_settings] = lambda: runtime_settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Admin-API-Key",
            "X-Request-ID",
        ],
    )
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(RequestIDMiddleware)
    install_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    app.include_router(ingestion_router)
    app.include_router(evaluation_router)
    app.include_router(usage_router)
    app.include_router(metrics_router)
    app.state.telemetry_provider = configure_telemetry(app, runtime_settings)
    return app


app = create_app()


def run() -> None:
    """Run the development ASGI server from the console script."""
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=False)
