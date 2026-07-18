"""Composition helpers for configuration-selected model providers."""

from __future__ import annotations

from typing import Literal

from pydantic import SecretStr

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.providers.base import EmbeddingProvider, LLMProvider, RerankerProvider
from app.providers.embedding import (
    AzureOpenAIEmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from app.providers.llm import (
    AzureOpenAILLMProvider,
    DeterministicLLMProvider,
    OpenAICompatibleLLMProvider,
)
from app.providers.reranker import LocalRerankerProvider, RemoteRerankerProvider
from app.providers.testing import DeterministicEmbeddingProvider, DeterministicRerankerProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Build the configured generation adapter without issuing a network request."""
    if settings.llm_provider in {"azure", "azure_openai"}:
        return AzureOpenAILLMProvider(
            endpoint=_required(settings.llm_base_url, "LLM_BASE_URL", "Azure OpenAI"),
            api_key=_required_secret(settings.llm_api_key, "LLM_API_KEY", "Azure OpenAI"),
            api_version=_required(
                settings.llm_api_version,
                "LLM_API_VERSION",
                "Azure OpenAI",
            ),
            deployment=_required(
                settings.llm_deployment,
                "LLM_DEPLOYMENT",
                "Azure OpenAI",
            ),
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.provider_max_retries,
            max_concurrency=settings.llm_max_concurrency,
            healthcheck_mode=_remote_healthcheck_mode(
                settings.llm_healthcheck_mode,
                auto_default="inference",
            ),
        )
    if settings.llm_provider in {"openai", "openai_compatible", "ollama", "vllm"}:
        if not settings.llm_base_url:
            raise ConfigurationError("LLM_BASE_URL is required for an OpenAI-compatible provider")
        if not settings.llm_model:
            raise ConfigurationError("LLM_MODEL is required for an OpenAI-compatible provider")
        return OpenAICompatibleLLMProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.provider_max_retries,
            max_concurrency=settings.llm_max_concurrency,
            healthcheck_mode=_remote_healthcheck_mode(
                settings.llm_healthcheck_mode,
                auto_default="catalog",
            ),
        )
    if settings.llm_provider == "deterministic" and settings.app_env == "test":
        return DeterministicLLMProvider()
    raise ConfigurationError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


def create_judge_provider(settings: Settings) -> LLMProvider | None:
    """Build an independently configured evaluation judge, when enabled."""
    if settings.judge_provider is None and settings.judge_model is None:
        return None
    provider = settings.judge_provider or "openai_compatible"
    if provider in {"azure", "azure_openai"}:
        return AzureOpenAILLMProvider(
            endpoint=_required(settings.judge_base_url, "JUDGE_BASE_URL", "Azure OpenAI"),
            api_key=_required_secret(
                settings.judge_api_key,
                "JUDGE_API_KEY",
                "Azure OpenAI",
            ),
            api_version=_required(
                settings.judge_api_version,
                "JUDGE_API_VERSION",
                "Azure OpenAI",
            ),
            deployment=_required(
                settings.judge_deployment,
                "JUDGE_DEPLOYMENT",
                "Azure OpenAI",
            ),
            model=settings.judge_model,
            timeout_seconds=settings.judge_timeout_seconds,
            max_retries=settings.provider_max_retries,
            max_concurrency=settings.llm_max_concurrency,
        )
    if provider in {"openai", "openai_compatible", "ollama", "vllm"}:
        if not settings.judge_base_url:
            raise ConfigurationError("JUDGE_BASE_URL is required when a judge is configured")
        if not settings.judge_model:
            raise ConfigurationError("JUDGE_MODEL is required when a judge is configured")
        return OpenAICompatibleLLMProvider(
            base_url=settings.judge_base_url,
            api_key=settings.judge_api_key,
            model=settings.judge_model,
            timeout_seconds=settings.judge_timeout_seconds,
            max_retries=settings.provider_max_retries,
            max_concurrency=settings.llm_max_concurrency,
        )
    if provider == "deterministic" and settings.app_env == "test":
        return DeterministicLLMProvider()
    raise ConfigurationError(f"Unsupported JUDGE_PROVIDER: {provider}")


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the selected embedding adapter without loading local weights eagerly."""
    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider(
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            max_concurrency=settings.embedding_max_concurrency,
        )
    if settings.embedding_provider in {"azure", "azure_openai"}:
        return AzureOpenAIEmbeddingProvider(
            endpoint=_required(
                settings.embedding_base_url,
                "EMBEDDING_BASE_URL",
                "Azure OpenAI",
            ),
            api_key=_required_secret(
                settings.embedding_api_key,
                "EMBEDDING_API_KEY",
                "Azure OpenAI",
            ),
            api_version=_required(
                settings.embedding_api_version,
                "EMBEDDING_API_VERSION",
                "Azure OpenAI",
            ),
            deployment=_required(
                settings.embedding_deployment,
                "EMBEDDING_DEPLOYMENT",
                "Azure OpenAI",
            ),
            model=None,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_retries=settings.provider_max_retries,
            max_concurrency=settings.embedding_max_concurrency,
            healthcheck_mode=_remote_healthcheck_mode(
                settings.embedding_healthcheck_mode,
                auto_default="inference",
            ),
        )
    if settings.embedding_provider in {"openai", "openai_compatible", "remote"}:
        if not settings.embedding_base_url:
            raise ConfigurationError("EMBEDDING_BASE_URL is required for a remote provider")
        return OpenAICompatibleEmbeddingProvider(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_retries=settings.provider_max_retries,
            max_concurrency=settings.embedding_max_concurrency,
            healthcheck_mode=_remote_healthcheck_mode(
                settings.embedding_healthcheck_mode,
                auto_default="catalog",
            ),
        )
    if settings.embedding_provider == "deterministic" and settings.app_env == "test":
        return DeterministicEmbeddingProvider(settings.embedding_dimension)
    raise ConfigurationError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")


def create_reranker_provider(settings: Settings) -> RerankerProvider:
    """Build the selected cross-encoder adapter without loading local weights eagerly."""
    if settings.reranker_provider == "local":
        return LocalRerankerProvider(
            model=settings.reranker_model,
            batch_size=settings.reranker_batch_size,
            max_concurrency=settings.reranker_max_concurrency,
        )
    if settings.reranker_provider in {"openai", "openai_compatible", "remote"}:
        if not settings.reranker_base_url:
            raise ConfigurationError("RERANKER_BASE_URL is required for a remote provider")
        return RemoteRerankerProvider(
            base_url=settings.reranker_base_url,
            api_key=settings.reranker_api_key,
            model=settings.reranker_model,
            batch_size=settings.reranker_batch_size,
            timeout_seconds=settings.reranker_timeout_seconds,
            max_retries=settings.provider_max_retries,
            max_concurrency=settings.reranker_max_concurrency,
            healthcheck_mode=(
                "inference"
                if settings.reranker_healthcheck_mode == "auto"
                else settings.reranker_healthcheck_mode
            ),
            healthcheck_resource=settings.reranker_healthcheck_resource,
        )
    if settings.reranker_provider == "deterministic" and settings.app_env == "test":
        return DeterministicRerankerProvider()
    raise ConfigurationError(f"Unsupported RERANKER_PROVIDER: {settings.reranker_provider}")


def _required(value: str | None, setting: str, provider: str) -> str:
    normalized = value.strip() if value is not None else ""
    if not normalized:
        raise ConfigurationError(f"{setting} is required for {provider}")
    return normalized


def _required_secret(
    value: SecretStr | None,
    setting: str,
    provider: str,
) -> SecretStr:
    if value is None or not value.get_secret_value().strip():
        raise ConfigurationError(f"{setting} is required for {provider}")
    return value


def _remote_healthcheck_mode(
    configured: Literal["auto", "catalog", "inference"],
    *,
    auto_default: Literal["catalog", "inference"],
) -> Literal["catalog", "inference"]:
    """Resolve provider-aware defaults while keeping an explicit override."""
    return auto_default if configured == "auto" else configured
