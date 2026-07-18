"""Model-aware token counters and their asynchronous composition factory."""

from __future__ import annotations

import asyncio
import importlib
from typing import Any, Protocol, runtime_checkable

import structlog

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, ProviderError
from app.ingestion.chunkers.structure import RegexTokenCounter, TokenCounter
from app.providers.base import EmbeddingProvider, LLMProvider


@runtime_checkable
class TokenizerSource(Protocol):
    """Optional provider capability for sharing an already loaded tokenizer."""

    async def get_tokenizer(self) -> Any:
        """Return the tokenizer owned by the provider's model instance."""


class HuggingFaceTokenCounter:
    """Count tokens with a Hugging Face-compatible tokenizer.

    The counter itself is synchronous because structure-aware chunking is a
    synchronous, CPU-bound transformation. Model/tokenizer acquisition is
    exposed through :meth:`from_model` so callers can keep blocking I/O off the
    event loop.
    """

    def __init__(self, tokenizer: Any, *, model_name: str) -> None:
        if not hasattr(tokenizer, "encode"):
            raise ProviderError(
                "Configured tokenizer does not expose an encode method",
                details={"model": model_name},
            )
        self._tokenizer = tokenizer
        self.model_name = model_name

    @classmethod
    async def from_model(cls, model_name: str) -> HuggingFaceTokenCounter:
        """Load only the tokenizer on a worker thread, never on the event loop."""

        tokenizer = await asyncio.to_thread(_load_huggingface_tokenizer, model_name)
        return cls(tokenizer, model_name=model_name)

    def count(self, text: str) -> int:
        """Count model tokens without adding synthetic BOS/EOS tokens."""

        try:
            token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ProviderError(
                "Configured tokenizer failed to encode chunk content",
                details={"model": self.model_name},
            ) from exc
        try:
            return len(token_ids)
        except TypeError as exc:
            raise ProviderError(
                "Configured tokenizer returned an invalid encoding",
                details={"model": self.model_name},
            ) from exc


async def create_chunk_token_counter(
    settings: Settings,
    embedding_provider: EmbeddingProvider | None,
) -> TokenCounter:
    """Create the configured ingestion token counter.

    ``auto`` first shares the tokenizer from a capable local embedding
    provider. If there is no shareable provider, local deployments use the
    embedding model name while remote deployments must explicitly configure
    ``CHUNK_TOKENIZER_MODEL``. Regex counting is never an implicit fallback.
    """

    if settings.chunk_tokenizer_provider == "regex":
        return RegexTokenCounter()

    try:
        if (
            settings.chunk_tokenizer_provider == "auto"
            and embedding_provider is not None
            and isinstance(embedding_provider, TokenizerSource)
        ):
            tokenizer = await embedding_provider.get_tokenizer()
            return HuggingFaceTokenCounter(tokenizer, model_name=embedding_provider.model)

        model_name = _configured_tokenizer_model(settings)
        return await HuggingFaceTokenCounter.from_model(model_name)
    except (ConfigurationError, ProviderError) as exc:
        if not settings.chunk_tokenizer_allow_regex_fallback:
            raise
        structlog.get_logger(__name__).warning(
            "chunk_tokenizer_regex_fallback",
            configured_provider=settings.chunk_tokenizer_provider,
            configured_model=settings.chunk_tokenizer_model,
            reason=exc.message,
        )
        return RegexTokenCounter()


async def create_llm_token_counter(
    settings: Settings,
    llm_provider: LLMProvider | None,
) -> TokenCounter | None:
    """Create the model-aware counter used for LLM context budgeting.

    An unconfigured LLM returns ``None`` because generation cannot occur. Test-only
    deterministic providers use the deterministic regex counter. Every production
    provider otherwise requires an explicit tokenizer model unless regex counting or
    regex fallback was deliberately enabled.
    """

    if llm_provider is None:
        return None
    if settings.llm_tokenizer_provider == "regex":
        return RegexTokenCounter()
    if settings.app_env == "test" and llm_provider.name == "deterministic":
        return RegexTokenCounter()
    try:
        if not settings.llm_tokenizer_model:
            raise ConfigurationError(
                "LLM_TOKENIZER_MODEL is required for model-accurate context budgeting",
                details={"llm_provider": settings.llm_provider},
            )
        return await HuggingFaceTokenCounter.from_model(settings.llm_tokenizer_model)
    except (ConfigurationError, ProviderError) as exc:
        if not settings.llm_tokenizer_allow_regex_fallback:
            raise
        structlog.get_logger(__name__).warning(
            "llm_tokenizer_regex_fallback",
            configured_provider=settings.llm_tokenizer_provider,
            configured_model=settings.llm_tokenizer_model,
            reason=exc.message,
        )
        return RegexTokenCounter()


def _configured_tokenizer_model(settings: Settings) -> str:
    if settings.chunk_tokenizer_model:
        return settings.chunk_tokenizer_model
    if settings.embedding_provider == "local":
        return settings.embedding_model
    raise ConfigurationError(
        "CHUNK_TOKENIZER_MODEL is required when ingestion uses a remote embedding provider",
        details={"embedding_provider": settings.embedding_provider},
    )


def _load_huggingface_tokenizer(model_name: str) -> Any:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise ConfigurationError(
            "Hugging Face tokenization requires the transformers runtime dependency",
            details={"model": model_name, "dependency": "transformers"},
        ) from exc

    try:
        return transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ProviderError(
            "Unable to load the configured chunk tokenizer",
            details={"model": model_name},
        ) from exc


__all__ = [
    "HuggingFaceTokenCounter",
    "TokenizerSource",
    "create_chunk_token_counter",
    "create_llm_token_counter",
]
