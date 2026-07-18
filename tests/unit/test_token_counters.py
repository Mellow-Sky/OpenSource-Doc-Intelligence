"""Model token counters load safely and require explicit fallback policy."""

from __future__ import annotations

import threading
from typing import Any, cast

import pytest

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, ProviderError
from app.ingestion.chunkers import (
    HuggingFaceTokenCounter,
    RegexTokenCounter,
    create_chunk_token_counter,
    create_llm_token_counter,
)
from app.providers.base import EmbeddingProvider, LLMProvider
from app.providers.llm import DeterministicLLMProvider


class _FakeTokenizer:
    def __init__(self) -> None:
        self.add_special_tokens: bool | None = None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.add_special_tokens = add_special_tokens
        return list(range(len(text.split())))


class _TokenizerEmbeddingProvider:
    model = "local-test-model"
    tokenizer = _FakeTokenizer()
    calls = 0

    async def get_tokenizer(self) -> _FakeTokenizer:
        self.calls += 1
        return self.tokenizer


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, app_env="test", **overrides)


def test_huggingface_counter_uses_model_encoding_without_special_tokens() -> None:
    tokenizer = _FakeTokenizer()
    counter = HuggingFaceTokenCounter(tokenizer, model_name="test-model")

    assert counter.count("one two three") == 3
    assert tokenizer.add_special_tokens is False


@pytest.mark.asyncio
async def test_huggingface_tokenizer_loading_runs_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread = threading.get_ident()
    loader_threads: list[int] = []

    def fake_loader(_model_name: str) -> _FakeTokenizer:
        loader_threads.append(threading.get_ident())
        return _FakeTokenizer()

    monkeypatch.setattr(
        "app.ingestion.chunkers.tokenizers._load_huggingface_tokenizer",
        fake_loader,
    )

    counter = await HuggingFaceTokenCounter.from_model("test-model")

    assert counter.count("one two") == 2
    assert loader_threads and loader_threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_auto_counter_reuses_local_embedding_provider_tokenizer() -> None:
    provider = _TokenizerEmbeddingProvider()

    counter = await create_chunk_token_counter(
        _settings(chunk_tokenizer_provider="auto"),
        cast(EmbeddingProvider, provider),
    )

    assert isinstance(counter, HuggingFaceTokenCounter)
    assert counter.model_name == "local-test-model"
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_explicit_regex_mode_never_loads_provider_tokenizer() -> None:
    provider = _TokenizerEmbeddingProvider()
    provider.calls = 0

    counter = await create_chunk_token_counter(
        _settings(chunk_tokenizer_provider="regex"),
        cast(EmbeddingProvider, provider),
    )

    assert isinstance(counter, RegexTokenCounter)
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_remote_embedding_requires_explicit_tokenizer_model() -> None:
    settings = _settings(
        embedding_provider="remote",
        chunk_tokenizer_provider="auto",
        chunk_tokenizer_model=None,
    )

    with pytest.raises(ConfigurationError, match="CHUNK_TOKENIZER_MODEL"):
        await create_chunk_token_counter(settings, None)


@pytest.mark.asyncio
async def test_huggingface_mode_loads_explicit_remote_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_models: list[str] = []

    async def fake_from_model(
        cls: type[HuggingFaceTokenCounter], model_name: str
    ) -> HuggingFaceTokenCounter:
        loaded_models.append(model_name)
        return cls(_FakeTokenizer(), model_name=model_name)

    monkeypatch.setattr(HuggingFaceTokenCounter, "from_model", classmethod(fake_from_model))
    settings = _settings(
        embedding_provider="remote",
        chunk_tokenizer_provider="huggingface",
        chunk_tokenizer_model="org/remote-tokenizer",
    )

    counter = await create_chunk_token_counter(settings, None)

    assert isinstance(counter, HuggingFaceTokenCounter)
    assert loaded_models == ["org/remote-tokenizer"]


@pytest.mark.asyncio
async def test_regex_fallback_must_be_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_load(
        _cls: type[HuggingFaceTokenCounter], _model_name: str
    ) -> HuggingFaceTokenCounter:
        raise ProviderError("tokenizer unavailable")

    monkeypatch.setattr(HuggingFaceTokenCounter, "from_model", classmethod(fail_load))
    strict = _settings(chunk_tokenizer_provider="huggingface")
    fallback = _settings(
        chunk_tokenizer_provider="huggingface",
        chunk_tokenizer_allow_regex_fallback=True,
    )

    with pytest.raises(ProviderError, match="tokenizer unavailable"):
        await create_chunk_token_counter(strict, None)
    assert isinstance(await create_chunk_token_counter(fallback, None), RegexTokenCounter)


@pytest.mark.asyncio
async def test_production_llm_context_requires_an_explicit_model_tokenizer() -> None:
    provider = cast(LLMProvider, type("Provider", (), {"name": "remote"})())
    settings = Settings(
        _env_file=None,
        app_env="production",
        llm_tokenizer_model=None,
    )

    with pytest.raises(ConfigurationError, match="LLM_TOKENIZER_MODEL"):
        await create_llm_token_counter(settings, provider)


@pytest.mark.asyncio
async def test_test_llm_and_explicit_regex_context_counters_do_not_load_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def forbidden_load(
        _cls: type[HuggingFaceTokenCounter], _model_name: str
    ) -> HuggingFaceTokenCounter:
        raise AssertionError("model tokenizer should not load")

    monkeypatch.setattr(HuggingFaceTokenCounter, "from_model", classmethod(forbidden_load))

    deterministic = await create_llm_token_counter(
        _settings(llm_provider="deterministic"),
        DeterministicLLMProvider(),
    )
    explicit = await create_llm_token_counter(
        Settings(
            _env_file=None,
            app_env="production",
            llm_tokenizer_provider="regex",
        ),
        cast(LLMProvider, type("Provider", (), {"name": "remote"})()),
    )

    assert isinstance(deterministic, RegexTokenCounter)
    assert isinstance(explicit, RegexTokenCounter)
