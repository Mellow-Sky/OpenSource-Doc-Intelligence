"""Model adapters validate ordering, retry behavior, and test-only fallbacks."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.providers.embedding import OpenAICompatibleEmbeddingProvider
from app.providers.embedding.local import LocalEmbeddingProvider
from app.providers.factory import create_embedding_provider, create_reranker_provider
from app.providers.reranker import RemoteRerankerProvider
from app.providers.reranker.local import LocalRerankerProvider


class _EncodedRows:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return self._rows


class _LocalTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))


class _LocalSentenceModel:
    tokenizer = _LocalTokenizer()

    def encode(self, texts: list[str], **_kwargs: object) -> _EncodedRows:
        return _EncodedRows([[float(index), 1.0] for index, _text in enumerate(texts)])


@pytest.mark.asyncio
async def test_openai_embedding_batches_and_restores_provider_indexes() -> None:
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        inputs = payload["input"]
        calls.append(inputs)
        data = [
            {"index": index, "embedding": [float(len(text)), float(index)]}
            for index, text in reversed(list(enumerate(inputs)))
        ]
        return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 3}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://models.example/v1",
        api_key=None,
        model="embedding-test",
        dimension=2,
        batch_size=2,
        timeout_seconds=1,
        max_retries=0,
        max_concurrency=2,
        client=client,
    )

    response = await provider.embed(["a", "bbbb", "cc"])

    assert calls == [["a", "bbbb"], ["cc"]]
    assert response.vectors == [[1.0, 0.0], [4.0, 1.0], [2.0, 0.0]]
    assert response.usage.prompt_tokens == 6
    await client.aclose()


@pytest.mark.asyncio
async def test_embedding_retries_temporary_rate_limit() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://models.example/v1",
        api_key=None,
        model="embedding-test",
        dimension=1,
        batch_size=8,
        timeout_seconds=1,
        max_retries=1,
        max_concurrency=1,
        client=client,
    )

    assert (await provider.embed(["retry"])).vectors == [[1.0]]
    assert attempts == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_local_embedding_reports_tokenizer_input_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LocalEmbeddingProvider(model="local-test", dimension=2, batch_size=8)
    model = _LocalSentenceModel()

    async def loaded_model() -> _LocalSentenceModel:
        return model

    monkeypatch.setattr(provider, "_ensure_loaded", loaded_model)

    response = await provider.embed(["one two", "three four five"])

    assert response.vectors == [[0.0, 1.0], [1.0, 1.0]]
    assert response.usage.prompt_tokens == 5
    assert response.usage.total_tokens == 5


@pytest.mark.asyncio
async def test_local_embedding_honors_configured_inference_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LocalEmbeddingProvider(
        model="local-test",
        dimension=2,
        batch_size=8,
        max_concurrency=2,
    )
    model = _LocalSentenceModel()
    active = 0
    peak = 0

    async def loaded_model() -> _LocalSentenceModel:
        return model

    async def observed_to_thread(
        function: Callable[..., object], *args: object, **kwargs: object
    ) -> object:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        try:
            return function(*args, **kwargs)
        finally:
            active -= 1

    monkeypatch.setattr(provider, "_ensure_loaded", loaded_model)
    monkeypatch.setattr(asyncio, "to_thread", observed_to_thread)

    await asyncio.gather(*(provider.embed([f"input {index}"]) for index in range(6)))

    assert peak == 2


@pytest.mark.asyncio
async def test_remote_reranker_restores_scores_by_index() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.1},
                    {"index": 0, "relevance_score": 0.9},
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = RemoteRerankerProvider(
        base_url="https://models.example/v1",
        api_key=None,
        model="reranker-test",
        batch_size=8,
        timeout_seconds=1,
        max_retries=0,
        max_concurrency=1,
        client=client,
    )

    assert (await provider.rerank("query", ["first", "second"])).scores == [0.9, 0.1]
    await client.aclose()


@pytest.mark.asyncio
async def test_local_reranker_honors_configured_inference_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model:
        def compute_score(self, pairs: list[list[str]], **_kwargs: object) -> list[float]:
            return [0.5] * len(pairs)

    provider = LocalRerankerProvider(
        model="local-reranker",
        batch_size=8,
        max_concurrency=1,
    )
    active = 0
    peak = 0

    async def loaded_model() -> Model:
        return Model()

    async def observed_to_thread(
        function: Callable[..., object], *args: object, **kwargs: object
    ) -> object:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        try:
            return function(*args, **kwargs)
        finally:
            active -= 1

    monkeypatch.setattr(provider, "_ensure_loaded", loaded_model)
    monkeypatch.setattr(asyncio, "to_thread", observed_to_thread)

    await asyncio.gather(*(provider.rerank("query", [str(index)]) for index in range(4)))

    assert peak == 1


def test_deterministic_providers_require_test_environment() -> None:
    test_settings = Settings(
        _env_file=None,
        app_env="test",
        embedding_provider="deterministic",
        reranker_provider="deterministic",
    )
    assert create_embedding_provider(test_settings).name == "deterministic"
    assert create_reranker_provider(test_settings).name == "deterministic"

    production = Settings(
        _env_file=None,
        app_env="production",
        embedding_provider="deterministic",
    )
    with pytest.raises(ConfigurationError, match="Unsupported EMBEDDING_PROVIDER"):
        create_embedding_provider(production)
