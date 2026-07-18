"""Remote readiness probes follow each provider's actual HTTP contract."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from app.core.exceptions import ProviderError
from app.providers.embedding import (
    AzureOpenAIEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from app.providers.llm import AzureOpenAILLMProvider, OpenAICompatibleLLMProvider
from app.providers.reranker import RemoteRerankerProvider


@pytest.mark.asyncio
async def test_openai_llm_healthcheck_validates_configured_catalog_model() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": "configured-model"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleLLMProvider(
        base_url="https://models.example/v1",
        api_key=None,
        model="configured-model",
        timeout_seconds=1,
        max_retries=0,
        max_concurrency=1,
        client=client,
    )

    await provider.healthcheck()

    assert [(request.method, request.url.path) for request in requests] == [("GET", "/v1/models")]
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_embedding_healthcheck_rejects_missing_catalog_model() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"data": [{"id": "another-model"}]})
        )
    )
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://models.example/v1",
        api_key=None,
        model="configured-embedding",
        dimension=2,
        batch_size=8,
        timeout_seconds=1,
        max_retries=0,
        max_concurrency=1,
        client=client,
    )

    with pytest.raises(ProviderError, match="absent"):
        await provider.healthcheck()

    await client.aclose()


@pytest.mark.asyncio
async def test_azure_llm_healthcheck_probes_the_configured_deployment() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AzureOpenAILLMProvider(
        endpoint="https://rag-resource.openai.azure.com",
        api_key=SecretStr("secret"),
        api_version="2024-10-21",
        deployment="answer-prod",
        model=None,
        timeout_seconds=1,
        max_retries=0,
        max_concurrency=1,
        client=client,
    )

    await provider.healthcheck()

    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/openai/deployments/answer-prod/chat/completions"
    assert json.loads(request.content) == {
        "messages": [{"role": "user", "content": "healthcheck"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    await client.aclose()


@pytest.mark.asyncio
async def test_azure_embedding_healthcheck_validates_deployment_and_dimension() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.25, 0.75]}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AzureOpenAIEmbeddingProvider(
        endpoint="https://rag-resource.openai.azure.com",
        api_key=SecretStr("secret"),
        api_version="2024-10-21",
        deployment="embedding-prod",
        model=None,
        dimension=2,
        batch_size=8,
        timeout_seconds=1,
        max_retries=0,
        max_concurrency=1,
        client=client,
    )

    await provider.healthcheck()

    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/openai/deployments/embedding-prod/embeddings"
    assert json.loads(request.content) == {"input": ["healthcheck"]}
    await client.aclose()


@pytest.mark.asyncio
async def test_remote_reranker_healthcheck_uses_rerank_not_models() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": 1.0}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = RemoteRerankerProvider(
        base_url="https://reranker.example/v1",
        api_key=None,
        model="reranker-prod",
        batch_size=8,
        timeout_seconds=1,
        max_retries=0,
        max_concurrency=1,
        client=client,
    )

    await provider.healthcheck()

    request = requests[0]
    payload = json.loads(request.content)
    assert (request.method, request.url.path) == ("POST", "/v1/rerank")
    assert payload["model"] == "reranker-prod"
    assert payload["documents"] == ["healthcheck"]
    await client.aclose()


@pytest.mark.asyncio
async def test_remote_reranker_can_use_a_dedicated_json_health_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = RemoteRerankerProvider(
        base_url="https://reranker.example/v1",
        api_key=None,
        model="reranker-prod",
        batch_size=8,
        timeout_seconds=1,
        max_retries=0,
        max_concurrency=1,
        client=client,
        healthcheck_mode="endpoint",
        healthcheck_resource="healthz",
    )

    await provider.healthcheck()

    assert [(request.method, request.url.path) for request in requests] == [("GET", "/v1/healthz")]
    await client.aclose()
