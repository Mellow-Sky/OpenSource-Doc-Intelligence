"""Azure OpenAI providers use deployment-scoped URLs and API-key auth."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.providers.azure_openai import azure_deployment_url
from app.providers.embedding import AzureOpenAIEmbeddingProvider
from app.providers.factory import (
    create_embedding_provider,
    create_judge_provider,
    create_llm_provider,
)
from app.providers.llm import AzureOpenAILLMProvider


@pytest.mark.asyncio
async def test_azure_llm_uses_deployment_url_api_key_and_no_model_payload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-2024-11-20",
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AzureOpenAILLMProvider(
        endpoint="https://rag-resource.openai.azure.com/",
        api_key=SecretStr("azure-secret"),
        api_version="2024-10-21",
        deployment="answer deployment",
        model="gpt-4o",
        timeout_seconds=1,
        max_retries=0,
        max_concurrency=1,
        client=client,
    )

    response = await provider.generate(
        messages=[{"role": "user", "content": "question"}],
        max_tokens=50,
        response_format={"type": "json_object"},
    )

    request = requests[0]
    payload = json.loads(request.content)
    assert request.url.path == ("/openai/deployments/answer deployment/chat/completions")
    assert request.url.params["api-version"] == "2024-10-21"
    assert request.headers["api-key"] == "azure-secret"
    assert "authorization" not in request.headers
    assert "model" not in payload
    assert payload["response_format"] == {"type": "json_object"}
    assert response.text == "answer"
    assert response.usage.total_tokens == 6
    await client.aclose()


@pytest.mark.asyncio
async def test_azure_embedding_uses_deployment_url_and_input_only_payload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [{"index": 0, "embedding": [0.25, 0.75]}],
                "usage": {"prompt_tokens": 2},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AzureOpenAIEmbeddingProvider(
        endpoint="https://rag-resource.openai.azure.com",
        api_key=SecretStr("embedding-secret"),
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

    response = await provider.embed(["document"])

    request = requests[0]
    assert request.url.path == "/openai/deployments/embedding-prod/embeddings"
    assert request.url.params["api-version"] == "2024-10-21"
    assert request.headers["api-key"] == "embedding-secret"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {"input": ["document"]}
    assert response.model == "embedding-prod"
    assert response.vectors == [[0.25, 0.75]]
    await client.aclose()


@pytest.mark.asyncio
async def test_azure_factories_select_role_specific_adapters() -> None:
    llm = create_llm_provider(
        Settings(
            _env_file=None,
            llm_provider="azure",
            llm_base_url="https://answer.openai.azure.com",
            llm_api_key="answer-key",
            llm_api_version="2024-10-21",
            llm_deployment="answer-prod",
        )
    )
    judge = create_judge_provider(
        Settings(
            _env_file=None,
            judge_provider="azure_openai",
            judge_base_url="https://judge.openai.azure.com",
            judge_api_key="judge-key",
            judge_api_version="2024-10-21",
            judge_deployment="judge-prod",
        )
    )
    embedding = create_embedding_provider(
        Settings(
            _env_file=None,
            embedding_provider="azure",
            embedding_base_url="https://embedding.openai.azure.com",
            embedding_api_key="embedding-key",
            embedding_api_version="2024-10-21",
            embedding_deployment="embedding-prod",
        )
    )

    assert isinstance(llm, AzureOpenAILLMProvider)
    assert isinstance(judge, AzureOpenAILLMProvider)
    assert isinstance(embedding, AzureOpenAIEmbeddingProvider)
    assert llm.model == "answer-prod"
    assert judge.model == "judge-prod"
    assert embedding.model == "embedding-prod"
    await llm.close()
    await judge.close()
    await embedding.close()


@pytest.mark.parametrize(
    ("override", "missing_setting"),
    [
        ({"llm_base_url": None}, "LLM_BASE_URL"),
        ({"llm_api_key": None}, "LLM_API_KEY"),
        ({"llm_api_version": None}, "LLM_API_VERSION"),
        ({"llm_deployment": None}, "LLM_DEPLOYMENT"),
    ],
)
def test_azure_llm_factory_rejects_incomplete_configuration(
    override: dict[str, object],
    missing_setting: str,
) -> None:
    values: dict[str, object] = {
        "_env_file": None,
        "llm_provider": "azure",
        "llm_base_url": "https://answer.openai.azure.com",
        "llm_api_key": "answer-key",
        "llm_api_version": "2024-10-21",
        "llm_deployment": "answer-prod",
    }
    values.update(override)
    with pytest.raises(ConfigurationError, match=missing_setting):
        create_llm_provider(Settings(**values))


def test_azure_url_builder_encodes_deployment_and_rejects_non_root_endpoint() -> None:
    url = azure_deployment_url(
        endpoint="https://resource.openai.azure.com",
        deployment="prod/chat #1",
        resource="chat/completions",
        api_version="2024-10-21-preview",
    )
    assert url == (
        "https://resource.openai.azure.com/openai/deployments/prod%2Fchat%20%231/"
        "chat/completions?api-version=2024-10-21-preview"
    )

    with pytest.raises(ConfigurationError, match="resource root"):
        azure_deployment_url(
            endpoint="https://resource.openai.azure.com/openai/v1",
            deployment="prod",
            resource="embeddings",
            api_version="2024-10-21",
        )
