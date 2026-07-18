"""OpenAI-compatible generation and streaming are validated without paid APIs."""

from __future__ import annotations

import json

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.providers.factory import create_llm_provider
from app.providers.llm import OpenAICompatibleLLMProvider


@pytest.mark.asyncio
async def test_llm_provider_parses_completion_and_usage() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "served-model",
                "choices": [{"message": {"content": "grounded answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
        )

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

    response = await provider.generate(
        messages=[{"role": "user", "content": "question"}],
        max_tokens=100,
        response_format={"type": "json_object"},
    )

    assert captured["model"] == "configured-model"
    assert captured["response_format"] == {"type": "json_object"}
    assert response.text == "grounded answer"
    assert response.model == "served-model"
    assert response.usage.total_tokens == 16
    await client.aclose()


@pytest.mark.asyncio
async def test_llm_provider_streams_sse_text_deltas() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=(
                'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

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

    deltas = [
        delta
        async for delta in provider.stream(
            messages=[{"role": "user", "content": "question"}],
            max_tokens=100,
        )
    ]

    assert deltas == ["hello ", "world"]
    await client.aclose()


def test_llm_factory_allows_deterministic_only_in_test() -> None:
    provider = create_llm_provider(
        Settings(_env_file=None, app_env="test", llm_provider="deterministic")
    )
    assert provider.name == "deterministic"

    with pytest.raises(ConfigurationError, match="LLM_BASE_URL"):
        create_llm_provider(Settings(_env_file=None))
