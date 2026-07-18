"""Query rewriting handles reference, omission, topic switch, and safe fallback."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from app.core.exceptions import ProviderError
from app.domain.evaluation import ConversationTurn
from app.providers.base import GenerationResponse, LLMProvider, TokenUsage
from app.services.query_rewrite_service import QueryRewriteService

PROMPT = Path(__file__).parents[2] / "prompts" / "query_rewrite.md"


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, text: str = "", *, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[Sequence[dict[str, str]]] = []

    async def healthcheck(self) -> None:
        return None

    async def generate(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        self.calls.append(messages)
        if self.fail:
            raise ProviderError("offline")
        return GenerationResponse(
            text=self.text,
            model="fake",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )

    def stream(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        raise NotImplementedError


def _service(provider: LLMProvider | None, *, enabled: bool = True) -> QueryRewriteService:
    return QueryRewriteService(
        provider=provider,
        prompt_path=PROMPT,
        enabled=enabled,
        history_turns=3,
        max_tokens=128,
        timeout_seconds=1,
        max_query_length=4000,
    )


@pytest.mark.asyncio
async def test_pronoun_is_rewritten_to_standalone_question() -> None:
    provider = FakeLLM('{"rewritten_query":"Kubernetes Deployment 滚动更新失败后如何回滚?"}')
    history = [
        ConversationTurn(role="user", content="Deployment 如何滚动更新?"),
        ConversationTurn(role="assistant", content="可以配置 rolling update。"),
    ]

    result = await _service(provider).rewrite("它失败后怎么回滚?", history)

    assert result.rewritten_query == "Kubernetes Deployment 滚动更新失败后如何回滚?"
    assert result.changed is True
    assert result.usage.total_tokens == 15
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_omitted_subject_after_failure_is_recovered_from_recent_history() -> None:
    provider = FakeLLM('{"rewritten_query":"Kubernetes Deployment rollout 失败后如何回滚?"}')
    history = [
        ConversationTurn(role="user", content="如何查看 Kubernetes Deployment rollout 状态?"),
        ConversationTurn(role="assistant", content="可以查看 rollout status。"),
    ]

    result = await _service(provider).rewrite("失败后怎么回滚?", history)

    assert result.rewritten_query == "Kubernetes Deployment rollout 失败后如何回滚?"
    assert result.changed is True
    assert result.reason == "rewritten"


@pytest.mark.asyncio
async def test_topic_switch_model_must_preserve_an_explicit_new_subject() -> None:
    provider = FakeLLM('{"rewritten_query":"然后, Service 有哪些类型?"}')
    history = [ConversationTurn(role="user", content="Deployment 如何回滚?")]

    result = await _service(provider).rewrite("然后, Service 有哪些类型?", history)

    assert result.rewritten_query == "然后, Service 有哪些类型?"
    assert result.changed is False
    assert result.reason == "model_kept_original"


@pytest.mark.asyncio
async def test_independent_topic_switch_is_not_over_rewritten() -> None:
    provider = FakeLLM('{"rewritten_query":"should not be called"}')
    history = [ConversationTurn(role="user", content="Deployment 如何更新?")]

    result = await _service(provider).rewrite("Service 有哪些类型?", history)

    assert result.rewritten_query == "Service 有哪些类型?"
    assert result.reason == "independent_query"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_provider_failure_falls_back_to_original_query() -> None:
    result = await _service(FakeLLM(fail=True)).rewrite(
        "它怎么回滚?",
        [ConversationTurn(role="user", content="Deployment 如何更新?")],
    )

    assert result.rewritten_query == "它怎么回滚?"
    assert result.degraded is True
    assert result.reason == "rewrite_failed"


@pytest.mark.asyncio
async def test_rewrite_cannot_invent_a_version() -> None:
    provider = FakeLLM('{"rewritten_query":"Kubernetes v1.99 Deployment 如何回滚?"}')

    result = await _service(provider).rewrite(
        "它怎么回滚?",
        [ConversationTurn(role="user", content="Deployment 如何更新?")],
    )

    assert result.rewritten_query == "它怎么回滚?"
    assert result.reason == "unsafe_rewrite"
