"""Deterministic language-model adapter for offline tests only."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.providers.base import GenerationResponse, LLMProvider, TokenUsage


class DeterministicLLMProvider(LLMProvider):
    """Echo the last user message through deterministic generate/stream paths."""

    @property
    def name(self) -> str:
        return "deterministic"

    @property
    def model(self) -> str:
        return "deterministic-echo"

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
        text = _last_user_content(messages)
        return GenerationResponse(
            text=text,
            model=self.model,
            usage=TokenUsage(prompt_tokens=sum(len(item["content"].split()) for item in messages)),
            finish_reason="stop",
        )

    def stream(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        return _single_delta(_last_user_content(messages))


def _last_user_content(messages: Sequence[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


async def _single_delta(text: str) -> AsyncIterator[str]:
    yield text
