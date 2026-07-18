"""Grounded answer generation using versioned prompt files and bounded context."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import ConfigurationError, ProviderError
from app.domain.citations import BuiltContext
from app.providers.base import GenerationResponse, LLMProvider, TokenUsage


@dataclass(frozen=True, slots=True)
class AnswerGeneration:
    """Generated text and accounting values needed by chat persistence."""

    text: str
    model: str
    usage: TokenUsage
    latency_ms: float
    finish_reason: str | None = None


class AnswerService:
    """Render the trusted answer policy around untrusted retrieved evidence."""

    def __init__(
        self,
        *,
        provider: LLMProvider | None,
        prompt_path: Path,
        max_tokens: int,
        timeout_seconds: float,
    ) -> None:
        self._provider = provider
        self._prompt = _load_prompt(prompt_path)
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds

    async def generate(self, question: str, context: BuiltContext) -> AnswerGeneration:
        """Generate one answer solely from the supplied numbered context."""
        if self._provider is None:
            raise ConfigurationError("LLM provider is not configured")
        started = time.perf_counter()
        response = await _generate_with_timeout(
            self._provider,
            messages=self.messages(question, context),
            max_tokens=self._max_tokens,
            timeout_seconds=self._timeout_seconds,
        )
        if not response.text.strip():
            raise ProviderError("Language model returned an empty answer")
        return AnswerGeneration(
            text=response.text.strip(),
            model=response.model,
            usage=response.usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason=response.finish_reason,
        )

    def stream(self, question: str, context: BuiltContext) -> AsyncIterator[str]:
        """Return the provider's SSE-compatible text delta iterator."""
        if self._provider is None:
            raise ConfigurationError("LLM provider is not configured")
        return self._provider.stream(
            messages=self.messages(question, context),
            max_tokens=self._max_tokens,
            temperature=0.0,
        )

    def messages(self, question: str, context: BuiltContext) -> list[dict[str, str]]:
        """Render a policy prompt whose context blocks were already boundary-escaped."""
        values = {"question": question.strip(), "context": context.text}
        rendered = re.sub(
            r"\{\{\s*(question|context)\s*\}\}",
            lambda match: values[match.group(1)],
            self._prompt,
        )
        return [{"role": "system", "content": rendered}]


async def _generate_with_timeout(
    provider: LLMProvider,
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_seconds: float,
) -> GenerationResponse:
    try:
        async with asyncio.timeout(timeout_seconds):
            return await provider.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
    except TimeoutError as exc:
        raise ProviderError("Answer generation timed out") from exc


def _load_prompt(path: Path) -> str:
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ConfigurationError(f"Unable to read answer prompt: {path}") from exc
    if not prompt or "{{ question }}" not in prompt or "{{ context }}" not in prompt:
        raise ConfigurationError("Answer prompt must contain question and context placeholders")
    return prompt
