"""Conservative multi-turn query rewriting with timeout and safe fallback."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.core.exceptions import ProviderError
from app.domain.evaluation import ConversationTurn
from app.providers.base import LLMProvider, TokenUsage

_CONTEXT_DEPENDENT = re.compile(
    r"(?:\b(?:it|its|this|that|they|them|those|these|there|then)\b|"
    r"它|这个|那个|该(?:对象|资源|字段|版本)?|上述|前面|刚才|其中|呢|"
    r"^(?:如果|那么|然后|失败后|成功后|再|又|还有))",
    re.IGNORECASE,
)
_TECHNICAL_FACT = re.compile(
    r"(?<![\w.-])(?:v?\d+\.\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9.-]*/v\d+"
    r"(?:(?:alpha|beta)\d+)?)(?![\w.-])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """Observable rewrite result retained with the user message."""

    original_query: str
    rewritten_query: str
    latency_ms: float
    usage: TokenUsage = field(default_factory=TokenUsage)
    changed: bool = False
    degraded: bool = False
    reason: str = "independent_query"
    model: str | None = None


class QueryRewriteService:
    """Rewrite only context-dependent questions and never block retrieval on failure."""

    def __init__(
        self,
        *,
        provider: LLMProvider | None,
        prompt_path: Path,
        enabled: bool,
        history_turns: int,
        max_tokens: int,
        timeout_seconds: float,
        max_query_length: int,
    ) -> None:
        self._provider = provider
        self._prompt = _load_prompt(prompt_path)
        self._enabled = enabled
        self._history_turns = history_turns
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds
        self._max_query_length = max_query_length

    async def rewrite(
        self,
        query: str,
        history: Sequence[ConversationTurn],
    ) -> RewriteResult:
        """Return an independent query, falling back byte-for-byte on any provider issue."""
        started = time.perf_counter()
        original = query.strip()
        if not self._enabled:
            return self._unchanged(original, started, "disabled")
        relevant_history = [turn for turn in history if turn.role in {"user", "assistant"}]
        if not relevant_history:
            return self._unchanged(original, started, "no_history")
        if not _CONTEXT_DEPENDENT.search(original):
            return self._unchanged(original, started, "independent_query")
        if self._provider is None:
            return self._unchanged(original, started, "provider_unavailable", degraded=True)

        bounded_history = relevant_history[-(self._history_turns * 2) :]
        payload = {
            "conversation": [
                {"role": turn.role, "content": turn.content[: self._max_query_length]}
                for turn in bounded_history
            ],
            "current_query": original,
        }
        messages = [
            {"role": "system", "content": self._prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        try:
            async with asyncio.timeout(self._timeout_seconds):
                response = await self._provider.generate(
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
            rewritten = _parse_rewrite(response.text)
            if not _safe_rewrite(original, bounded_history, rewritten, self._max_query_length):
                return self._unchanged(
                    original,
                    started,
                    "unsafe_rewrite",
                    degraded=True,
                    usage=response.usage,
                )
            return RewriteResult(
                original_query=original,
                rewritten_query=rewritten,
                latency_ms=(time.perf_counter() - started) * 1000,
                usage=response.usage,
                changed=rewritten != original,
                reason="rewritten" if rewritten != original else "model_kept_original",
                model=response.model,
            )
        except (ProviderError, TimeoutError, ValueError):
            # Query rewriting is an optional recall enhancement. Retrieval must still
            # receive the exact user query if the provider or response is unusable.
            return self._unchanged(original, started, "rewrite_failed", degraded=True)

    @staticmethod
    def _unchanged(
        query: str,
        started: float,
        reason: str,
        *,
        degraded: bool = False,
        usage: TokenUsage | None = None,
    ) -> RewriteResult:
        return RewriteResult(
            original_query=query,
            rewritten_query=query,
            latency_ms=(time.perf_counter() - started) * 1000,
            usage=usage or TokenUsage(),
            degraded=degraded,
            reason=reason,
        )


def _load_prompt(path: Path) -> str:
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"Unable to read query rewrite prompt: {path}") from exc
    if not prompt:
        raise ValueError("Query rewrite prompt must not be empty")
    return prompt


def _parse_rewrite(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict):
        value = decoded.get("rewritten_query")
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("rewrite JSON lacks rewritten_query")
    if "\n" not in stripped and stripped:
        return stripped
    raise ValueError("rewrite response is neither JSON nor a single query")


def _safe_rewrite(
    original: str,
    history: Sequence[ConversationTurn],
    rewritten: str,
    max_query_length: int,
) -> bool:
    if not rewritten or len(rewritten) > max_query_length:
        return False
    if len(rewritten) > max(len(original) * 6, 512):
        return False
    source = " ".join([*(turn.content for turn in history), original])
    source_facts = {value.casefold() for value in _TECHNICAL_FACT.findall(source)}
    rewritten_facts = {value.casefold() for value in _TECHNICAL_FACT.findall(rewritten)}
    return rewritten_facts.issubset(source_facts)
