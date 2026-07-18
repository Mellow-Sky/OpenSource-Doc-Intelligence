"""Strict-JSON LLM adapters for evidence sufficiency and citation entailment."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.core.exceptions import ConfigurationError, ProviderError
from app.domain.citations import CitationJudgeDecision
from app.domain.retrieval import EvidenceSufficiency, RetrievalCandidate
from app.providers.base import GenerationResponse, LLMProvider


class LLMEvidenceSufficiencyJudge:
    """Ask a lightweight model whether retrieved excerpts can answer a question."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        prompt_path: Path,
        max_tokens: int,
        timeout_seconds: float,
    ) -> None:
        self._provider = provider
        self._prompt = _read_prompt(prompt_path)
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds

    async def evaluate(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> EvidenceSufficiency:
        started = time.perf_counter()
        evidence = [
            {
                "title": item.document_title,
                "section": " > ".join(item.heading_path),
                "document_type": item.document_type,
                "content": item.content[:6000],
            }
            for item in candidates
        ]
        rendered = _render_prompt(
            self._prompt,
            {
                "question": json.dumps(query, ensure_ascii=False),
                "evidence": json.dumps(evidence, ensure_ascii=False),
            },
        )
        response = await _judge_generate(
            self._provider,
            rendered,
            max_tokens=self._max_tokens,
            timeout_seconds=self._timeout_seconds,
        )
        decoded = _json_object(response.text)
        sufficient = decoded.get("sufficient")
        score = decoded.get("score")
        reason = decoded.get("reason")
        if (
            not isinstance(sufficient, bool)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= float(score) <= 1
            or not isinstance(reason, str)
        ):
            raise ProviderError("Evidence judge returned an invalid JSON schema")
        return EvidenceSufficiency(
            sufficient=sufficient,
            score=float(score),
            reason=reason,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
            model=response.model,
        )


class LLMCitationValidator:
    """Validate one claim/evidence pair without trusting evidence instructions."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        prompt_path: Path,
        max_tokens: int,
        timeout_seconds: float,
    ) -> None:
        self._provider = provider
        self._prompt = _read_prompt(prompt_path)
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds

    async def validate(
        self,
        *,
        claim: str,
        evidence: str,
        title: str,
        section: str,
    ) -> CitationJudgeDecision:
        rendered = _render_prompt(
            self._prompt,
            {
                "claim": json.dumps(claim, ensure_ascii=False),
                "evidence": json.dumps(evidence, ensure_ascii=False),
                "title": json.dumps(title, ensure_ascii=False),
                "section": json.dumps(section, ensure_ascii=False),
            },
        )
        started = time.perf_counter()
        response = await _judge_generate(
            self._provider,
            rendered,
            max_tokens=self._max_tokens,
            timeout_seconds=self._timeout_seconds,
        )
        decoded = _json_object(response.text)
        supported = decoded.get("supported")
        score = decoded.get("score")
        reason = decoded.get("reason")
        if (
            not isinstance(supported, bool)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= float(score) <= 1
            or not isinstance(reason, str)
        ):
            raise ProviderError("Citation judge returned an invalid JSON schema")
        return CitationJudgeDecision(
            supported=supported,
            score=float(score),
            reason=reason,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
            model=response.model,
        )


async def _judge_generate(
    provider: LLMProvider,
    prompt: str,
    *,
    max_tokens: int,
    timeout_seconds: float,
) -> GenerationResponse:
    try:
        async with asyncio.timeout(timeout_seconds):
            return await provider.generate(
                messages=[{"role": "system", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
    except TimeoutError as exc:
        raise ProviderError("LLM judge timed out") from exc


def _json_object(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProviderError("LLM judge returned invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ProviderError("LLM judge returned a non-object response")
    return decoded


def _render_prompt(prompt: str, values: Mapping[str, str]) -> str:
    """Substitute only placeholders present in the template, never inserted data."""
    names = "|".join(re.escape(name) for name in values)
    return re.sub(
        rf"\{{\{{\s*({names})\s*\}}\}}",
        lambda match: values[match.group(1)],
        prompt,
    )


def _read_prompt(path: Path) -> str:
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ConfigurationError(f"Unable to read judge prompt: {path}") from exc
    if not prompt:
        raise ConfigurationError("Judge prompt must not be empty")
    return prompt
