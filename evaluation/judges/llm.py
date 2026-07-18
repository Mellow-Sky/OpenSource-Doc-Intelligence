"""Strict-JSON LLM judge kept separate from the evaluated answer provider."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import ConfigurationError, ProviderError
from app.domain.evaluation import EvaluationCase, JudgeScores
from app.providers.base import LLMProvider, TokenUsage
from app.services.pricing_service import PricingCatalog


@dataclass(frozen=True, slots=True)
class EvaluationJudgeResult:
    scores: JudgeScores
    provider: str
    model: str
    usage: TokenUsage
    estimated_cost_usd: float | None
    latency_ms: float


class LLMEvaluationJudge:
    """Score correctness, completeness, relevance, and groundedness."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        prompt_path: Path,
        pricing_catalog: PricingCatalog,
        max_tokens: int = 512,
        timeout_seconds: float = 60.0,
    ) -> None:
        try:
            self._prompt = prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigurationError(
                f"Unable to read evaluation judge prompt: {prompt_path}"
            ) from exc
        if not self._prompt:
            raise ConfigurationError("Evaluation judge prompt must not be empty")
        self._provider = provider
        self._pricing = pricing_catalog
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds

    async def evaluate(
        self,
        *,
        case: EvaluationCase,
        generated_answer: str,
        evidence: list[dict[str, str]],
    ) -> EvaluationJudgeResult:
        """Judge one answer with all required inputs and a bounded timeout."""
        values = {
            "question": json.dumps(case.question, ensure_ascii=False),
            "reference_answer": json.dumps(case.reference_answer, ensure_ascii=False),
            "generated_answer": json.dumps(generated_answer, ensure_ascii=False),
            "evidence": json.dumps(evidence, ensure_ascii=False),
        }
        rendered = re.sub(
            r"\{\{\s*(question|reference_answer|generated_answer|evidence)\s*\}\}",
            lambda match: values[match.group(1)],
            self._prompt,
        )
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                response = await self._provider.generate(
                    messages=[{"role": "system", "content": rendered}],
                    max_tokens=self._max_tokens,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
        except TimeoutError as exc:
            raise ProviderError("Evaluation judge timed out") from exc
        scores = _parse_scores(response.text)
        cost = self._pricing.estimate(
            provider=self._provider.name,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        return EvaluationJudgeResult(
            scores=scores,
            provider=self._provider.name,
            model=response.model,
            usage=response.usage,
            estimated_cost_usd=float(cost) if cost is not None else None,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def _parse_scores(text: str) -> JudgeScores:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    try:
        decoded = json.loads(stripped)
        return JudgeScores.model_validate(decoded)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProviderError("Evaluation judge returned invalid JSON") from exc
