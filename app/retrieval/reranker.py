"""Cross-encoder reranking with a deterministic fusion-order degradation path."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass

from app.core.exceptions import ProviderError
from app.domain.retrieval import RetrievalCandidate
from app.providers.base import RerankerProvider


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    """Ranked candidates and observable provider degradation state."""

    candidates: list[RetrievalCandidate]
    latency_ms: float
    all_candidates: list[RetrievalCandidate] | None = None
    degraded: bool = False
    reason: str | None = None


async def rerank_candidates(
    *,
    query: str,
    candidates: list[RetrievalCandidate],
    provider: RerankerProvider | None,
    top_n: int,
    timeout_seconds: float,
    enabled: bool = True,
    score_threshold: float | None = None,
) -> RerankOutcome:
    """Score candidates in one provider call or preserve fused order on failure."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    started = time.perf_counter()
    if not candidates:
        return RerankOutcome(candidates=[], latency_ms=0.0, all_candidates=[])
    if not enabled:
        return _fallback(candidates, top_n, started, reason="disabled", degraded=False)
    if provider is None:
        return _fallback(candidates, top_n, started, reason="provider_unavailable")
    try:
        async with asyncio.timeout(timeout_seconds):
            response = await provider.rerank(query, [item.content for item in candidates])
        if len(response.scores) != len(candidates) or not all(
            math.isfinite(score) for score in response.scores
        ):
            raise ProviderError("Reranker returned invalid scores")
    except (ProviderError, TimeoutError) as exc:
        return _fallback(candidates, top_n, started, reason=type(exc).__name__)

    scored = [
        candidate.model_copy(update={"rerank_score": score})
        for candidate, score in zip(candidates, response.scores, strict=True)
    ]
    scored.sort(
        key=lambda item: (
            -(item.rerank_score if item.rerank_score is not None else float("-inf")),
            item.fused_rank if item.fused_rank is not None else 10**9,
            str(item.chunk_id),
        )
    )
    ranked = [
        item.model_copy(update={"rerank_rank": rank}) for rank, item in enumerate(scored, start=1)
    ]
    selected = (
        [
            item
            for item in ranked
            if item.rerank_score is not None and item.rerank_score >= score_threshold
        ]
        if score_threshold is not None
        else ranked
    )
    return RerankOutcome(
        candidates=selected[:top_n],
        latency_ms=(time.perf_counter() - started) * 1000,
        all_candidates=ranked,
    )


def _fallback(
    candidates: list[RetrievalCandidate],
    top_n: int,
    started: float,
    *,
    reason: str,
    degraded: bool = True,
) -> RerankOutcome:
    """Return fusion order without fabricating cross-encoder provenance.

    Fusion and reranker scores have unrelated scales (RRF is commonly around
    1 / 60 while cross-encoder scores are provider specific).  Keeping the
    reranker fields empty lets downstream confidence gates select the correct
    score family and prevents calibrated reranker thresholds from rejecting a
    healthy fusion-only fallback.
    """
    ordered = [
        item.model_copy(update={"rerank_rank": None, "rerank_score": None})
        for item in sorted(
            candidates,
            key=lambda item: (
                item.fused_rank if item.fused_rank is not None else 10**9,
                str(item.chunk_id),
            ),
        )
    ]
    return RerankOutcome(
        candidates=ordered[:top_n],
        latency_ms=(time.perf_counter() - started) * 1000,
        all_candidates=ordered,
        degraded=degraded,
        reason=reason,
    )
