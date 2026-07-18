"""Chunk-level retrieval metrics with both hit and relevant-set coverage recall."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

DEFAULT_K_VALUES = (1, 3, 5, 10, 20)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Per-query retrieval measurements."""

    recall_at_k: dict[int, float]
    relevant_set_coverage_at_k: dict[int, float]
    reciprocal_rank: float


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Return the reciprocal rank of the first relevant chunk, or zero."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RetrievalMetrics:
    """Calculate binary Recall@K, relevant-set coverage, and reciprocal rank."""
    relevant = set(relevant_ids)
    recall: dict[int, float] = {}
    coverage: dict[int, float] = {}
    for k in k_values:
        if k <= 0:
            msg = "k values must be positive"
            raise ValueError(msg)
        matches = relevant.intersection(retrieved_ids[:k])
        recall[k] = float(bool(matches)) if relevant else 0.0
        coverage[k] = len(matches) / len(relevant) if relevant else 0.0
    return RetrievalMetrics(
        recall_at_k=recall,
        relevant_set_coverage_at_k=coverage,
        reciprocal_rank=reciprocal_rank(retrieved_ids, relevant),
    )
