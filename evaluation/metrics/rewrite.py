"""Metrics comparing retrieval before and after multi-turn query rewriting."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from evaluation.metrics.retrieval import evaluate_retrieval


@dataclass(frozen=True, slots=True)
class RewriteComparison:
    original_recall_at_k: float
    rewritten_recall_at_k: float
    recall_delta: float
    topic_switch_error: bool
    unnecessary_rewrite: bool


def compare_rewrite_retrieval(
    *,
    original_results: Sequence[str],
    rewritten_results: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
    topic_switched: bool,
    query_was_independent: bool,
    query_changed: bool,
) -> RewriteComparison:
    """Compare Recall@K and label common rewrite errors."""
    original = evaluate_retrieval(original_results, relevant_ids, k_values=(k,))
    rewritten = evaluate_retrieval(rewritten_results, relevant_ids, k_values=(k,))
    original_recall = original.recall_at_k[k]
    rewritten_recall = rewritten.recall_at_k[k]
    return RewriteComparison(
        original_recall_at_k=original_recall,
        rewritten_recall_at_k=rewritten_recall,
        recall_delta=rewritten_recall - original_recall,
        topic_switch_error=topic_switched and query_changed,
        unnecessary_rewrite=query_was_independent and query_changed,
    )
