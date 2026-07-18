"""Citation validity, claim coverage, correctness, and completeness metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CitationMetrics:
    """Rule/judge-combined citation metrics for one answer."""

    precision: float
    recall: float
    correctness: float
    completeness: float


def citation_metrics(
    *,
    citation_validity: Sequence[bool],
    support_scores: Sequence[float],
    claim_requires_citation: Sequence[bool],
    claim_supported: Sequence[bool],
) -> CitationMetrics:
    """Calculate citation metrics from explicit validation and claim decisions."""
    if len(claim_requires_citation) != len(claim_supported):
        msg = "claim requirement and support sequences must have equal length"
        raise ValueError(msg)
    if support_scores and len(support_scores) != len(citation_validity):
        msg = "support scores must align with citations"
        raise ValueError(msg)

    precision = sum(citation_validity) / len(citation_validity) if citation_validity else 0.0
    required_indexes = [index for index, required in enumerate(claim_requires_citation) if required]
    covered = sum(claim_supported[index] for index in required_indexes)
    recall = covered / len(required_indexes) if required_indexes else 1.0
    correctness = (
        sum(max(0.0, min(1.0, score)) for score in support_scores) / len(support_scores)
        if support_scores
        else precision
    )
    return CitationMetrics(
        precision=precision,
        recall=recall,
        correctness=correctness,
        completeness=recall,
    )
