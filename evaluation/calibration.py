"""Supervised no-answer threshold calibration from real evaluation traces."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from evaluation.metrics.answerability import AnswerabilityMetrics, answerability_metrics


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    answerable: bool
    scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ThresholdRecommendation:
    threshold: float | None
    metrics: AnswerabilityMetrics
    sample_count: int


def calibrate_threshold(
    samples: list[CalibrationSample],
    *,
    statistic: str,
    top_k: int = 3,
) -> ThresholdRecommendation:
    """Choose the F1-optimal threshold, with accuracy then lower threshold tie-breaks."""
    values: list[tuple[bool, float]] = []
    for sample in samples:
        if not sample.scores:
            values.append((sample.answerable, float("-inf")))
        elif statistic == "top1":
            values.append((sample.answerable, sample.scores[0]))
        elif statistic == "average":
            values.append((sample.answerable, fmean(sample.scores[:top_k])))
        else:
            msg = "statistic must be top1 or average"
            raise ValueError(msg)
    finite = sorted({score for _, score in values if score != float("-inf")})
    if not finite:
        empty_metrics = answerability_metrics(
            [expected for expected, _ in values],
            [False] * len(values),
        )
        return ThresholdRecommendation(None, empty_metrics, len(values))
    epsilon = max(1e-9, (finite[-1] - finite[0]) * 1e-9)
    candidates = [finite[0] - epsilon, *finite, finite[-1] + epsilon]
    expected = [label for label, _ in values]
    ranked: list[tuple[float, float, float, AnswerabilityMetrics]] = []
    for threshold in candidates:
        predicted = [score >= threshold for _, score in values]
        metrics = answerability_metrics(expected, predicted)
        ranked.append((metrics.f1, metrics.accuracy, -threshold, metrics))
    _, _, negated_threshold, best_metrics = max(ranked, key=lambda item: item[:3])
    return ThresholdRecommendation(
        threshold=-negated_threshold,
        metrics=best_metrics,
        sample_count=len(values),
    )
