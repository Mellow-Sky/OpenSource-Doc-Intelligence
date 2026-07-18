"""Binary answerability/no-answer classification metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnswerabilityMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    false_answers: int
    false_refusals: int


def answerability_metrics(
    expected_answerable: Sequence[bool],
    predicted_answerable: Sequence[bool],
) -> AnswerabilityMetrics:
    """Compute answerability metrics where answerable is the positive class."""
    if len(expected_answerable) != len(predicted_answerable):
        msg = "expected and predicted labels must have equal length"
        raise ValueError(msg)
    tp = sum(
        expected and predicted
        for expected, predicted in zip(expected_answerable, predicted_answerable, strict=True)
    )
    tn = sum(
        not expected and not predicted
        for expected, predicted in zip(expected_answerable, predicted_answerable, strict=True)
    )
    fp = sum(
        not expected and predicted
        for expected, predicted in zip(expected_answerable, predicted_answerable, strict=True)
    )
    fn = sum(
        expected and not predicted
        for expected, predicted in zip(expected_answerable, predicted_answerable, strict=True)
    )
    total = len(expected_answerable)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return AnswerabilityMetrics(
        accuracy=(tp + tn) / total if total else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fp / (fp + tn) if fp + tn else 0.0,
        false_negative_rate=fn / (fn + tp) if fn + tp else 0.0,
        false_answers=fp,
        false_refusals=fn,
    )
