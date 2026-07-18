"""Deterministic answer-quality metrics that do not require an LLM judge."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"[\w]+(?:[./:+-][\w]+)*", re.UNICODE)
_NUMBER_PATTERN = re.compile(r"(?<!\w)v?\d+(?:\.\d+)*(?:[-+][a-z0-9.]+)?", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AnswerMetrics:
    """All deterministic answer scores for one generated/reference pair."""

    exact_match: float
    token_f1: float
    keyword_coverage: float
    numeric_version_consistency: float


def normalize_answer(text: str) -> str:
    """Normalize Unicode, case, and whitespace while retaining technical identifiers."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_TOKEN_PATTERN.findall(normalized))


def tokenize(text: str) -> list[str]:
    """Tokenize prose and common Kubernetes/code identifiers consistently."""
    return _TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text).casefold())


def exact_match(generated: str, reference: str) -> float:
    """Return one when normalized strings match exactly."""
    return float(normalize_answer(generated) == normalize_answer(reference))


def token_f1(generated: str, reference: str) -> float:
    """Calculate multiset token F1."""
    generated_tokens = tokenize(generated)
    reference_tokens = tokenize(reference)
    if not generated_tokens and not reference_tokens:
        return 1.0
    if not generated_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(generated_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(generated_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(
    generated: str,
    reference: str,
    *,
    keywords: list[str] | None = None,
) -> float:
    """Measure how many required keywords occur in the generated answer."""
    required = {token.casefold() for token in (keywords or tokenize(reference)) if token.strip()}
    if not required:
        return 1.0
    generated_normalized = unicodedata.normalize("NFKC", generated).casefold()
    hits = sum(keyword in generated_normalized for keyword in required)
    return hits / len(required)


def numeric_version_consistency(generated: str, reference: str) -> float:
    """Penalize missing or invented numeric/version claims deterministically."""
    reference_values = {item.casefold() for item in _NUMBER_PATTERN.findall(reference)}
    generated_values = {item.casefold() for item in _NUMBER_PATTERN.findall(generated)}
    if not reference_values:
        return float(not generated_values)
    intersection = reference_values.intersection(generated_values)
    precision = len(intersection) / len(generated_values) if generated_values else 0.0
    recall = len(intersection) / len(reference_values)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_answer(
    generated: str,
    reference: str,
    *,
    keywords: list[str] | None = None,
) -> AnswerMetrics:
    """Compute the complete deterministic answer metric set."""
    return AnswerMetrics(
        exact_match=exact_match(generated, reference),
        token_f1=token_f1(generated, reference),
        keyword_coverage=keyword_coverage(generated, reference, keywords=keywords),
        numeric_version_consistency=numeric_version_consistency(generated, reference),
    )
