"""Lossless query normalization and lightweight language analysis."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.domain.retrieval import QueryFilters
from app.retrieval.filters import QueryFilterExtractor

_WHITESPACE = re.compile(r"\s+")
_HAN_CHARACTER = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
_LATIN_CHARACTER = re.compile(r"[A-Za-z]")
_BACKTICK_TERM = re.compile(r"`(?P<term>[^`\r\n]+)`")
_VERSION_TERM = re.compile(r"(?<![\w.-])v?\d+\.\d+(?:\.\d+)?(?![\w.-])", re.IGNORECASE)
_API_TERM = re.compile(
    r"(?<![\w./-])[A-Za-z][A-Za-z0-9.-]*/v\d+(?:(?:alpha|beta)\d+)?(?![\w.-])",
    re.IGNORECASE,
)
_IDENTIFIER_TERM = re.compile(
    r"(?<![\w.-])(?:[A-Za-z][A-Za-z0-9]*(?:[._/-][A-Za-z0-9]+)+|"
    r"[a-z]+[A-Z][A-Za-z0-9]*)(?![\w.-])"
)
_KUBERNETES_TERM = re.compile(
    r"(?<![A-Za-z0-9_])(?:Kubernetes|K8s|kubectl|kubelet|kubeadm|"
    r"kube-apiserver|kube-controller-manager|kube-scheduler|kube-proxy)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PreprocessedQuery:
    """A normalized query plus analysis that does not alter technical terms."""

    original: str
    normalized: str
    language: str
    filters: QueryFilters
    protected_terms: tuple[str, ...]


def normalize_query(query: str, *, max_length: int = 4000) -> str:
    """Normalize Unicode and whitespace without lowercasing technical tokens."""
    if max_length < 1:
        msg = "max_length must be positive"
        raise ValueError(msg)
    if not isinstance(query, str):
        msg = "query must be a string"
        raise TypeError(msg)

    normalized = unicodedata.normalize("NFKC", query)
    # Whitespace controls are converted below; other C0/C1 controls are not
    # meaningful query text and can confuse PostgreSQL tokenization.
    normalized = "".join(
        character
        for character in normalized
        if character.isspace() or unicodedata.category(character) != "Cc"
    )
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    if not normalized:
        msg = "query must contain non-whitespace text"
        raise ValueError(msg)
    if len(normalized) > max_length:
        msg = f"normalized query exceeds maximum length of {max_length} characters"
        raise ValueError(msg)
    return normalized


def detect_language(query: str) -> str:
    """Classify a query as English, Chinese, mixed, or unknown."""
    contains_han = _HAN_CHARACTER.search(query) is not None
    contains_latin = _LATIN_CHARACTER.search(query) is not None
    if contains_han and contains_latin:
        return "mixed"
    if contains_han:
        return "zh"
    if contains_latin:
        return "en"
    return "unknown"


def extract_protected_terms(query: str) -> tuple[str, ...]:
    """Return version, API, Kubernetes, and code tokens in source order."""
    matches: list[tuple[int, int, str]] = []
    for pattern in (
        _BACKTICK_TERM,
        _VERSION_TERM,
        _API_TERM,
        _IDENTIFIER_TERM,
        _KUBERNETES_TERM,
    ):
        for match in pattern.finditer(query):
            value = match.groupdict().get("term", match.group(0)).strip()
            if value:
                matches.append((match.start(), match.end(), value))

    result: list[str] = []
    seen: set[str] = set()
    for _, _, value in sorted(matches, key=lambda item: (item[0], item[1], item[2].casefold())):
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)


class QueryPreprocessor:
    """Injectable, side-effect-free query preprocessing component."""

    def __init__(
        self,
        *,
        max_length: int = 4000,
        filter_extractor: QueryFilterExtractor | None = None,
    ) -> None:
        if max_length < 1:
            msg = "max_length must be positive"
            raise ValueError(msg)
        self._max_length = max_length
        self._filter_extractor = filter_extractor or QueryFilterExtractor()

    def preprocess(
        self,
        query: str,
        supplied_filters: QueryFilters | None = None,
    ) -> PreprocessedQuery:
        """Normalize and analyze a query while preserving its original form."""
        normalized = normalize_query(query, max_length=self._max_length)
        filters = self._filter_extractor.extract_and_merge(normalized, supplied_filters)
        protected_terms = list(extract_protected_terms(normalized))
        protected_keys = {term.casefold() for term in protected_terms}
        # Known API kinds are meaningful even when they look like ordinary
        # PascalCase words, so the filter extractor supplies that domain signal.
        for kind in filters.kinds:
            if kind.casefold() not in protected_keys:
                protected_terms.append(kind)
                protected_keys.add(kind.casefold())
        return PreprocessedQuery(
            original=query,
            normalized=normalized,
            language=detect_language(normalized),
            filters=filters,
            protected_terms=tuple(protected_terms),
        )

    def process(
        self,
        query: str,
        supplied_filters: QueryFilters | None = None,
    ) -> PreprocessedQuery:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(query, supplied_filters)


def preprocess_query(
    query: str,
    supplied_filters: QueryFilters | None = None,
    *,
    max_length: int = 4000,
) -> PreprocessedQuery:
    """Convenience wrapper for one-off query preprocessing."""
    return QueryPreprocessor(max_length=max_length).preprocess(query, supplied_filters)
