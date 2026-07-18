"""Exact and near-duplicate detection with auditable decisions."""

from app.ingestion.deduplication.content import (
    ContentDeduplicator,
    DeduplicationCandidate,
    DeduplicationDecision,
    DeduplicationMethod,
    DocumentDeduplicator,
    hamming_distance,
    normalize_content_for_hash,
    normalized_content_hash,
    simhash64,
    simhash_similarity,
)

__all__ = [
    "ContentDeduplicator",
    "DeduplicationCandidate",
    "DeduplicationDecision",
    "DeduplicationMethod",
    "DocumentDeduplicator",
    "hamming_distance",
    "normalize_content_for_hash",
    "normalized_content_hash",
    "simhash64",
    "simhash_similarity",
]
