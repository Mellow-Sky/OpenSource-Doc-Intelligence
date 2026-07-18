"""Exact SHA-256 and conservative SimHash document deduplication."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import blake2b, sha256
from typing import Any

from app.domain.documents import ParsedDocument, RawDocument

_FENCE_LINE_RE = re.compile(r"^(?: {0,3})(`{3,}|~{3,})")
_FEATURE_RE = re.compile(r"[\w]+(?:[./:@-][\w]+)*", re.UNICODE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SEMANTIC_METADATA_KEYS = (
    "api_group",
    "api_version",
    "branch",
    "kind",
    "kubernetes_version",
    "release_version",
    "version",
)


def normalize_content_for_hash(content: str) -> str:
    """Normalize prose deterministically while preserving fenced-code whitespace."""

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines: list[str] = []
    fence_marker: str | None = None
    fence_length = 0
    previous_blank = False

    for line in content.split("\n"):
        fence_match = _FENCE_LINE_RE.match(line)
        if fence_marker is None and fence_match:
            marker = fence_match.group(1)
            fence_marker = marker[0]
            fence_length = len(marker)
            normalized_lines.append(line.rstrip())
            previous_blank = False
            continue
        if fence_marker is not None:
            normalized_lines.append(line)
            if (
                fence_match
                and fence_match.group(1)[0] == fence_marker
                and len(fence_match.group(1)) >= fence_length
            ):
                fence_marker = None
                fence_length = 0
            previous_blank = False
            continue

        normalized = unicodedata.normalize("NFKC", _CONTROL_RE.sub("", line))
        # Leading whitespace is semantic in indented code and YAML. Preserve such
        # lines conservatively so structurally different configurations cannot collide.
        if normalized.startswith((" ", "\t")):
            normalized = normalized.rstrip()
        else:
            normalized = re.sub(r"[\t ]+", " ", normalized).strip()
        is_blank = not normalized
        if is_blank and previous_blank:
            continue
        normalized_lines.append(normalized)
        previous_blank = is_blank

    return "\n".join(normalized_lines).strip()


def normalized_content_hash(content: str) -> str:
    """Return the SHA-256 digest of normalized content."""

    normalized = normalize_content_for_hash(content)
    return sha256(normalized.encode("utf-8", errors="surrogatepass")).hexdigest()


def simhash64(content: str) -> int:
    """Compute a stable 64-bit SimHash over technical-token three-grams."""

    normalized = unicodedata.normalize("NFKC", content).casefold()
    tokens = _FEATURE_RE.findall(normalized)
    if not tokens:
        return 0
    if len(tokens) >= 3:
        features = ["\x1f".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    else:
        features = tokens
    weights = Counter(features)
    vector = [0] * 64
    for feature, weight in weights.items():
        digest = int.from_bytes(blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += weight if digest & (1 << bit) else -weight
    fingerprint = 0
    for bit, value in enumerate(vector):
        if value >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(left: int, right: int) -> int:
    """Return the number of differing bits between two 64-bit fingerprints."""

    return (left ^ right).bit_count()


def simhash_similarity(left: int, right: int) -> float:
    """Convert 64-bit Hamming distance to a normalized similarity score."""

    return 1.0 - hamming_distance(left, right) / 64.0


class DeduplicationMethod(StrEnum):
    """How a duplicate decision was reached."""

    NONE = "none"
    EXACT_SHA256 = "exact_sha256"
    SIMHASH = "simhash"
    PROTECTED = "protected"


@dataclass(frozen=True, slots=True)
class DeduplicationCandidate:
    """Minimal immutable input used by the deduplication index."""

    external_id: str
    content: str
    source_type: str
    source_version: str | None = None
    document_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, document: RawDocument) -> DeduplicationCandidate:
        """Build a fingerprintable candidate from a loader result."""

        value = document.metadata.get("document_type")
        return cls(
            external_id=document.external_id,
            content=document.content,
            source_type=document.source_type,
            source_version=document.source_version,
            document_type=str(value) if value is not None else None,
            metadata=dict(document.metadata),
        )

    @classmethod
    def from_parsed(cls, document: ParsedDocument) -> DeduplicationCandidate:
        """Build a fingerprintable candidate from a normalized document."""

        return cls(
            external_id=document.external_id,
            content=document.content,
            source_type=document.source_type,
            source_version=document.source_version,
            document_type=document.document_type.value,
            metadata=dict(document.metadata),
        )


@dataclass(frozen=True, slots=True)
class DeduplicationDecision:
    """Auditable outcome, including why a document was retained or skipped."""

    is_duplicate: bool
    method: DeduplicationMethod
    reason: str
    similarity: float | None = None
    matched_external_id: str | None = None


@dataclass(frozen=True, slots=True)
class _FingerprintRecord:
    candidate: DeduplicationCandidate
    exact_hash: str
    simhash: int


class ContentDeduplicator:
    """In-memory decision engine suitable for a batch before repository persistence."""

    def __init__(self, similarity_threshold: float = 0.90) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        self.similarity_threshold = similarity_threshold
        self._records: list[_FingerprintRecord] = []

    def compare(
        self,
        candidate: DeduplicationCandidate,
        existing: DeduplicationCandidate,
    ) -> DeduplicationDecision:
        """Compare two documents with exact-first and metadata-aware near matching."""

        candidate_hash = normalized_content_hash(candidate.content)
        existing_hash = normalized_content_hash(existing.content)
        protection_reason = self._protection_reason(candidate, existing)

        if candidate_hash == existing_hash and protection_reason is None:
            return DeduplicationDecision(
                is_duplicate=True,
                method=DeduplicationMethod.EXACT_SHA256,
                reason="normalized content has the same SHA-256 digest",
                similarity=1.0,
                matched_external_id=existing.external_id,
            )
        if protection_reason is not None:
            return DeduplicationDecision(
                is_duplicate=False,
                method=DeduplicationMethod.PROTECTED,
                reason=protection_reason,
                matched_external_id=existing.external_id,
            )
        if candidate.external_id == existing.external_id:
            return DeduplicationDecision(
                is_duplicate=False,
                method=DeduplicationMethod.PROTECTED,
                reason="same logical document changed; retain it as a new document version",
                matched_external_id=existing.external_id,
            )

        similarity = simhash_similarity(simhash64(candidate.content), simhash64(existing.content))
        if similarity >= self.similarity_threshold:
            return DeduplicationDecision(
                is_duplicate=True,
                method=DeduplicationMethod.SIMHASH,
                reason=(
                    f"SimHash similarity {similarity:.4f} meets threshold "
                    f"{self.similarity_threshold:.4f}"
                ),
                similarity=similarity,
                matched_external_id=existing.external_id,
            )
        return DeduplicationDecision(
            is_duplicate=False,
            method=DeduplicationMethod.NONE,
            reason=(
                f"SimHash similarity {similarity:.4f} is below threshold "
                f"{self.similarity_threshold:.4f}"
            ),
            similarity=similarity,
        )

    def find_duplicate(self, candidate: DeduplicationCandidate) -> DeduplicationDecision:
        """Compare a candidate with indexed records and return the strongest decision."""

        candidate_hash = normalized_content_hash(candidate.content)
        candidate_simhash = simhash64(candidate.content)
        protected: DeduplicationDecision | None = None
        best_non_match: DeduplicationDecision | None = None

        for record in self._records:
            protection_reason = self._protection_reason(candidate, record.candidate)
            if candidate_hash == record.exact_hash and protection_reason is None:
                return DeduplicationDecision(
                    True,
                    DeduplicationMethod.EXACT_SHA256,
                    "normalized content has the same SHA-256 digest",
                    1.0,
                    record.candidate.external_id,
                )
            decision = self._compare_fingerprints(
                candidate,
                candidate_simhash,
                record,
                protection_reason,
            )
            if decision.is_duplicate:
                return decision
            if decision.method is DeduplicationMethod.PROTECTED:
                protected = protected or decision
            elif best_non_match is None or (decision.similarity or 0.0) > (
                best_non_match.similarity or 0.0
            ):
                best_non_match = decision
        return (
            best_non_match
            or protected
            or DeduplicationDecision(
                False, DeduplicationMethod.NONE, "no prior document fingerprints"
            )
        )

    def add(self, candidate: DeduplicationCandidate) -> None:
        """Add a retained document to the current batch index."""

        self._records.append(
            _FingerprintRecord(
                candidate,
                normalized_content_hash(candidate.content),
                simhash64(candidate.content),
            )
        )

    def add_fingerprint(
        self,
        candidate: DeduplicationCandidate,
        *,
        exact_hash: str,
        simhash: int,
    ) -> None:
        """Seed a persisted fingerprint without reloading the complete document body."""

        if not re.fullmatch(r"[0-9a-f]{64}", exact_hash):
            raise ValueError("exact_hash must be a lowercase SHA-256 digest")
        if not 0 <= simhash < 2**64:
            raise ValueError("simhash must be an unsigned 64-bit integer")
        self._records.append(_FingerprintRecord(candidate, exact_hash, simhash))

    def check_and_add(self, candidate: DeduplicationCandidate) -> DeduplicationDecision:
        """Check a candidate and index it only when it should be retained."""

        decision = self.find_duplicate(candidate)
        if not decision.is_duplicate:
            self.add(candidate)
        return decision

    def extend(self, candidates: Sequence[DeduplicationCandidate]) -> list[DeduplicationDecision]:
        """Process a deterministic batch and return one audit decision per input."""

        return [self.check_and_add(candidate) for candidate in candidates]

    def _compare_fingerprints(
        self,
        candidate: DeduplicationCandidate,
        candidate_simhash: int,
        record: _FingerprintRecord,
        protection_reason: str | None,
    ) -> DeduplicationDecision:
        if protection_reason is not None:
            return DeduplicationDecision(
                False,
                DeduplicationMethod.PROTECTED,
                protection_reason,
                matched_external_id=record.candidate.external_id,
            )
        if candidate.external_id == record.candidate.external_id:
            return DeduplicationDecision(
                False,
                DeduplicationMethod.PROTECTED,
                "same logical document changed; retain it as a new document version",
                matched_external_id=record.candidate.external_id,
            )
        similarity = simhash_similarity(candidate_simhash, record.simhash)
        if similarity >= self.similarity_threshold:
            return DeduplicationDecision(
                True,
                DeduplicationMethod.SIMHASH,
                (
                    f"SimHash similarity {similarity:.4f} meets threshold "
                    f"{self.similarity_threshold:.4f}"
                ),
                similarity,
                record.candidate.external_id,
            )
        return DeduplicationDecision(
            False,
            DeduplicationMethod.NONE,
            (
                f"SimHash similarity {similarity:.4f} is below threshold "
                f"{self.similarity_threshold:.4f}"
            ),
            similarity,
        )

    @staticmethod
    def _protection_reason(
        candidate: DeduplicationCandidate,
        existing: DeduplicationCandidate,
    ) -> str | None:
        if (
            candidate.document_type
            and existing.document_type
            and candidate.document_type != existing.document_type
        ):
            return "different document types are retained independently"
        for key in _SEMANTIC_METADATA_KEYS:
            candidate_value = candidate.metadata.get(key)
            existing_value = existing.metadata.get(key)
            if (
                candidate_value is not None
                and existing_value is not None
                and str(candidate_value).casefold() != str(existing_value).casefold()
            ):
                return f"metadata mismatch for {key}; retain distinct semantic versions"
        if (
            candidate.external_id != existing.external_id
            and candidate.source_version
            and existing.source_version
            and candidate.source_version != existing.source_version
        ):
            return "different source versions are retained for distinct documents"
        return None


# Backward-friendly name for callers that describe the component by behavior.
DocumentDeduplicator = ContentDeduplicator
