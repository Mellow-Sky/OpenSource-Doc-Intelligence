"""Parse, resolve, and validate answer citations against supplied context only."""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from app.core.exceptions import ProviderError
from app.domain.chunks import Chunk
from app.domain.citations import (
    BuiltContext,
    Citation,
    CitationDetail,
    CitationJudgeDecision,
    CitationMarkerResult,
    CitationReport,
    CitationValidation,
    Claim,
    ContextChunkReference,
    ContextSource,
)

_CITATION_RE = re.compile(r"(?<!\\)\[(\d{1,4})\]")
_WORD_RE = re.compile(r"[\w]+(?:[./:@-][\w]+)*", re.UNICODE)
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")
_LEADING_MARKDOWN_RE = re.compile(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)")
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARIES = frozenset("!?\u3002\uff01\uff1f;\uff1b\n")
_QUESTION_ENDINGS = ("?", "\uff1f")
_CLAIM_TRIM_CHARACTERS = " .,:;!?\u3002\uff01\uff1f\uff1b\uff1a"
_CITATION_GAP_CHARACTERS = " \t\r\n.,;:!?\u3002\uff01\uff1f\uff1b\uff1a()"
_REFUSAL_PHRASES = (
    "not enough evidence",
    "insufficient evidence",
    "cannot determine",
    "do not know",
    "don't know",
    "没有足够证据",
    "无法确定",
    "不知道",
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}


class CitationValidationPort(Protocol):
    """Optional semantic judge port implemented by an LLM adapter."""

    async def validate(
        self,
        *,
        claim: str,
        evidence: str,
        title: str,
        section: str,
    ) -> CitationJudgeDecision:
        """Return a strict semantic entailment decision."""


@dataclass(frozen=True, slots=True)
class _Marker:
    number: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _ResolvedEvidence:
    reference: ContextChunkReference
    quote: str
    score: float


class CitationService:
    """Resolve only numbered context sources and measure claim-level grounding."""

    def __init__(
        self,
        *,
        validator: CitationValidationPort | None = None,
        support_threshold: float = 0.2,
        judge_weight: float = 0.65,
        validation_timeout_seconds: float = 15.0,
        max_quote_characters: int = 500,
    ) -> None:
        if not 0 <= support_threshold <= 1:
            raise ValueError("support_threshold must be between zero and one")
        if not 0 <= judge_weight <= 1:
            raise ValueError("judge_weight must be between zero and one")
        if validation_timeout_seconds <= 0:
            raise ValueError("validation_timeout_seconds must be positive")
        if max_quote_characters < 1:
            raise ValueError("max_quote_characters must be positive")
        self._validator = validator
        self._support_threshold = support_threshold
        self._judge_weight = judge_weight
        self._validation_timeout_seconds = validation_timeout_seconds
        self._max_quote_characters = max_quote_characters

    async def analyze(self, answer: str, context: BuiltContext) -> CitationReport:
        """Parse an answer and validate every citation occurrence against its source."""
        markers = _citation_markers(answer)
        claims = _extract_claims(answer, markers)
        assignments = _assign_markers(answer, markers, claims)
        claim_citations: dict[int, list[int]] = defaultdict(list)
        allowed_numbers = {source.number for source in context.sources}
        for marker_index, claim_index in assignments.items():
            marker = markers[marker_index]
            if (
                marker.number in allowed_numbers
                and marker.number not in claim_citations[claim_index]
            ):
                claim_citations[claim_index].append(marker.number)
        claims = [
            claim.model_copy(update={"citation_numbers": claim_citations.get(index, [])})
            for index, claim in enumerate(claims)
        ]

        validations: list[CitationValidation] = []
        marker_results: list[CitationMarkerResult] = []
        occurrence_citations: list[tuple[int, Citation, CitationValidation]] = []
        invalid_numbers: set[int] = set()
        supported_markers = 0
        correctness_total = 0.0
        judge_decisions: list[CitationJudgeDecision] = []

        for marker_index, marker in enumerate(markers):
            assigned_claim_index = assignments.get(marker_index)
            source = context.source(marker.number)
            if source is None:
                invalid_numbers.add(marker.number)
                claim = claims[assigned_claim_index] if assigned_claim_index is not None else None
                reason = "citation number was not present in the supplied context"
                if assigned_claim_index is not None:
                    validations.append(
                        CitationValidation(
                            claim=claim,
                            citation=None,
                            supported=False,
                            score=0.0,
                            reason=reason,
                        )
                    )
                marker_results.append(
                    CitationMarkerResult(
                        number=marker.number,
                        claim=claim,
                        citation=None,
                        supported=False,
                        score=0.0,
                        reason=reason,
                    )
                )
                continue
            if assigned_claim_index is None:
                marker_results.append(
                    CitationMarkerResult(
                        number=marker.number,
                        claim=None,
                        citation=None,
                        supported=False,
                        score=0.0,
                        reason="citation marker was not attached to an answer claim",
                    )
                )
                continue
            claim = claims[assigned_claim_index]
            evidence = _resolve_evidence(
                claim.text,
                source.chunks,
                max_quote_characters=self._max_quote_characters,
            )
            supported, score, reason, judge_decision = await self._validate_support(
                claim=claim,
                source=source,
                evidence=evidence,
            )
            citation = _to_citation(
                source=source,
                evidence=evidence,
                supported=supported,
                validation_score=score,
            )
            validation = CitationValidation(
                claim=claim,
                citation=citation,
                supported=supported,
                score=score,
                reason=reason,
            )
            validations.append(validation)
            marker_results.append(
                CitationMarkerResult(
                    number=marker.number,
                    claim=claim,
                    citation=citation,
                    supported=supported,
                    score=score,
                    reason=reason,
                )
            )
            occurrence_citations.append((marker.number, citation, validation))
            correctness_total += score
            if supported:
                supported_markers += 1
            if judge_decision is not None:
                judge_decisions.append(judge_decision)

        citations = _aggregate_citations(occurrence_citations)
        required_claims = {index for index, claim in enumerate(claims) if claim.requires_citation}
        complete_claims = {
            index
            for index in required_claims
            if any(number in allowed_numbers for number in claims[index].citation_numbers)
        }
        supported_claims = {
            index
            for index in required_claims
            if any(
                validation.supported
                and validation.claim.start_offset == claims[index].start_offset
                and validation.claim.end_offset == claims[index].end_offset
                for validation in validations
            )
        }
        marker_count = len(markers)
        required_count = len(required_claims)
        return CitationReport(
            claims=claims,
            citations=citations,
            validations=validations,
            marker_results=marker_results,
            invalid_citation_numbers=sorted(invalid_numbers),
            citation_marker_count=marker_count,
            citation_precision=(supported_markers / marker_count if marker_count else 1.0),
            citation_recall=(len(supported_claims) / required_count if required_count else 1.0),
            claim_coverage=(len(supported_claims) / required_count if required_count else 1.0),
            citation_correctness=(correctness_total / marker_count if marker_count else 1.0),
            citation_completeness=(
                len(complete_claims) / required_count if required_count else 1.0
            ),
            judge_prompt_tokens=sum(item.prompt_tokens for item in judge_decisions),
            judge_completion_tokens=sum(item.completion_tokens for item in judge_decisions),
            judge_latency_ms=sum(item.latency_ms for item in judge_decisions),
            judge_model=next(
                (item.model for item in judge_decisions if item.model is not None),
                None,
            ),
        )

    async def _validate_support(
        self,
        *,
        claim: Claim,
        source: ContextSource,
        evidence: _ResolvedEvidence,
    ) -> tuple[bool, float, str, CitationJudgeDecision | None]:
        rule_supported = evidence.score >= self._support_threshold
        if self._validator is None:
            reason = (
                "lexical evidence overlap met the configured threshold"
                if rule_supported
                else "lexical evidence overlap was below the configured threshold"
            )
            return rule_supported, evidence.score, reason, None
        try:
            async with asyncio.timeout(self._validation_timeout_seconds):
                decision = await self._validator.validate(
                    claim=claim.text,
                    evidence=evidence.quote,
                    title=source.title,
                    section=source.section,
                )
        except (ProviderError, TimeoutError):
            reason = (
                "semantic validator unavailable; lexical validation accepted the citation"
                if rule_supported
                else "semantic validator unavailable; lexical validation rejected the citation"
            )
            return rule_supported, evidence.score, reason, None

        score = (1 - self._judge_weight) * evidence.score + self._judge_weight * decision.score
        supported = decision.supported and score >= self._support_threshold
        return supported, min(1.0, max(0.0, score)), decision.reason, decision

    @staticmethod
    def citation_detail(
        chunk: Chunk,
        *,
        previous_chunk: Chunk | None = None,
        next_chunk: Chunk | None = None,
        document_metadata: dict[str, Any] | None = None,
    ) -> CitationDetail:
        """Build the read model returned by a future citation-detail repository adapter."""
        for neighbour in (previous_chunk, next_chunk):
            if neighbour is not None and neighbour.document_id != chunk.document_id:
                raise ValueError("citation neighbours must belong to the cited document")
        return CitationDetail(
            chunk=chunk,
            previous_chunk=previous_chunk,
            next_chunk=next_chunk,
            document_metadata=document_metadata or {},
            source_url=chunk.canonical_url,
        )


def _citation_markers(answer: str) -> list[_Marker]:
    code_ranges = _fenced_code_ranges(answer)
    return [
        _Marker(number=int(match.group(1)), start=match.start(), end=match.end())
        for match in _CITATION_RE.finditer(answer)
        if not any(start <= match.start() < end for start, end in code_ranges)
    ]


def _fenced_code_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    cursor = 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            if start is None:
                start = cursor
            else:
                ranges.append((start, cursor + len(line)))
                start = None
        cursor += len(line)
    if start is not None:
        ranges.append((start, len(text)))
    return ranges


def _extract_claims(answer: str, markers: Sequence[_Marker]) -> list[Claim]:
    masked = list(answer)
    for marker in markers:
        masked[marker.start : marker.end] = " " * (marker.end - marker.start)
    text = "".join(masked)
    claims: list[Claim] = []
    for start, end in _sentence_spans(text):
        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        if trailing <= leading:
            continue
        raw_text = raw[leading:trailing]
        markdown = _LEADING_MARKDOWN_RE.match(raw_text)
        markdown_length = markdown.end() if markdown else 0
        claim_text = _SPACE_RE.sub(" ", raw_text[markdown_length:]).strip()
        if not claim_text:
            continue
        claims.append(
            Claim(
                text=claim_text,
                start_offset=start + leading + markdown_length,
                end_offset=start + trailing,
                requires_citation=_requires_citation(claim_text),
            )
        )
    return claims


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index, character in enumerate(text):
        boundary = character in _SENTENCE_BOUNDARIES
        if character == ".":
            previous = text[index - 1] if index else ""
            following = text[index + 1] if index + 1 < len(text) else ""
            boundary = not (previous.isdigit() and following.isdigit()) and (
                not following or following.isspace()
            )
        if boundary:
            end = index + 1
            if text[start:end].strip():
                spans.append((start, end))
            start = end
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _requires_citation(text: str) -> bool:
    normalized = text.casefold()
    if any(phrase in normalized for phrase in _REFUSAL_PHRASES):
        return False
    if text.rstrip().endswith(_QUESTION_ENDINGS):
        return False
    return len(_WORD_RE.findall(text)) >= 3 or bool(_CJK_RE.search(text))


def _assign_markers(
    answer: str,
    markers: Sequence[_Marker],
    claims: Sequence[Claim],
) -> dict[int, int]:
    assignments: dict[int, int] = {}
    for marker_index, marker in enumerate(markers):
        containing = [
            index
            for index, claim in enumerate(claims)
            if claim.start_offset <= marker.start < claim.end_offset
        ]
        if containing:
            assignments[marker_index] = containing[-1]
            continue
        preceding = [
            (index, claim) for index, claim in enumerate(claims) if claim.end_offset <= marker.start
        ]
        if not preceding:
            continue
        claim_index, claim = preceding[-1]
        gap = _CITATION_RE.sub("", answer[claim.end_offset : marker.start])
        if not gap.strip(_CITATION_GAP_CHARACTERS):
            assignments[marker_index] = claim_index
    return assignments


def _resolve_evidence(
    claim: str,
    references: Sequence[ContextChunkReference],
    *,
    max_quote_characters: int,
) -> _ResolvedEvidence:
    best: _ResolvedEvidence | None = None
    for reference in references:
        excerpts = _evidence_excerpts(reference.content)
        for excerpt in excerpts:
            score = _support_score(claim, excerpt)
            quote = excerpt[:max_quote_characters]
            candidate = _ResolvedEvidence(reference=reference, quote=quote, score=score)
            if best is None or (candidate.score, len(candidate.quote)) > (
                best.score,
                len(best.quote),
            ):
                best = candidate
    if best is None:  # ContextSource requires at least one chunk, but content may be blank.
        reference = references[0]
        return _ResolvedEvidence(reference=reference, quote="", score=0.0)
    return best


def _evidence_excerpts(content: str) -> list[str]:
    excerpts = [content[start:end].strip() for start, end in _sentence_spans(content)]
    excerpts = [excerpt for excerpt in excerpts if excerpt]
    return excerpts or ([content.strip()] if content.strip() else [""])


def _support_score(claim: str, evidence: str) -> float:
    normalized_claim = _SPACE_RE.sub(" ", claim.casefold()).strip(_CLAIM_TRIM_CHARACTERS)
    normalized_evidence = _SPACE_RE.sub(" ", evidence.casefold())
    if normalized_claim and normalized_claim in normalized_evidence:
        return 1.0
    claim_terms = _semantic_terms(normalized_claim)
    evidence_terms = _semantic_terms(normalized_evidence)
    if not claim_terms:
        return 0.0
    overlap = claim_terms & evidence_terms
    recall = len(overlap) / len(claim_terms)
    union = claim_terms | evidence_terms
    jaccard = len(overlap) / len(union) if union else 0.0
    return min(1.0, 0.85 * recall + 0.15 * jaccard)


def _semantic_terms(text: str) -> set[str]:
    terms = {
        token.casefold()
        for token in _WORD_RE.findall(text)
        if token.casefold() not in _STOPWORDS and len(token) > 1
    }
    for sequence in _CJK_RE.findall(text):
        if len(sequence) == 1:
            terms.add(f"cjk:{sequence}")
        else:
            terms.update(f"cjk:{sequence[index : index + 2]}" for index in range(len(sequence) - 1))
    return terms


def _to_citation(
    *,
    source: ContextSource,
    evidence: _ResolvedEvidence,
    supported: bool,
    validation_score: float,
) -> Citation:
    reference = evidence.reference
    retrieval_score = reference.retrieval_score or 0.0
    return Citation(
        number=source.number,
        chunk_id=reference.chunk_id,
        document_id=source.document_id,
        title=source.title,
        section=" > ".join(reference.heading_path) or source.section,
        url=source.url,
        quoted_text=evidence.quote,
        document_type=source.document_type,
        score=min(1.0, max(-1.0, retrieval_score)),
        start_offset=reference.start_offset,
        end_offset=reference.end_offset,
        valid=supported,
        validation_score=validation_score,
    )


def _aggregate_citations(
    occurrences: Sequence[tuple[int, Citation, CitationValidation]],
) -> list[Citation]:
    # One numbered context source may merge adjacent chunks. Keep one public
    # citation per (source number, supporting chunk) so merging evidence never
    # discards the exact chunk mappings needed for provenance and persistence.
    grouped: dict[tuple[int, UUID], list[tuple[Citation, CitationValidation]]] = defaultdict(list)
    for number, citation, validation in occurrences:
        grouped[(number, citation.chunk_id)].append((citation, validation))
    result: list[Citation] = []
    # Preserve document provenance order within a merged numbered source. UUID
    # ordering is unrelated to context/source order and made the public citation
    # sequence nondeterministic across otherwise identical ingestions.
    for number_and_chunk in sorted(
        grouped,
        key=lambda item: (
            item[0],
            grouped[item][0][0].start_offset,
            grouped[item][0][0].end_offset,
            str(item[1]),
        ),
    ):
        values = grouped[number_and_chunk]
        strongest_citation, _ = max(values, key=lambda item: item[1].score)
        average_score = sum(validation.score for _, validation in values) / len(values)
        result.append(
            strongest_citation.model_copy(
                update={
                    "valid": all(validation.supported for _, validation in values),
                    "validation_score": average_score,
                }
            )
        )
    return result
