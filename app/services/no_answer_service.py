"""Combined retrieval, evidence, and citation based no-answer detection."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Protocol

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.domain.citations import CitationReport
from app.domain.retrieval import (
    EvidenceSufficiency,
    NoAnswerDecision,
    RetrievalCandidate,
    RetrievalOutcome,
)

_QUERY_TERM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*|[\u3400-\u9fff]{2,}")
_STOP_TERMS = {
    "a",
    "an",
    "and",
    "for",
    "how",
    "in",
    "is",
    "of",
    "the",
    "to",
    "what",
    "with",
    "如何",
    "什么",
    "怎么",
}


class EvidenceSufficiencyJudge(Protocol):
    """Port for an optional lightweight gray-zone evidence judge."""

    async def evaluate(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> EvidenceSufficiency:
        """Assess whether retrieved text can support a grounded answer."""
        ...


class NoAnswerService:
    """Combine calibrated score gates with channel, topic, judge, and citation signals."""

    def __init__(
        self,
        *,
        top1_threshold: float | None,
        average_threshold: float | None,
        margin_threshold: float | None,
        score_top_k: int,
        topic_overlap_threshold: float,
        gray_zone_lower: float,
        gray_zone_upper: float,
        evidence_threshold: float,
        citation_coverage_threshold: float,
        judge: EvidenceSufficiencyJudge | None = None,
    ) -> None:
        self._top1_threshold = top1_threshold
        self._average_threshold = average_threshold
        self._margin_threshold = margin_threshold
        self._score_top_k = score_top_k
        self._topic_overlap_threshold = topic_overlap_threshold
        self._gray_zone_lower = gray_zone_lower
        self._gray_zone_upper = gray_zone_upper
        self._evidence_threshold = evidence_threshold
        self._citation_coverage_threshold = citation_coverage_threshold
        self._judge = judge

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        judge: EvidenceSufficiencyJudge | None = None,
    ) -> NoAnswerService:
        """Create a detector from centralized settings rather than magic constants."""
        return cls(
            top1_threshold=settings.no_answer_top1_threshold,
            average_threshold=settings.no_answer_avg_threshold,
            margin_threshold=settings.no_answer_margin_threshold,
            score_top_k=settings.no_answer_top_k,
            topic_overlap_threshold=settings.no_answer_topic_overlap_threshold,
            gray_zone_lower=settings.no_answer_gray_zone_lower,
            gray_zone_upper=settings.no_answer_gray_zone_upper,
            evidence_threshold=settings.evidence_sufficiency_threshold,
            citation_coverage_threshold=settings.citation_coverage_threshold,
            judge=judge,
        )

    async def assess(
        self,
        query: str,
        retrieval: RetrievalOutcome,
    ) -> NoAnswerDecision:
        """Decide whether generation is justified by the current retrieval evidence."""
        candidates = retrieval.candidates
        if not candidates or (retrieval.keyword_count == 0 and retrieval.vector_count == 0):
            return _decision(
                answerable=False,
                retrieval_confidence=0.0,
                evidence_score=0.0,
                reason="retrieval_empty",
                diagnostics={"keyword_count": 0, "vector_count": 0},
            )

        reranker_scores_available = retrieval.reranked_count > 0 and not (
            retrieval.reranker_degraded
        )
        score_family = "reranker" if reranker_scores_available else "vector"
        scores = [
            score
            for candidate in candidates[: self._score_top_k]
            if (
                score := _relevance_score(
                    candidate,
                    use_reranker=reranker_scores_available,
                )
            )
            is not None
        ]
        if not scores and not reranker_scores_available:
            # Keyword-only and RRF-only scores do not share a calibrated relevance
            # scale. Topic/channel signals remain available without pretending that
            # a rank-fusion score is a cross-encoder probability.
            score_family = "topic_only"
        top1 = scores[0] if scores else None
        average = sum(scores) / len(scores) if scores else None
        margin = scores[0] - sum(scores[1:]) / len(scores[1:]) if len(scores) > 1 else None
        topic_overlap = max(
            (_topic_overlap(query, candidate) for candidate in candidates[: self._score_top_k]),
            default=0.0,
        )
        channel_signal = 1.0 if retrieval.keyword_count > 0 and retrieval.vector_count > 0 else 0.65
        score_signal = _clip01(top1) if top1 is not None else 0.5
        topic_signal = _clip01(topic_overlap / max(self._topic_overlap_threshold * 4, 0.01))
        retrieval_confidence = _clip01(
            0.55 * score_signal + 0.30 * topic_signal + 0.15 * channel_signal
        )
        diagnostics: dict[str, object] = {
            "top1_score": top1,
            "top_k_average": average,
            "top1_margin": margin,
            "topic_overlap": topic_overlap,
            "keyword_count": retrieval.keyword_count,
            "vector_count": retrieval.vector_count,
            "score_family": score_family,
            "calibrated_score_thresholds_applied": reranker_scores_available,
            "reranker_degraded": retrieval.reranker_degraded,
            "reranker_reason": retrieval.reranker_reason,
            "margin_below_threshold": (
                margin is not None
                and reranker_scores_available
                and self._margin_threshold is not None
                and margin < self._margin_threshold
            ),
        }

        if (
            reranker_scores_available
            and self._top1_threshold is not None
            and (top1 is None or top1 < self._top1_threshold)
        ):
            return _decision(
                answerable=False,
                retrieval_confidence=retrieval_confidence,
                evidence_score=retrieval_confidence,
                reason="top1_below_threshold",
                diagnostics=diagnostics,
            )
        if (
            reranker_scores_available
            and self._average_threshold is not None
            and (average is None or average < self._average_threshold)
        ):
            return _decision(
                answerable=False,
                retrieval_confidence=retrieval_confidence,
                evidence_score=retrieval_confidence,
                reason="top_k_average_below_threshold",
                diagnostics=diagnostics,
            )
        if topic_overlap < self._topic_overlap_threshold:
            return _decision(
                answerable=False,
                retrieval_confidence=retrieval_confidence,
                evidence_score=retrieval_confidence,
                reason="topic_mismatch",
                diagnostics=diagnostics,
            )

        margin_low = bool(diagnostics["margin_below_threshold"])
        if margin_low:
            retrieval_confidence *= 0.85

        if retrieval_confidence >= self._gray_zone_upper and not margin_low:
            return _decision(
                answerable=True,
                retrieval_confidence=retrieval_confidence,
                evidence_score=retrieval_confidence,
                reason="retrieval_confident",
                diagnostics=diagnostics,
            )

        if retrieval_confidence < self._gray_zone_lower:
            return _decision(
                answerable=False,
                retrieval_confidence=retrieval_confidence,
                evidence_score=retrieval_confidence,
                reason="retrieval_confidence_low",
                diagnostics=diagnostics,
            )

        if self._judge is None:
            return _decision(
                answerable=False,
                retrieval_confidence=retrieval_confidence,
                evidence_score=retrieval_confidence,
                reason="gray_zone_without_evidence_judge",
                diagnostics=diagnostics,
            )
        try:
            judged = await self._judge.evaluate(query, candidates[: self._score_top_k])
        except (ProviderError, TimeoutError, ValueError):
            return _decision(
                answerable=False,
                retrieval_confidence=retrieval_confidence,
                evidence_score=retrieval_confidence,
                reason="evidence_judge_unavailable",
                diagnostics=diagnostics,
            )
        answerable = judged.sufficient and judged.score >= self._evidence_threshold
        diagnostics["judge_score"] = judged.score
        diagnostics["judge_prompt_tokens"] = judged.prompt_tokens
        diagnostics["judge_completion_tokens"] = judged.completion_tokens
        diagnostics["judge_latency_ms"] = judged.latency_ms
        diagnostics["judge_model"] = judged.model
        return _decision(
            answerable=answerable,
            retrieval_confidence=retrieval_confidence,
            evidence_score=judged.score,
            reason="evidence_judge_sufficient" if answerable else "evidence_judge_insufficient",
            diagnostics=diagnostics,
        )

    def apply_citation_coverage(
        self,
        decision: NoAnswerDecision,
        citation_coverage: float,
    ) -> NoAnswerDecision:
        """Reject a generated answer when key factual claims remain unsupported."""
        coverage = _clip01(citation_coverage)
        diagnostics = dict(decision.diagnostics)
        diagnostics["citation_coverage"] = coverage
        if not decision.answerable or coverage >= self._citation_coverage_threshold:
            return decision.model_copy(update={"diagnostics": diagnostics})
        return _decision(
            answerable=False,
            retrieval_confidence=decision.retrieval_confidence,
            evidence_score=min(decision.evidence_sufficiency_score, coverage),
            reason="citation_coverage_below_threshold",
            diagnostics=diagnostics,
        )

    def apply_citation_report(
        self,
        decision: NoAnswerDecision,
        report: CitationReport,
    ) -> NoAnswerDecision:
        """Fail closed on forged, unattached, or unsupported citation markers.

        Coverage alone is insufficient: a draft can cover every real claim while
        also emitting a source number that was never supplied. The literal answer
        must never be returned when any of its markers fails validation.
        """

        diagnostics = dict(decision.diagnostics)
        rejected_markers = sum(not marker.supported for marker in report.marker_results)
        if not report.marker_results and report.citation_marker_count:
            rejected_markers = max(
                0,
                report.citation_marker_count
                - int(report.citation_marker_count * report.citation_precision),
            )
        diagnostics.update(
            {
                "citation_coverage": report.claim_coverage,
                "citation_precision": report.citation_precision,
                "citation_marker_count": report.citation_marker_count,
                "citation_rejected_marker_count": rejected_markers,
                "invalid_citation_numbers": list(report.invalid_citation_numbers),
            }
        )
        unsafe_markers = bool(report.invalid_citation_numbers) or (
            report.citation_precision < 1.0 or rejected_markers > 0
        )
        if decision.answerable and unsafe_markers:
            return _decision(
                answerable=False,
                retrieval_confidence=decision.retrieval_confidence,
                evidence_score=min(
                    decision.evidence_sufficiency_score,
                    report.citation_precision,
                ),
                reason="citation_validation_failed",
                diagnostics=diagnostics,
            )
        covered = self.apply_citation_coverage(decision, report.claim_coverage)
        merged_diagnostics = dict(covered.diagnostics)
        merged_diagnostics.update(diagnostics)
        return covered.model_copy(update={"diagnostics": merged_diagnostics})


def _relevance_score(
    candidate: RetrievalCandidate,
    *,
    use_reranker: bool,
) -> float | None:
    scores = (candidate.rerank_score,) if use_reranker else (candidate.vector_score,)
    for score in scores:
        if score is not None and math.isfinite(score):
            return score
    return None


def _topic_overlap(query: str, candidate: RetrievalCandidate) -> float:
    query_terms = _terms(query)
    if not query_terms:
        return 0.0
    evidence = " ".join([candidate.document_title, *candidate.heading_path, candidate.content])
    evidence_terms = _terms(evidence)
    return len(query_terms & evidence_terms) / len(query_terms)


def _terms(text: str) -> set[str]:
    return {
        term.casefold()
        for term in _QUERY_TERM.findall(text)
        if term.casefold() not in _STOP_TERMS and len(term) > 1
    }


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _decision(
    *,
    answerable: bool,
    retrieval_confidence: float,
    evidence_score: float,
    reason: str,
    diagnostics: dict[str, object],
) -> NoAnswerDecision:
    retrieval_confidence = _clip01(retrieval_confidence)
    evidence_score = _clip01(evidence_score)
    confidence = _clip01((retrieval_confidence + evidence_score) / 2)
    return NoAnswerDecision(
        answerable=answerable,
        confidence=confidence,
        reason=reason,
        retrieval_confidence=retrieval_confidence,
        evidence_sufficiency_score=evidence_score,
        diagnostics=diagnostics,
    )
