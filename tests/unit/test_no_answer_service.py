"""Combined no-answer gates reject missing, weak, and unsupported evidence."""

from __future__ import annotations

from uuid import UUID

import pytest

from app.domain.citations import CitationMarkerResult, CitationReport
from app.domain.retrieval import (
    EvidenceSufficiency,
    QueryFilters,
    RetrievalCandidate,
    RetrievalMode,
    RetrievalOutcome,
    RetrievalQuery,
)
from app.services.no_answer_service import NoAnswerService


def _candidate(
    number: int,
    *,
    score: float | None,
    vector_score: float | None = None,
    fused_score: float | None = None,
    content: str = "Deployment rollback",
):
    return RetrievalCandidate(
        chunk_id=UUID(int=number),
        document_id=UUID(int=100 + number),
        document_title="Kubernetes Deployment",
        document_type="official_documentation",
        heading_path=["Rollback"],
        content=content,
        rerank_rank=number if score is not None else None,
        rerank_score=score,
        vector_score=vector_score,
        fused_rank=number,
        fused_score=fused_score,
    )


def _outcome(
    candidates,
    *,
    keyword_count: int = 1,
    vector_count: int = 1,
    reranker_degraded: bool = False,
):
    return RetrievalOutcome(
        query=RetrievalQuery(
            original="Deployment rollback",
            rewritten="Deployment rollback",
            filters=QueryFilters(),
            mode=RetrievalMode.HYBRID,
            top_k=8,
        ),
        candidates=candidates,
        keyword_count=keyword_count,
        vector_count=vector_count,
        reranked_count=0 if reranker_degraded else len(candidates),
        reranker_degraded=reranker_degraded,
        reranker_reason="ProviderError" if reranker_degraded else None,
    )


class SufficientJudge:
    async def evaluate(self, query, candidates):
        return EvidenceSufficiency(sufficient=True, score=0.8, reason="direct support")


def _service(*, judge=None, top1_threshold=None):
    return NoAnswerService(
        top1_threshold=top1_threshold,
        average_threshold=None,
        margin_threshold=0.05,
        score_top_k=3,
        topic_overlap_threshold=0.05,
        gray_zone_lower=0.35,
        gray_zone_upper=0.65,
        evidence_threshold=0.6,
        citation_coverage_threshold=0.6,
        judge=judge,
    )


@pytest.mark.asyncio
async def test_empty_retrieval_is_always_unanswerable() -> None:
    decision = await _service().assess(
        "Deployment rollback",
        _outcome([], keyword_count=0, vector_count=0),
    )

    assert decision.answerable is False
    assert decision.reason == "retrieval_empty"
    assert decision.confidence == 0


@pytest.mark.asyncio
async def test_calibrated_top1_gate_rejects_weak_result() -> None:
    decision = await _service(top1_threshold=0.7).assess(
        "Deployment rollback",
        _outcome([_candidate(1, score=0.4)]),
    )

    assert decision.answerable is False
    assert decision.reason == "top1_below_threshold"
    assert decision.diagnostics["top1_score"] == 0.4
    assert decision.diagnostics["score_family"] == "reranker"
    assert decision.diagnostics["calibrated_score_thresholds_applied"] is True


@pytest.mark.asyncio
async def test_reranker_threshold_is_not_applied_to_rrf_fallback_scores() -> None:
    decision = await _service(judge=SufficientJudge(), top1_threshold=0.7).assess(
        "Deployment rollback",
        _outcome(
            [
                _candidate(
                    1,
                    score=None,
                    vector_score=0.2,
                    fused_score=1 / 61,
                )
            ],
            reranker_degraded=True,
        ),
    )

    assert decision.answerable is True
    assert decision.reason == "evidence_judge_sufficient"
    assert decision.diagnostics["score_family"] == "vector"
    assert decision.diagnostics["top1_score"] == 0.2
    assert decision.diagnostics["calibrated_score_thresholds_applied"] is False


@pytest.mark.asyncio
async def test_keyword_only_fallback_uses_topic_signals_not_rrf_as_confidence() -> None:
    decision = await _service(top1_threshold=0.7).assess(
        "Deployment rollback",
        _outcome(
            [_candidate(1, score=None, fused_score=1 / 61)],
            keyword_count=1,
            vector_count=0,
            reranker_degraded=True,
        ),
    )

    assert decision.answerable is True
    assert decision.reason == "retrieval_confident"
    assert decision.diagnostics["score_family"] == "topic_only"
    assert decision.diagnostics["top1_score"] is None


@pytest.mark.asyncio
async def test_topic_mismatch_rejects_plausible_high_score() -> None:
    decision = await _service().assess(
        "PostgreSQL vacuum tuning",
        _outcome([_candidate(1, score=0.95)]),
    )

    assert decision.answerable is False
    assert decision.reason == "topic_mismatch"


@pytest.mark.asyncio
async def test_gray_zone_uses_evidence_sufficiency_judge() -> None:
    decision = await _service(judge=SufficientJudge()).assess(
        "Deployment rollback",
        _outcome([_candidate(1, score=0.2)]),
    )

    assert decision.answerable is True
    assert decision.reason == "evidence_judge_sufficient"
    assert decision.evidence_sufficiency_score == 0.8


@pytest.mark.asyncio
async def test_citation_coverage_can_downgrade_generated_answer() -> None:
    service = _service()
    decision = await service.assess(
        "Deployment rollback",
        _outcome([_candidate(1, score=0.95)]),
    )

    downgraded = service.apply_citation_coverage(decision, 0.25)

    assert decision.answerable is True
    assert downgraded.answerable is False
    assert downgraded.reason == "citation_coverage_below_threshold"
    assert downgraded.diagnostics["citation_coverage"] == 0.25


@pytest.mark.asyncio
async def test_forged_or_unsupported_marker_rejects_otherwise_covered_answer() -> None:
    service = _service()
    decision = await service.assess(
        "Deployment rollback",
        _outcome([_candidate(1, score=0.95)]),
    )
    report = CitationReport(
        marker_results=[
            CitationMarkerResult(
                number=1,
                supported=True,
                score=1.0,
                reason="supported",
            ),
            CitationMarkerResult(
                number=999,
                supported=False,
                score=0.0,
                reason="not supplied",
            ),
        ],
        invalid_citation_numbers=[999],
        citation_marker_count=2,
        citation_precision=0.5,
        citation_recall=1.0,
        claim_coverage=1.0,
        citation_correctness=0.5,
        citation_completeness=1.0,
    )

    rejected = service.apply_citation_report(decision, report)

    assert rejected.answerable is False
    assert rejected.reason == "citation_validation_failed"
    assert rejected.diagnostics["invalid_citation_numbers"] == [999]
    assert rejected.diagnostics["citation_precision"] == 0.5
