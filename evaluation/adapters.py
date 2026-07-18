"""Adapters from application chat results to stable evaluation responses."""

from __future__ import annotations

from app.domain.chat import ChatResult
from app.domain.citations import CitationReport
from app.domain.evaluation import EvaluationCase
from app.domain.retrieval import RetrievalCandidate
from app.services.chat_service import ChatService
from evaluation.models import (
    EvaluationCitationMarker,
    EvaluationCitationSummary,
    EvaluationEvidence,
    EvaluationResponse,
)


class ChatEvaluationExecutor:
    """Execute dataset cases through the same ChatService used by the HTTP API."""

    def __init__(self, service: ChatService, *, top_k: int | None = None) -> None:
        self._service = service
        self._top_k = top_k

    async def execute(self, case: EvaluationCase) -> EvaluationResponse:
        """Run a typed evaluation case while preserving its conversation history."""
        result = await self._service.complete(
            case.question,
            top_k=self._top_k,
            history_override=case.conversation_history,
        )
        return response_from_chat(result)


def response_from_chat(result: ChatResult) -> EvaluationResponse:
    """Preserve retrieval rank provenance, citations, latency, and usage."""
    candidates = sorted(result.retrieval.trace_candidates or result.retrieval.candidates, key=_rank)
    evidence = [
        EvaluationEvidence(
            chunk_id=str(candidate.chunk_id),
            document_id=str(candidate.document_id),
            title=candidate.document_title,
            section=" > ".join(candidate.heading_path),
            document_type=candidate.document_type,
            content=candidate.content,
            score=(
                candidate.rerank_score
                if candidate.rerank_score is not None
                else candidate.fused_score
            ),
            rank=index,
        )
        for index, candidate in enumerate(candidates, start=1)
    ]
    # ChatResult.citations intentionally exposes only valid citations. Evaluation
    # must instead retain every marker outcome, otherwise invalid markers vanish
    # and Citation Precision is systematically inflated. A post-generation
    # refusal does not expose the rejected draft, so its draft citation report is
    # excluded from metrics for the returned refusal text.
    report = result.citation_report if result.answerable else None
    if report is None:
        citations = EvaluationCitationSummary()
    else:
        markers = _evaluation_markers(report)
        citations = EvaluationCitationSummary(
            citation_ids=[str(item.chunk_id) for item in report.citations],
            markers=markers,
            validity=[marker.valid for marker in markers],
            support_scores=[marker.support_score for marker in markers],
            claim_requires_citation=[claim.requires_citation for claim in report.claims],
            claim_supported=[
                any(
                    validation.claim == claim and validation.supported
                    for validation in report.validations
                )
                for claim in report.claims
            ],
        )
    return EvaluationResponse(
        generated_answer=result.answer,
        predicted_answerable=result.answerable,
        rewritten_query=result.rewritten_query,
        evidence=evidence,
        citations=citations,
        latency_ms=result.latency.model_dump(),
        usage=result.usage.model_dump(),
    )


def _evaluation_markers(report: CitationReport) -> list[EvaluationCitationMarker]:
    if report.marker_results:
        return [
            EvaluationCitationMarker(
                number=item.number,
                chunk_id=(str(item.citation.chunk_id) if item.citation is not None else None),
                valid=item.supported,
                support_score=item.score,
            )
            for item in report.marker_results
        ]

    # Compatibility for persisted/manual reports produced before marker-level
    # outcomes were introduced. Unknown legacy marker numbers remain null rather
    # than inventing a source identifier, but still count as rejected markers.
    invalid_numbers = iter(report.invalid_citation_numbers)
    markers: list[EvaluationCitationMarker] = []
    for validation in report.validations:
        citation = validation.citation
        markers.append(
            EvaluationCitationMarker(
                number=citation.number if citation is not None else next(invalid_numbers, None),
                chunk_id=str(citation.chunk_id) if citation is not None else None,
                valid=validation.supported,
                support_score=validation.score,
            )
        )
    markers.extend(
        EvaluationCitationMarker(
            number=next(invalid_numbers, None),
            chunk_id=None,
            valid=False,
            support_score=0.0,
        )
        for _ in range(max(0, report.citation_marker_count - len(markers)))
    )
    return markers


def _rank(candidate: RetrievalCandidate) -> tuple[int, int, str]:
    return (
        candidate.rerank_rank or 1_000_000,
        candidate.fused_rank or candidate.keyword_rank or candidate.vector_rank or 1_000_000,
        str(candidate.chunk_id),
    )
