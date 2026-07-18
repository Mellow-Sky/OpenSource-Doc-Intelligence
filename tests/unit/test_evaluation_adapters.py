"""Regression tests for lossless chat-to-evaluation trace conversion."""

from uuid import uuid4

import pytest

from app.domain.chat import ChatResult
from app.domain.citations import Citation, CitationReport, CitationValidation, Claim
from app.domain.retrieval import (
    NoAnswerDecision,
    RetrievalCandidate,
    RetrievalOutcome,
    RetrievalQuery,
)
from app.services.citation_service import CitationService
from app.services.context_builder import ContextBuilder
from evaluation.adapters import response_from_chat
from evaluation.metrics.citations import citation_metrics


def _result(*, answerable: bool) -> ChatResult:
    chunk_id = uuid4()
    document_id = uuid4()
    candidate = RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title="Deployment",
        document_type="official_documentation",
        content="Deployments support rolling updates.",
        fused_rank=1,
        fused_score=0.1,
        start_offset=0,
        end_offset=36,
    )
    claim = Claim(
        text="PostgreSQL uses MVCC.",
        start_offset=0,
        end_offset=21,
        citation_numbers=[1],
    )
    invalid_citation = Citation(
        number=1,
        chunk_id=chunk_id,
        document_id=document_id,
        title="Deployment",
        section="",
        quoted_text=candidate.content,
        document_type=candidate.document_type,
        score=0.1,
        start_offset=0,
        end_offset=36,
        valid=False,
        validation_score=0.1,
    )
    report = CitationReport(
        claims=[claim],
        citations=[invalid_citation],
        validations=[
            CitationValidation(
                claim=claim,
                citation=invalid_citation,
                supported=False,
                score=0.1,
                reason="unrelated evidence",
            )
        ],
        invalid_citation_numbers=[99],
        citation_marker_count=2,
        citation_precision=0,
        citation_recall=0,
        claim_coverage=0,
        citation_correctness=0.05,
        citation_completeness=1,
    )
    return ChatResult(
        request_id=uuid4(),
        conversation_id=uuid4(),
        message_id=uuid4(),
        original_query="question",
        rewritten_query="question",
        answer=(
            "PostgreSQL uses MVCC [1][99]."
            if answerable
            else "The current knowledge base does not contain enough evidence."
        ),
        answerable=answerable,
        confidence=0.1,
        # Production ChatService filters invalid citations from this public list.
        citations=[],
        retrieval=RetrievalOutcome(
            query=RetrievalQuery(original="question", rewritten="question"),
            candidates=[candidate],
            trace_candidates=[candidate],
            keyword_count=1,
            vector_count=0,
            reranked_count=0,
        ),
        no_answer=NoAnswerDecision(
            answerable=answerable,
            confidence=0.1,
            reason="test",
            retrieval_confidence=0.1,
            evidence_sufficiency_score=0.1,
        ),
        citation_report=report,
    )


def test_adapter_retains_invalid_marker_outcomes_for_precision() -> None:
    response = response_from_chat(_result(answerable=True))

    assert len(response.citations.citation_ids) == 1
    assert response.citations.validity == [False, False]
    assert response.citations.support_scores == [0.1, 0.0]
    assert response.citations.claim_requires_citation == [True]
    assert response.citations.claim_supported == [False]


def test_adapter_preserves_a_zero_reranker_score_without_fusion_substitution() -> None:
    result = _result(answerable=True)
    candidate = result.retrieval.candidates[0].model_copy(update={"rerank_score": 0.0})
    result = result.model_copy(
        update={
            "retrieval": result.retrieval.model_copy(
                update={"candidates": [candidate], "trace_candidates": [candidate]}
            )
        }
    )

    response = response_from_chat(result)

    assert response.evidence[0].score == 0.0


@pytest.mark.asyncio
async def test_valid_and_forged_markers_produce_half_precision_from_real_report() -> None:
    base = _result(answerable=True)
    candidate = base.retrieval.candidates[0]
    answer = (
        "Deployments support rolling updates [1]. "
        "A fabricated source proves PostgreSQL uses MVCC [99]."
    )
    report = await CitationService().analyze(
        answer,
        ContextBuilder(max_context_tokens=10_000).build([candidate]),
    )
    result = base.model_copy(
        update={
            "answer": answer,
            "citations": [citation for citation in report.citations if citation.valid],
            "citation_report": report,
        }
    )

    # The public response contains only the valid citation. Evaluation must use
    # both marker outcomes from CitationReport or it would report 1.0 precision.
    assert len(result.citations) == 1
    response = response_from_chat(result)
    metrics = citation_metrics(
        citation_validity=response.citations.validity,
        support_scores=response.citations.support_scores,
        claim_requires_citation=response.citations.claim_requires_citation,
        claim_supported=response.citations.claim_supported,
    )

    assert [marker.number for marker in response.citations.markers] == [1, 99]
    assert [marker.chunk_id for marker in response.citations.markers] == [
        str(candidate.chunk_id),
        None,
    ]
    assert response.citations.validity == [True, False]
    assert response.citations.support_scores == [1.0, 0.0]
    assert metrics.precision == pytest.approx(0.5)


def test_adapter_does_not_score_a_hidden_rejected_draft_against_refusal() -> None:
    response = response_from_chat(_result(answerable=False))

    assert response.citations.citation_ids == []
    assert response.citations.validity == []
    assert response.citations.claim_requires_citation == []
