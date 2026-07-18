from hashlib import sha256
from uuid import UUID, uuid4

import pytest

from app.domain.chunks import Chunk, SourcePosition
from app.domain.citations import CitationJudgeDecision
from app.domain.retrieval import RetrievalCandidate
from app.services.citation_service import CitationService
from app.services.context_builder import ContextBuilder


def _candidate(
    content: str,
    *,
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
    rank: int = 1,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        document_title="Kubernetes Deployment rollback",
        document_type="official_documentation",
        heading_path=["Deployments", "Rolling Back"],
        content=content,
        canonical_url="https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
        rerank_rank=rank,
        rerank_score=0.91,
        start_offset=100,
        end_offset=100 + len(content),
    )


@pytest.mark.asyncio
async def test_citations_resolve_only_supplied_numbers_and_measure_claim_coverage() -> None:
    source = _candidate("A Deployment can be rolled back with kubectl rollout undo.")
    context = ContextBuilder(max_context_tokens=10_000).build([source])
    answer = (
        "A Deployment can be rolled back with kubectl rollout undo [1]. "
        "A Service exposes Pods [2]. "
        "StatefulSets always use host networking."
    )

    report = await CitationService().analyze(answer, context)

    assert report.citation_marker_count == 2
    assert report.invalid_citation_numbers == [2]
    assert len(report.citations) == 1
    assert report.citations[0].number == 1
    assert report.citations[0].chunk_id == source.chunk_id
    assert report.citations[0].url == source.canonical_url
    assert report.citations[0].quoted_text in source.content
    assert report.citations[0].valid is True
    assert report.citation_precision == pytest.approx(0.5)
    assert report.citation_completeness == pytest.approx(1 / 3)
    assert report.claim_coverage == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_valid_but_unrelated_citation_is_not_counted_as_grounding() -> None:
    context = ContextBuilder(max_context_tokens=10_000).build(
        [_candidate("Deployments support rolling updates.")]
    )

    report = await CitationService().analyze(
        "PostgreSQL uses MVCC for transaction isolation [1].",
        context,
    )

    assert report.citations[0].valid is False
    assert report.citation_precision == 0
    assert report.citation_completeness == 1
    assert report.claim_coverage == 0


@pytest.mark.asyncio
async def test_answer_cannot_forge_source_number_url_or_chunk_identity() -> None:
    source = _candidate("Rollback a Deployment with kubectl rollout undo.")
    context = ContextBuilder(max_context_tokens=10_000).build([source])

    report = await CitationService().analyze(
        "A fabricated source is at https://evil.example/fake [999]. "
        "Rollback uses kubectl rollout undo [1].",
        context,
    )

    assert report.invalid_citation_numbers == [999]
    assert len(report.citations) == 1
    citation = report.citations[0]
    assert citation.number == 1
    assert citation.chunk_id == source.chunk_id
    assert citation.document_id == source.document_id
    assert citation.url == source.canonical_url
    assert "evil.example" not in (citation.url or "")


@pytest.mark.asyncio
async def test_citation_after_sentence_and_multiple_markers_attach_to_adjacent_claim() -> None:
    first = _candidate("Deployment rollback uses kubectl rollout undo.", rank=1)
    second = _candidate("The rollout history lists Deployment revisions.", rank=2)
    context = ContextBuilder(max_context_tokens=10_000).build([first, second])

    report = await CitationService(support_threshold=0.1).analyze(
        "Deployment rollback uses kubectl rollout undo. [1][2]",
        context,
    )

    assert len(report.claims) == 1
    assert report.claims[0].citation_numbers == [1, 2]
    assert report.citation_marker_count == 2


@pytest.mark.asyncio
async def test_merged_source_preserves_each_supporting_chunk_mapping() -> None:
    document_id = uuid4()
    first_content = "Deployment rollback uses kubectl rollout undo."
    second_content = "Deployment revision history uses kubectl rollout history."
    first = _candidate(
        first_content,
        chunk_id=UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),
        document_id=document_id,
        rank=1,
    ).model_copy(
        update={
            "start_offset": 0,
            "end_offset": len(first_content),
            "metadata": {"chunk_index": 0},
        }
    )
    second = _candidate(
        second_content,
        chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
        document_id=document_id,
        rank=2,
    ).model_copy(
        update={
            "start_offset": len(first_content) + 1,
            "end_offset": len(first_content) + 1 + len(second_content),
            "metadata": {"chunk_index": 1},
        }
    )
    context = ContextBuilder(max_context_tokens=10_000).build([first, second])

    assert len(context.sources) == 1
    report = await CitationService(support_threshold=0.1).analyze(
        "Deployment rollback uses kubectl rollout undo [1]. "
        "Deployment revision history uses kubectl rollout history [1].",
        context,
    )

    assert [(item.number, item.chunk_id) for item in report.citations] == [
        (1, first.chunk_id),
        (1, second.chunk_id),
    ]


@pytest.mark.asyncio
async def test_citation_like_text_inside_fenced_code_is_not_parsed() -> None:
    context = ContextBuilder(max_context_tokens=10_000).build(
        [_candidate("Deployment rollback uses kubectl rollout undo.")]
    )

    report = await CitationService().analyze(
        "Example syntax:\n```text\nvalue[1]\n```",
        context,
    )

    assert report.citation_marker_count == 0
    assert report.citations == []
    assert report.invalid_citation_numbers == []


class _SemanticValidator:
    def __init__(self) -> None:
        self.calls = 0

    async def validate(
        self,
        *,
        claim: str,
        evidence: str,
        title: str,
        section: str,
    ) -> CitationJudgeDecision:
        self.calls += 1
        assert claim
        assert evidence
        assert title
        assert section
        return CitationJudgeDecision(
            supported=True,
            score=1.0,
            reason="the evidence semantically entails the claim",
        )


@pytest.mark.asyncio
async def test_optional_semantic_validation_port_can_confirm_paraphrase() -> None:
    validator = _SemanticValidator()
    context = ContextBuilder(max_context_tokens=10_000).build(
        [_candidate("Use the undo command to restore an earlier rollout revision.")]
    )

    report = await CitationService(
        validator=validator,
        support_threshold=0.2,
    ).analyze("kubectl rollout undo returns a workload to a prior revision [1].", context)

    assert validator.calls == 1
    assert report.citations[0].valid is True
    assert report.validations[0].reason == "the evidence semantically entails the claim"


def _chunk(*, chunk_index: int, document_id: UUID) -> Chunk:
    content = f"chunk {chunk_index}"
    return Chunk(
        id=uuid4(),
        document_id=document_id,
        chunk_index=chunk_index,
        heading_path=["Section"],
        content=content,
        contextualized_content=content,
        token_count=2,
        content_hash=sha256(content.encode()).hexdigest(),
        position=SourcePosition(start_offset=chunk_index * 10, end_offset=chunk_index * 10 + 7),
        document_title="Document",
        canonical_url="https://example.com/document",
        document_type="official_documentation",
    )


def test_citation_detail_contract_contains_document_local_neighbours() -> None:
    document_id = uuid4()
    previous = _chunk(chunk_index=0, document_id=document_id)
    cited = _chunk(chunk_index=1, document_id=document_id)
    following = _chunk(chunk_index=2, document_id=document_id)

    detail = CitationService.citation_detail(
        cited,
        previous_chunk=previous,
        next_chunk=following,
        document_metadata={"version": "v1.34"},
    )

    assert detail.previous_chunk == previous
    assert detail.next_chunk == following
    assert detail.source_url == cited.canonical_url
    assert detail.document_metadata["version"] == "v1.34"


def test_citation_detail_rejects_cross_document_neighbour() -> None:
    cited = _chunk(chunk_index=1, document_id=uuid4())
    wrong_document = _chunk(chunk_index=0, document_id=uuid4())

    with pytest.raises(ValueError, match="cited document"):
        CitationService.citation_detail(cited, previous_chunk=wrong_document)
