"""Core provenance and retrieval model invariants."""

from hashlib import sha256
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.domain.chunks import ChunkDraft, SourcePosition
from app.domain.documents import RawDocument
from app.domain.retrieval import RetrievalCandidate


def test_raw_document_has_uniform_loader_contract() -> None:
    document = RawDocument(
        source_type="github_repo",
        external_id="docs/concepts/workloads/controllers/deployment.md",
        title="Deployments",
        content="# Deployments\n",
        canonical_url="https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
        source_version="abc123",
        metadata={"repository_path": "docs/concepts/workloads/controllers/deployment.md"},
    )

    assert document.metadata["repository_path"].endswith("deployment.md")


def test_chunk_draft_preserves_provenance() -> None:
    content = "A Deployment manages a replicated application."
    draft = ChunkDraft(
        chunk_index=0,
        heading_path=["Workloads", "Deployments"],
        content=content,
        contextualized_content=f"Deployments > Workloads\n{content}",
        token_count=9,
        content_hash=sha256(content.encode()).hexdigest(),
        position=SourcePosition(start_offset=10, end_offset=55, start_line=2, end_line=2),
    )

    assert draft.position.start_line == 2
    assert draft.heading_path[-1] == "Deployments"


def test_invalid_source_offsets_are_rejected() -> None:
    with pytest.raises(ValidationError, match="end_offset"):
        SourcePosition(start_offset=12, end_offset=2)


def test_retrieval_candidate_keeps_all_rank_channels() -> None:
    candidate = RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_title="Deployment",
        document_type="official_documentation",
        content="Use kubectl rollout undo.",
        keyword_rank=3,
        vector_rank=1,
        fused_rank=1,
        fused_score=0.032,
    )

    assert candidate.keyword_rank == 3
    assert candidate.vector_rank == 1
    assert candidate.fused_score == pytest.approx(0.032)
