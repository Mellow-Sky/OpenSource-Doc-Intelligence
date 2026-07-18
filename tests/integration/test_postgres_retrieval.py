from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.db.models.source_document import Chunk, Document, Source
from app.domain.retrieval import QueryFilters
from app.repositories.retrieval_repository import RetrievalRepository

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is required for PostgreSQL retrieval integration tests",
    ),
]


@pytest.mark.asyncio
async def test_postgres_fts_and_pgvector_retrieve_the_same_active_fixture() -> None:
    """Exercise the real trigger, GIN query, JSONB filters, and cosine operator."""
    assert TEST_DATABASE_URL is not None
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        try:
            async with AsyncSession(bind=connection, expire_on_commit=False) as session:
                source = Source(
                    id=uuid4(),
                    name=f"retrieval-fixture-{uuid4()}",
                    source_type="github_repository",
                    enabled=True,
                )
                document = Document(
                    id=uuid4(),
                    source_id=source.id,
                    external_id=f"fixture/{uuid4()}.md",
                    document_type="official_documentation",
                    title="Deployment rollout",
                    content_hash="a" * 64,
                    source_version="v1.34",
                    metadata_={"kind": "Deployment", "api_group": "apps", "version": "v1"},
                    status="active",
                )
                embedding = [1.0, *([0.0] * 1023)]
                chunk = Chunk(
                    id=uuid4(),
                    document_id=document.id,
                    chunk_index=0,
                    document_title=document.title,
                    heading_path=["Workloads", "Deployments"],
                    content="A Deployment performs a rolling update and can be rolled back.",
                    contextualized_content=(
                        "Deployment rollout Workloads Deployments A Deployment performs "
                        "a rolling update and can be rolled back."
                    ),
                    token_count=15,
                    content_hash="b" * 64,
                    start_offset=0,
                    end_offset=65,
                    metadata_={"kind": "Deployment", "api_group": "apps"},
                    embedding=embedding,
                    embedding_model="fixture-model",
                    embedding_dimension=1024,
                )
                session.add_all((source, document, chunk))
                await session.flush()

                repository = RetrievalRepository(session)
                filters = QueryFilters(
                    kinds=["Deployment"],
                    api_groups=["apps"],
                    metadata={"api_versions": ["v1"]},
                )
                keyword = await repository.keyword_search(
                    "Deployment rolling update",
                    filters=filters,
                    limit=5,
                )
                vector = await repository.vector_search(
                    embedding,
                    filters=filters,
                    limit=5,
                    embedding_model="fixture-model",
                )

                assert [candidate.chunk_id for candidate in keyword] == [chunk.id]
                assert [candidate.chunk_id for candidate in vector] == [chunk.id]
                assert vector[0].vector_score == pytest.approx(1.0)
        finally:
            if transaction.is_active:
                await transaction.rollback()
    await engine.dispose()
