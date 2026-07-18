from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models.ingestion import SyncCursor
from app.db.models.source_document import Chunk, Document, DocumentVersion, Source
from app.db.models.usage import UsageRecord
from app.domain.documents import RawDocument
from app.ingestion.chunkers import ChunkingConfig, RegexTokenCounter
from app.ingestion.factory import LoaderSpec
from app.ingestion.loaders.base import DocumentLoader
from app.providers.base import EmbeddingProvider, EmbeddingResponse, TokenUsage
from app.services.ingestion_service import IngestionService

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
EMBEDDING_DIMENSION = 1024

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is required for PostgreSQL ingestion E2E tests",
    ),
]


class _MutableLoader(DocumentLoader):
    """Return a caller-controlled repository snapshot without network access."""

    def __init__(self, documents: Sequence[RawDocument]) -> None:
        self.documents = list(documents)

    async def load(self) -> list[RawDocument]:
        return list(self.documents)


class _RecordingEmbeddingProvider(EmbeddingProvider):
    """Produce valid deterministic vectors while exposing the exact reindex workload."""

    def __init__(self, marker: str) -> None:
        self.calls: list[list[str]] = []
        self._model = f"ingestion-e2e-{marker}"

    @property
    def name(self) -> str:
        return "ingestion-e2e-deterministic"

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    async def healthcheck(self) -> None:
        return None

    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        batch = list(texts)
        self.calls.append(batch)
        vector = [1.0, *([0.0] * (EMBEDDING_DIMENSION - 1))]
        return EmbeddingResponse(
            vectors=[list(vector) for _ in batch],
            model=self.model,
            dimension=self.dimension,
            usage=TokenUsage(prompt_tokens=sum(len(item.split()) for item in batch)),
        )

    @property
    def embedded_text_count(self) -> int:
        return sum(len(batch) for batch in self.calls)


def _raw_document(*, commit_sha: str, rollback_text: str) -> RawDocument:
    return RawDocument(
        source_type="github_repository",
        external_id="docs/deployment.md",
        title="Deployment operations",
        content=(
            "# Deployment operations\n\n"
            "A Kubernetes Deployment manages replicated Pods and supports controlled "
            "rolling updates.\n\n"
            "## Rollout status\n\n"
            "Use kubectl rollout status deployment/web to inspect progress and wait "
            "for completion.\n\n"
            "## Rollback\n\n"
            f"{rollback_text}\n"
        ),
        canonical_url="https://example.test/docs/deployment",
        source_version=commit_sha,
        metadata={
            "commit_sha": commit_sha,
            "document_type": "official_documentation",
            "repository_path": "docs/deployment.md",
            "source_format": "markdown",
        },
    )


def _alembic_head() -> str:
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    heads = ScriptDirectory.from_config(config).get_heads()
    assert len(heads) == 1, f"expected one Alembic head, found {heads}"
    return heads[0]


async def _assert_database_ready(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        revision = await session.scalar(text("SELECT version_num FROM alembic_version"))
        vector_extension = await session.scalar(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
    assert revision == _alembic_head(), "TEST_DATABASE_URL must be upgraded to Alembic head"
    assert vector_extension == "vector"


@pytest.mark.asyncio
async def test_loader_to_postgres_sync_is_idempotent_and_reembeds_only_changes(
    tmp_path: Path,
) -> None:
    """Exercise the real parser, cleaner, chunker, pgvector, FTS trigger, and cursor."""

    assert TEST_DATABASE_URL is not None
    marker = uuid4().hex
    source_id = uuid4()
    first_commit = "a" * 40
    second_commit = "b" * 40
    loader = _MutableLoader(
        [
            _raw_document(
                commit_sha=first_commit,
                rollback_text=(
                    "Use kubectl rollout undo deployment/web to return the Deployment "
                    "to the previous revision."
                ),
            )
        ]
    )
    provider = _RecordingEmbeddingProvider(marker)
    engine = create_async_engine(TEST_DATABASE_URL)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    spec = LoaderSpec(
        loader=loader,
        complete_snapshot=True,
        cursor_type="repository_commit_sha",
    )

    try:
        await _assert_database_ready(session_factory)
        async with session_factory.begin() as session:
            session.add(
                Source(
                    id=source_id,
                    name=f"ingestion-e2e-{marker}",
                    source_type="github_repository",
                    repository="example/project",
                    branch="main",
                    enabled=True,
                    config={"deduplication_threshold": 0.95},
                )
            )

        service = IngestionService(
            session_factory,
            embedding_provider=provider,
            embedding_dimension=EMBEDDING_DIMENSION,
            embedding_batch_size=2,
            cache_root=tmp_path / "ingestion-cache",
            chunking_config=ChunkingConfig(
                target_tokens=50,
                max_tokens=80,
                overlap_tokens=8,
                min_tokens=5,
            ),
            token_counter=RegexTokenCounter(),
        )

        first = await service.sync_source(source_id, loader_spec=spec)
        assert first.stats.created == 1
        assert first.stats.chunks_created == 3
        assert first.stats.updated == 0
        assert first.checkpoint is not None
        assert first.checkpoint.cursor_value == first_commit
        assert provider.embedded_text_count == 3

        async with session_factory() as session:
            document = await session.scalar(select(Document).where(Document.source_id == source_id))
            assert document is not None
            initial_chunks = list(
                await session.scalars(
                    select(Chunk)
                    .where(Chunk.document_id == document.id)
                    .order_by(Chunk.chunk_index)
                )
            )
            initial_hashes = {chunk.chunk_index: chunk.content_hash for chunk in initial_chunks}
            initial_ids = {chunk.chunk_index: chunk.id for chunk in initial_chunks}
            assert len(initial_chunks) == 3
            assert all(chunk.embedding_model == provider.model for chunk in initial_chunks)
            assert all(chunk.embedding_dimension == EMBEDDING_DIMENSION for chunk in initial_chunks)
            assert all(chunk.search_vector is not None for chunk in initial_chunks)
            assert initial_chunks[1].parent_chunk_id == initial_chunks[0].id
            assert initial_chunks[2].parent_chunk_id == initial_chunks[0].id
            first_usage = list(
                await session.scalars(
                    select(UsageRecord).where(UsageRecord.request_id == first.request_id)
                )
            )
            assert len(first_usage) == 2
            assert all(record.operation == "ingestion_embedding" for record in first_usage)
            assert sum(record.input_text_count for record in first_usage) == 3
            assert sum(record.input_character_count for record in first_usage) > 0
            assert sum(record.prompt_tokens for record in first_usage) > 0
            assert all(record.estimated_cost is None for record in first_usage)
            assert all(record.latency_ms >= 0 for record in first_usage)

        second = await service.sync_source(source_id, loader_spec=spec)
        assert second.stats.unchanged == 1
        assert second.stats.created == 0
        assert second.stats.updated == 0
        assert second.stats.chunks_created == 0
        assert second.stats.chunks_updated == 0
        assert provider.embedded_text_count == 3

        loader.documents = [
            _raw_document(
                commit_sha=second_commit,
                rollback_text=(
                    "Use kubectl rollout undo deployment/web --to-revision=2 to restore "
                    "a known earlier revision."
                ),
            )
        ]
        third = await service.sync_source(source_id, loader_spec=spec)
        assert third.stats.updated == 1
        assert third.stats.created == 0
        assert third.stats.chunks_updated == 1
        assert third.stats.chunks_created == 0
        assert third.checkpoint is not None
        assert third.checkpoint.cursor_value == second_commit
        assert provider.embedded_text_count == 4

        async with session_factory() as session:
            document = await session.scalar(select(Document).where(Document.source_id == source_id))
            assert document is not None
            chunks = list(
                await session.scalars(
                    select(Chunk)
                    .where(Chunk.document_id == document.id)
                    .order_by(Chunk.chunk_index)
                )
            )
            versions = list(
                await session.scalars(
                    select(DocumentVersion)
                    .where(DocumentVersion.document_id == document.id)
                    .order_by(DocumentVersion.created_at, DocumentVersion.id)
                )
            )
            cursor = await session.scalar(
                select(SyncCursor).where(
                    SyncCursor.source_id == source_id,
                    SyncCursor.cursor_type == "repository_commit_sha",
                )
            )

            assert document.source_version == second_commit
            assert document.status == "active"
            assert document.deleted_at is None
            assert len(versions) == 2
            assert len(chunks) == 3
            assert {chunk.chunk_index: chunk.id for chunk in chunks} == initial_ids
            changed_indices = {
                chunk.chunk_index
                for chunk in chunks
                if chunk.content_hash != initial_hashes[chunk.chunk_index]
            }
            assert changed_indices == {2}
            assert cursor is not None
            assert cursor.cursor_value == second_commit
            third_usage = list(
                await session.scalars(
                    select(UsageRecord).where(UsageRecord.request_id == third.request_id)
                )
            )
            assert len(third_usage) == 1
            assert third_usage[0].input_text_count == 1
    finally:
        async with session_factory.begin() as session:
            await session.execute(delete(UsageRecord).where(UsageRecord.model == provider.model))
            await session.execute(delete(Source).where(Source.id == source_id))
        await engine.dispose()
