"""Idempotently seed the configured Kubernetes knowledge sources."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert

from app.core.config import get_settings
from app.db.models import Chunk, Document, DocumentVersion, Source
from app.db.session import Database
from app.domain.usage import EmbeddingBatchUsage
from app.ingestion.deduplication import normalized_content_hash, simhash64
from app.providers.base import EmbeddingProvider
from app.providers.factory import create_embedding_provider
from app.repositories.usage_repository import UsageRepository
from app.schemas.ingestion import SourceConfig, load_source_configs
from app.services.pricing_service import PricingCatalog
from app.services.usage_service import UsageService
from evaluation.dataset_builder import SourceChunk, load_source_chunks

_FIXTURE_SOURCE_ID = uuid5(NAMESPACE_URL, "opensource-doc-intelligence:evaluation-fixtures")


def _source_values(source: SourceConfig) -> dict[str, Any]:
    return {
        "name": source.name,
        "source_type": source.source_type,
        "base_url": str(source.base_url) if source.base_url is not None else None,
        "repository": source.repository,
        "branch": source.branch,
        "enabled": source.enabled,
        "config": source.config,
    }


async def seed(path: Path) -> int:
    """Upsert every source by name in one transaction and return the source count."""
    sources = load_source_configs(path)
    if not sources:
        return 0

    database = Database(get_settings())
    try:
        async with database.session_factory.begin() as session:
            statement = insert(Source).values([_source_values(source) for source in sources])
            statement = statement.on_conflict_do_update(
                index_elements=[Source.name],
                set_={
                    "source_type": statement.excluded.source_type,
                    "base_url": statement.excluded.base_url,
                    "repository": statement.excluded.repository,
                    "branch": statement.excluded.branch,
                    "enabled": statement.excluded.enabled,
                    "config": statement.excluded.config,
                },
            )
            await session.execute(statement)
    finally:
        await database.close()
    return len(sources)


async def seed_evaluation_fixtures(
    path: Path,
    *,
    provider: EmbeddingProvider | None,
) -> tuple[int, int]:
    """Idempotently seed the portable evaluation source catalog and optional vectors."""
    chunks = load_source_chunks(path)
    grouped: defaultdict[str, list[SourceChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id or chunk.chunk_id].append(chunk)

    settings = get_settings()
    if provider is not None and provider.dimension != settings.embedding_dimension:
        msg = "Fixture embedding provider dimension does not match EMBEDDING_DIMENSION"
        raise ValueError(msg)
    contexts = [
        _contextualized(chunk) for document_chunks in grouped.values() for chunk in document_chunks
    ]
    embedding_vectors: list[list[float] | None] = []
    embedding_model: str | None = None
    embedding_usage: list[EmbeddingBatchUsage] = []
    request_id = uuid4()
    if provider is None:
        embedding_vectors.extend([None] * len(contexts))
    else:
        started = time.perf_counter()
        response = await provider.embed(contexts)
        latency_ms = (time.perf_counter() - started) * 1000
        if len(response.vectors) != len(contexts):
            msg = "Fixture embedding provider returned an incompatible batch"
            raise ValueError(msg)
        embedding_vectors.extend(response.vectors)
        embedding_model = response.model
        embedding_usage.append(
            EmbeddingBatchUsage(
                model=response.model,
                provider=provider.name,
                input_text_count=len(contexts),
                input_character_count=sum(len(context) for context in contexts),
                prompt_tokens=response.usage.prompt_tokens,
                latency_ms=latency_ms,
            )
        )

    now = datetime.now(UTC)
    database = Database(settings)
    try:
        async with database.session_factory.begin() as session:
            source_id = await session.scalar(
                insert(Source)
                .values(
                    id=_FIXTURE_SOURCE_ID,
                    name="kubernetes-evaluation-fixtures",
                    source_type="evaluation_fixture",
                    base_url="fixture://kubernetes",
                    enabled=True,
                    config={"catalog": str(path), "generated": False},
                )
                .on_conflict_do_update(
                    index_elements=[Source.name],
                    set_={
                        "source_type": "evaluation_fixture",
                        "base_url": "fixture://kubernetes",
                        "enabled": True,
                        "config": {"catalog": str(path), "generated": False},
                        "updated_at": now,
                    },
                )
                .returning(Source.id)
            )
            if source_id is None:
                raise RuntimeError("Evaluation fixture source upsert returned no identifier")
            vector_index = 0
            active_document_ids: list[UUID] = []
            active_chunk_ids: list[UUID] = []
            for external_id, document_chunks in grouped.items():
                document_id = uuid5(
                    NAMESPACE_URL,
                    f"opensource-doc-intelligence:evaluation-document:{external_id}",
                )
                active_document_ids.append(document_id)
                raw_content = "\n\n".join(chunk.content for chunk in document_chunks)
                document_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
                primary = document_chunks[0]
                document_metadata = {
                    "source_type": primary.source_type,
                    "evaluation_fixture": True,
                    "deduplication_fingerprint": {
                        "normalized_sha256": normalized_content_hash(raw_content),
                        "simhash64": simhash64(raw_content),
                    },
                }
                await session.execute(
                    insert(Document)
                    .values(
                        id=document_id,
                        source_id=source_id,
                        external_id=external_id,
                        document_type=primary.document_type,
                        title=primary.title,
                        canonical_url=primary.url,
                        repository_path=None,
                        source_version=str(primary.metadata.get("source_version", "fixture-v1")),
                        language="en",
                        content_hash=document_hash,
                        metadata=document_metadata,
                        status="active",
                        first_seen_at=now,
                        last_seen_at=now,
                        indexed_at=now,
                        deleted_at=None,
                    )
                    .on_conflict_do_update(
                        constraint="uq_documents_source_external",
                        set_={
                            "document_type": primary.document_type,
                            "title": primary.title,
                            "canonical_url": primary.url,
                            "content_hash": document_hash,
                            "metadata": document_metadata,
                            "status": "active",
                            "last_seen_at": now,
                            "indexed_at": now,
                            "deleted_at": None,
                            "updated_at": now,
                        },
                    )
                )
                await session.execute(
                    insert(DocumentVersion)
                    .values(
                        id=uuid5(
                            NAMESPACE_URL,
                            f"opensource-doc-intelligence:evaluation-version:{external_id}:{document_hash}",
                        ),
                        document_id=document_id,
                        source_version="fixture-v1",
                        content_hash=document_hash,
                        raw_content=raw_content,
                        parsed_content=raw_content,
                    )
                    .on_conflict_do_nothing(constraint="uq_document_versions_document_hash")
                )
                offset = 0
                for chunk_index, chunk in enumerate(document_chunks):
                    chunk_id = uuid5(
                        NAMESPACE_URL,
                        f"opensource-doc-intelligence:evaluation-chunk:{chunk.chunk_id}",
                    )
                    active_chunk_ids.append(chunk_id)
                    content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
                    contextualized = contexts[vector_index]
                    vector = embedding_vectors[vector_index]
                    vector_index += 1
                    values = {
                        "id": chunk_id,
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "parent_chunk_id": None,
                        "document_title": chunk.title,
                        "heading_path": [
                            part.strip() for part in chunk.section.split(" > ") if part
                        ],
                        "content": chunk.content,
                        "contextualized_content": contextualized,
                        "token_count": max(1, len(contextualized.split())),
                        "content_hash": content_hash,
                        "start_offset": offset,
                        "end_offset": offset + len(chunk.content),
                        "start_line": 1,
                        "end_line": max(1, chunk.content.count("\n") + 1),
                        "metadata": {
                            **chunk.metadata,
                            "evaluation_chunk_id": chunk.chunk_id,
                            "evaluation_fixture": True,
                        },
                        "embedding": vector,
                        "embedding_model": embedding_model,
                        "embedding_dimension": provider.dimension if provider is not None else None,
                        "deleted_at": None,
                    }
                    await session.execute(
                        insert(Chunk)
                        .values(**values)
                        .on_conflict_do_update(
                            constraint="uq_chunks_document_index",
                            set_={
                                key: value
                                for key, value in values.items()
                                if key not in {"id", "document_id", "chunk_index"}
                            }
                            | {"updated_at": now},
                        )
                    )
                    offset += len(chunk.content) + 2
            await session.execute(
                update(Document)
                .where(
                    Document.source_id == source_id,
                    Document.id.not_in(active_document_ids),
                )
                .values(status="deleted", deleted_at=now, updated_at=now)
            )
            await session.execute(
                update(Chunk)
                .where(
                    Chunk.document_id.in_(active_document_ids),
                    Chunk.id.not_in(active_chunk_ids),
                )
                .values(deleted_at=now, updated_at=now)
            )
            await UsageService(
                UsageRepository(session),
                PricingCatalog.from_file(settings.pricing_config_path),
            ).record_embedding_batches(
                request_id=request_id,
                operation="seed_embedding",
                batches=embedding_usage,
                created_at=now,
            )
    finally:
        await database.close()
    return len(grouped), len(chunks)


def _contextualized(chunk: SourceChunk) -> str:
    section = f"\nSection: {chunk.section}" if chunk.section else ""
    return f"Document: {chunk.title}{section}\n\n{chunk.content}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("config/sources.yaml"),
        help="Path to the source configuration YAML",
    )
    parser.add_argument(
        "--evaluation-fixtures",
        type=Path,
        help="Also seed a portable evaluation source catalog",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Leave evaluation fixture vectors empty for keyword-only smoke tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = asyncio.run(seed(args.sources))
    print(f"Seeded {count} knowledge sources")
    if args.evaluation_fixtures is not None:
        provider = None if args.no_embeddings else create_embedding_provider(get_settings())
        try:
            documents, chunks = asyncio.run(
                seed_evaluation_fixtures(args.evaluation_fixtures, provider=provider)
            )
        finally:
            if provider is not None:
                asyncio.run(provider.close())
        print(f"Seeded {documents} evaluation documents and {chunks} chunks")


if __name__ == "__main__":
    main()
