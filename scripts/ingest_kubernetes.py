"""Synchronize persisted or YAML-configured Kubernetes knowledge sources."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy.dialects.postgresql import insert

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.db.models.source_document import Source
from app.db.session import Database
from app.ingestion.chunkers import ChunkingConfig, create_chunk_token_counter
from app.providers.base import EmbeddingProvider
from app.providers.factory import create_embedding_provider
from app.repositories.source_repository import SourceRepository
from app.schemas.ingestion import SourceConfig, load_source_configs
from app.services.ingestion_service import IngestionService, SyncResult
from app.services.pricing_service import PricingCatalog


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--source-id", type=UUID, help="Synchronize one persisted source")
    selection.add_argument("--all", action="store_true", help="Synchronize all enabled sources")
    selection.add_argument("--config", type=Path, help="Load and synchronize a source YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and plan without writes")
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Do not soft-delete documents missing from complete snapshots",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/ingestion"),
        help="Safe parent directory for Git checkouts",
    )
    return parser.parse_args(argv)


async def run(
    args: argparse.Namespace,
    *,
    settings: Settings | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> int:
    """Execute selected sources and return a process-style status code."""

    runtime = settings or get_settings()
    config_path: Path | None = args.config
    if args.source_id is None and not args.all and config_path is None:
        config_path = Path("config/sources.yaml")
    runtime_embedding = embedding_provider
    if not args.dry_run and runtime_embedding is None:
        runtime_embedding = create_embedding_provider(runtime)

    database = Database(runtime)
    failures = 0
    try:
        token_counter = await create_chunk_token_counter(runtime, runtime_embedding)
        service = IngestionService(
            database.session_factory,
            embedding_provider=runtime_embedding,
            embedding_dimension=runtime.embedding_dimension,
            embedding_batch_size=runtime.embedding_batch_size,
            cache_root=args.cache_dir,
            github_token=runtime.github_token,
            chunking_config=ChunkingConfig(
                target_tokens=runtime.chunk_target_tokens,
                max_tokens=runtime.chunk_max_tokens,
                overlap_tokens=runtime.chunk_overlap_tokens,
                min_tokens=runtime.chunk_min_tokens,
            ),
            token_counter=token_counter,
            pricing_catalog=PricingCatalog.from_file(runtime.pricing_config_path),
        )
        if config_path is not None:
            configs = load_source_configs(config_path)
            if args.dry_run:
                for config in (source for source in configs if source.enabled):
                    source = _transient_source(config)
                    failures += await _preview_one(service, source)
                return 1 if failures else 0
            persisted_sources = await _upsert_configs(database, configs)
            sources = [source for source in persisted_sources if source.enabled]
        else:
            async with database.session_factory() as session:
                repository = SourceRepository(session)
                if args.source_id is not None:
                    found_source = await repository.get(args.source_id)
                    if found_source is None:
                        raise ConfigurationError(f"Source not found: {args.source_id}")
                    sources = [found_source]
                else:
                    sources = await repository.list_enabled()

        for source in sources:
            try:
                result = await service.sync_source(
                    source.id,
                    dry_run=args.dry_run,
                    allow_delete_missing=not args.no_delete,
                )
            except Exception as exc:
                failures += 1
                _print_error(source, exc)
            else:
                _print_result(source, result)
    finally:
        await database.close()
        if runtime_embedding is not None:
            await runtime_embedding.close()
    return 1 if failures else 0


async def _preview_one(service: IngestionService, source: Source) -> int:
    try:
        result = await service.preview_source(source)
    except Exception as exc:
        _print_error(source, exc)
        return 1
    _print_result(source, result)
    return 0


async def _upsert_configs(database: Database, configs: list[SourceConfig]) -> list[Source]:
    if not configs:
        return []
    values = [_source_values(config) for config in configs]
    async with database.session_factory.begin() as session:
        insert_statement = insert(Source).values(values)
        statement = insert_statement.on_conflict_do_update(
            index_elements=[Source.name],
            set_={
                "source_type": insert_statement.excluded.source_type,
                "base_url": insert_statement.excluded.base_url,
                "repository": insert_statement.excluded.repository,
                "branch": insert_statement.excluded.branch,
                "enabled": insert_statement.excluded.enabled,
                "config": insert_statement.excluded.config,
            },
        ).returning(Source)
        result = await session.execute(statement)
        return list(result.scalars())


def _source_values(source: SourceConfig) -> dict[str, Any]:
    return {
        "name": source.name,
        "source_type": source.source_type,
        "base_url": str(source.base_url) if source.base_url else None,
        "repository": source.repository,
        "branch": source.branch,
        "enabled": source.enabled,
        "config": source.config,
    }


def _transient_source(config: SourceConfig) -> Source:
    return Source(
        id=uuid5(NAMESPACE_URL, f"opensource-doc-intelligence:{config.name}"),
        **_source_values(config),
    )


def _print_result(source: Source, result: SyncResult) -> None:
    print(
        json.dumps(
            {
                "source_id": str(source.id),
                "request_id": str(result.request_id),
                "source": source.name,
                "dry_run": result.dry_run,
                "complete_snapshot": result.complete_snapshot,
                "stats": result.stats.model_dump(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _print_error(source: Source, exc: Exception) -> None:
    # Provider/loaders already redact credentials; keep CLI output bounded and type-led.
    message = str(exc).replace("\n", " ")[:500]
    print(
        json.dumps(
            {
                "source_id": str(source.id),
                "source": source.name,
                "error_type": type(exc).__name__,
                "error": message,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
