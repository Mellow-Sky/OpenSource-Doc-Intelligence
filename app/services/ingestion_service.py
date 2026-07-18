"""Application service for one idempotent, citation-preserving source sync."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import ConfigurationError, IngestionError, ProviderError
from app.db.models.source_document import Source
from app.domain.chunks import ChunkDraft
from app.domain.documents import ParsedDocument, RawDocument
from app.domain.usage import EmbeddingBatchUsage
from app.ingestion.chunkers import ChunkingConfig, StructureAwareChunker, TokenCounter
from app.ingestion.cleaners import DocumentCleaner
from app.ingestion.deduplication import (
    ContentDeduplicator,
    DeduplicationCandidate,
    DeduplicationDecision,
    normalized_content_hash,
    simhash64,
)
from app.ingestion.factory import LoaderSpec, create_loader_spec
from app.ingestion.incremental import (
    ChunkAction,
    ChunkSyncPlan,
    CursorCheckpoint,
    DocumentAction,
    IncomingChunkState,
    IncomingDocumentState,
    SyncStats,
    plan_chunk_sync,
    plan_document_sync,
)
from app.ingestion.parsers import (
    HTMLDocumentParser,
    MarkdownDocumentParser,
    RSTDocumentParser,
    StructuredTextParser,
)
from app.providers.base import EmbeddingProvider
from app.repositories.chunk_repository import ChunkRepository, ChunkUpsert
from app.repositories.document_repository import (
    DocumentRepository,
    DocumentUpsert,
    DocumentVersionAppend,
    StoredDeduplicationFingerprint,
)
from app.repositories.source_repository import SourceRepository
from app.repositories.usage_repository import UsageRepository
from app.services.pricing_service import PricingCatalog
from app.services.usage_service import UsageService


class DocumentParser(Protocol):
    """Small parser port used to keep the pure preparation pipeline testable."""

    def parse(self, document: RawDocument) -> ParsedDocument:
        """Parse one raw document."""


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    """A retained document and its fully materialized retrieval chunks."""

    raw: RawDocument
    parsed: ParsedDocument
    content_hash: str
    chunks: tuple[ChunkDraft, ...]


@dataclass(frozen=True, slots=True)
class PreparedDuplicate:
    """A skipped document plus the auditable duplicate decision that excluded it."""

    document: PreparedDocument
    decision: DeduplicationDecision


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    """Pure processing output before incremental database decisions."""

    documents: tuple[PreparedDocument, ...]
    duplicates: tuple[PreparedDuplicate, ...]
    scanned: int
    duplicates_skipped: int

    @property
    def chunk_count(self) -> int:
        return sum(len(document.chunks) for document in self.documents)


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Observable result returned to API, CLI, and worker callers."""

    source_id: UUID
    stats: SyncStats
    complete_snapshot: bool
    dry_run: bool
    checkpoint: CursorCheckpoint | None = None
    request_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class _ChunkWork:
    document: PreparedDocument
    document_id: UUID
    plan: ChunkSyncPlan
    ids_by_index: Mapping[int, UUID]


@dataclass(frozen=True, slots=True)
class _EmbeddedVector:
    values: list[float]
    model: str
    dimension: int


LoaderSpecFactory = Callable[..., LoaderSpec]
UsageRepositoryFactory = Callable[[AsyncSession], UsageRepository]
_SEMANTIC_DOCUMENT_METADATA = (
    "api_group",
    "api_version",
    "document_type",
    "issue_number",
    "kind",
    "kubernetes_version",
    "labels",
    "release_version",
    "state",
    "tag_name",
    "version",
)


def prepare_documents(
    raw_documents: Sequence[RawDocument],
    *,
    chunker: StructureAwareChunker,
    cleaner: DocumentCleaner | None = None,
    markdown_parser: DocumentParser | None = None,
    html_parser: DocumentParser | None = None,
    rst_parser: DocumentParser | None = None,
    structured_parser: DocumentParser | None = None,
    deduplication_threshold: float = 0.90,
) -> PreparedBatch:
    """Parse, clean, deduplicate, and chunk an upstream batch deterministically."""

    safe_cleaner = cleaner or DocumentCleaner()
    markdown = markdown_parser or MarkdownDocumentParser()
    html = html_parser or HTMLDocumentParser()
    rst = rst_parser or RSTDocumentParser()
    structured = structured_parser or StructuredTextParser()
    deduplicator = ContentDeduplicator(deduplication_threshold)
    prepared: list[PreparedDocument] = []
    duplicates: list[PreparedDuplicate] = []
    duplicate_count = 0
    seen_external_ids: set[str] = set()

    for raw in raw_documents:
        if raw.external_id in seen_external_ids:
            raise IngestionError(f"Loader returned duplicate external_id: {raw.external_id}")
        seen_external_ids.add(raw.external_id)
        source_format = _source_format(raw)
        if source_format in {"html", "htm"}:
            parser = html
        elif source_format == "rst":
            parser = rst
        elif source_format in {"yaml", "yml", "json"}:
            parser = structured
        else:
            parser = markdown
        parsed = safe_cleaner.clean(parser.parse(raw), parser=parser)
        decision = deduplicator.check_and_add(DeduplicationCandidate.from_parsed(parsed))
        prepared_document = PreparedDocument(
            raw=raw,
            parsed=parsed,
            content_hash=_document_fingerprint(parsed),
            chunks=(),
        )
        if decision.is_duplicate:
            duplicate_count += 1
            duplicates.append(PreparedDuplicate(prepared_document, decision))
            continue
        prepared.append(
            replace(
                prepared_document,
                chunks=(
                    tuple(chunker.chunk(parsed))
                    if parsed.metadata.get("quality_status") != "too_short"
                    else ()
                ),
            )
        )

    return PreparedBatch(
        documents=tuple(prepared),
        duplicates=tuple(duplicates),
        scanned=len(raw_documents),
        duplicates_skipped=duplicate_count,
    )


async def embed_changed_chunks(
    provider: EmbeddingProvider | None,
    texts: Sequence[tuple[tuple[UUID, int], str]],
    *,
    expected_dimension: int,
    batch_size: int,
    usage_batches: list[EmbeddingBatchUsage] | None = None,
) -> dict[tuple[UUID, int], _EmbeddedVector]:
    """Embed exactly the planner-selected inputs and validate every provider claim."""

    if not texts:
        return {}
    if provider is None:
        raise ConfigurationError("Embedding provider is required for changed chunks")
    if provider.dimension != expected_dimension:
        raise ProviderError(
            "Embedding provider dimension does not match the configured database dimension"
        )
    embedded: dict[tuple[UUID, int], _EmbeddedVector] = {}
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_texts = [text for _, text in batch]
        started = time.perf_counter()
        response = await provider.embed(batch_texts)
        latency_ms = (time.perf_counter() - started) * 1000
        if response.dimension != expected_dimension:
            raise ProviderError("Embedding response reported an unexpected dimension")
        if len(response.vectors) != len(batch):
            raise ProviderError("Embedding provider returned a different vector count")
        for (key, _), vector in zip(batch, response.vectors, strict=True):
            if len(vector) != expected_dimension:
                raise ProviderError("Embedding vector has an unexpected dimension")
            embedded[key] = _EmbeddedVector(
                values=list(vector),
                model=response.model,
                dimension=response.dimension,
            )
        if usage_batches is not None:
            usage_batches.append(
                EmbeddingBatchUsage(
                    model=response.model,
                    provider=provider.name,
                    input_text_count=len(batch_texts),
                    input_character_count=sum(len(text) for text in batch_texts),
                    prompt_tokens=response.usage.prompt_tokens,
                    latency_ms=latency_ms,
                )
            )
    return embedded


class IngestionService:
    """Coordinate source loading and one atomic batch of persistence writes."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        embedding_provider: EmbeddingProvider | None,
        embedding_dimension: int,
        embedding_batch_size: int = 32,
        cache_root: Path = Path(".cache/ingestion"),
        github_token: SecretStr | str | None = None,
        chunking_config: ChunkingConfig | None = None,
        token_counter: TokenCounter | None = None,
        loader_factory: LoaderSpecFactory = create_loader_spec,
        clock: Callable[[], datetime] | None = None,
        usage_repository_factory: UsageRepositoryFactory | None = None,
        pricing_catalog: PricingCatalog | None = None,
    ) -> None:
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        self._session_factory = session_factory
        self._embedding_provider = embedding_provider
        self._embedding_dimension = embedding_dimension
        self._embedding_batch_size = embedding_batch_size
        self._cache_root = cache_root
        self._github_token = github_token
        self._chunker = StructureAwareChunker(chunking_config, token_counter)
        self._loader_factory = loader_factory
        self._clock = clock or (lambda: datetime.now(UTC))
        self._usage_repository_factory = usage_repository_factory or UsageRepository
        self._pricing = pricing_catalog or PricingCatalog()

    async def sync_source(
        self,
        source_id: UUID,
        *,
        dry_run: bool = False,
        allow_delete_missing: bool = True,
        loader_spec: LoaderSpec | None = None,
        request_id: UUID | None = None,
    ) -> SyncResult:
        """Synchronize one persisted source; writes and cursor share one transaction."""

        if not dry_run and self._embedding_provider is None:
            raise ConfigurationError("Embedding provider is not configured for ingestion")
        resolved_request_id = request_id or uuid4()
        source, cursors = await self._load_source(source_id)
        spec = loader_spec or self._loader_factory(
            source,
            cache_root=self._cache_root,
            github_token=self._github_token,
            cursors=cursors,
        )
        raw_documents = await spec.loader.load()
        batch = prepare_documents(
            raw_documents,
            chunker=self._chunker,
            deduplication_threshold=_deduplication_threshold(source),
        )
        synchronized_at = self._clock()
        checkpoint = _build_checkpoint(source.id, spec, raw_documents, synchronized_at)
        complete_snapshot = spec.complete_snapshot and allow_delete_missing

        if dry_run:
            async with self._session_factory() as session:
                return await self._apply(
                    session,
                    source,
                    raw_documents,
                    batch,
                    complete_snapshot=complete_snapshot,
                    checkpoint=checkpoint,
                    synchronized_at=synchronized_at,
                    persist=False,
                    request_id=resolved_request_id,
                )

        async with self._session_factory.begin() as session:
            return await self._apply(
                session,
                source,
                raw_documents,
                batch,
                complete_snapshot=complete_snapshot,
                checkpoint=checkpoint,
                synchronized_at=synchronized_at,
                persist=True,
                request_id=resolved_request_id,
            )

    async def preview_source(
        self,
        source: Source,
        *,
        loader_spec: LoaderSpec | None = None,
    ) -> SyncResult:
        """Run network and pure processing stages for an unpersisted configuration."""

        spec = loader_spec or self._loader_factory(
            source,
            cache_root=self._cache_root,
            github_token=self._github_token,
            cursors={},
        )
        raw_documents = await spec.loader.load()
        batch = prepare_documents(
            raw_documents,
            chunker=self._chunker,
            deduplication_threshold=_deduplication_threshold(source),
        )
        stats = SyncStats(
            scanned=batch.scanned,
            created=len(batch.documents),
            chunks_created=batch.chunk_count,
            duplicates_skipped=batch.duplicates_skipped,
        )
        return SyncResult(
            source_id=source.id,
            stats=stats,
            complete_snapshot=spec.complete_snapshot,
            dry_run=True,
            checkpoint=None,
            request_id=uuid4(),
        )

    async def _load_source(self, source_id: UUID) -> tuple[Source, dict[str, str]]:
        async with self._session_factory() as session:
            repository = SourceRepository(session)
            source = await repository.get(source_id)
            if source is None:
                raise IngestionError(f"Ingestion source not found: {source_id}")
            if not source.enabled:
                raise IngestionError(f"Ingestion source is disabled: {source_id}")
            cursor_rows = await repository.get_cursors(source_id)
            cursors = {row.cursor_type: row.cursor_value for row in cursor_rows}
            return source, cursors

    async def _apply(
        self,
        session: AsyncSession,
        source: Source,
        raw_documents: Sequence[RawDocument],
        batch: PreparedBatch,
        *,
        complete_snapshot: bool,
        checkpoint: CursorCheckpoint | None,
        synchronized_at: datetime,
        persist: bool,
        request_id: UUID,
    ) -> SyncResult:
        documents = DocumentRepository(session)
        chunks = ChunkRepository(session)
        sources = SourceRepository(session)
        if persist:
            # Separate jobs may target the same source. The transaction-scoped lock
            # prevents their document, deletion, and cursor writes from interleaving.
            await sources.acquire_sync_lock(source.id)
        stored_documents = await documents.list_states_for_source(source.id)
        seen_external_ids = {raw.external_id for raw in raw_documents}
        if not complete_snapshot:
            stored_fingerprints = await documents.list_deduplication_fingerprints(
                source.id,
                exclude_external_ids=tuple(seen_external_ids),
            )
            batch = _deduplicate_against_stored(
                batch,
                stored_fingerprints,
                threshold=_deduplication_threshold(source),
            )
        incoming = [
            IncomingDocumentState(
                source_id=source.id,
                external_id=item.raw.external_id,
                content_hash=item.content_hash,
            )
            for item in batch.documents
        ]
        document_plan = plan_document_sync(source.id, incoming, stored_documents)
        stored_by_id = {item.id: item for item in stored_documents}
        missing_document_ids = (
            [
                document_id
                for document_id in document_plan.missing_document_ids
                if stored_by_id[document_id].external_id not in seen_external_ids
            ]
            if complete_snapshot
            else []
        )
        base_values = document_plan.stats.model_dump()
        base_values.update(
            scanned=batch.scanned,
            deleted=len(missing_document_ids),
            duplicates_skipped=batch.duplicates_skipped,
        )
        stats = SyncStats.model_validate(base_values)
        prepared_by_external = {item.raw.external_id: item for item in batch.documents}
        duplicates_by_external = {item.document.raw.external_id: item for item in batch.duplicates}
        stored_by_external = {item.external_id: item for item in stored_documents}

        changed_decisions = [
            decision
            for decision in document_plan.decisions
            if decision.action in {DocumentAction.CREATE, DocumentAction.UPDATE}
        ]
        document_ids = {
            decision.incoming.external_id: decision.document_id or uuid4()
            for decision in changed_decisions
        }
        if persist and changed_decisions:
            upserts = [
                _document_upsert(
                    source.id,
                    prepared_by_external[decision.incoming.external_id],
                    document_ids[decision.incoming.external_id],
                )
                for decision in changed_decisions
            ]
            persisted_documents = await documents.upsert_many(upserts, seen_at=synchronized_at)
            persisted_ids = {item.external_id: item.id for item in persisted_documents}
            missing_returns = set(document_ids).difference(persisted_ids)
            if missing_returns:
                raise IngestionError("Document upsert did not return every changed document")
            document_ids.update(persisted_ids)
            await documents.append_versions(
                [
                    _document_version(
                        prepared_by_external[decision.incoming.external_id],
                        document_ids[decision.incoming.external_id],
                    )
                    for decision in changed_decisions
                ]
            )

        duplicate_document_ids = {
            external_id: (
                stored_by_external[external_id].id if external_id in stored_by_external else uuid4()
            )
            for external_id in duplicates_by_external
        }
        if persist and duplicates_by_external:
            duplicate_upserts = [
                _duplicate_document_upsert(
                    source.id,
                    duplicate,
                    duplicate_document_ids[external_id],
                )
                for external_id, duplicate in duplicates_by_external.items()
            ]
            persisted_duplicates = await documents.upsert_duplicates(
                duplicate_upserts,
                seen_at=synchronized_at,
            )
            persisted_duplicate_ids = {item.external_id: item.id for item in persisted_duplicates}
            missing_duplicate_returns = set(duplicate_document_ids).difference(
                persisted_duplicate_ids
            )
            if missing_duplicate_returns:
                raise IngestionError("Duplicate upsert did not return every skipped document")
            duplicate_document_ids.update(persisted_duplicate_ids)
            await documents.append_versions(
                [
                    _document_version(
                        duplicate.document,
                        duplicate_document_ids[external_id],
                    )
                    for external_id, duplicate in duplicates_by_external.items()
                ]
            )

        changed_document_ids = [
            document_ids[decision.incoming.external_id] for decision in changed_decisions
        ]
        stored_chunks_by_document = (
            await chunks.list_states_for_documents(changed_document_ids)
            if changed_document_ids
            else {}
        )
        chunk_work: list[_ChunkWork] = []
        for decision in changed_decisions:
            item = prepared_by_external[decision.incoming.external_id]
            document_id = document_ids[decision.incoming.external_id]
            stored_chunks = stored_chunks_by_document.get(document_id, [])
            incoming_chunks = [
                IncomingChunkState(
                    document_id=document_id,
                    chunk_index=draft.chunk_index,
                    # The embedding input includes title and heading context.
                    content_hash=normalized_content_hash(draft.contextualized_content),
                )
                for draft in item.chunks
            ]
            chunk_plan = plan_chunk_sync(document_id, incoming_chunks, stored_chunks)
            ids_by_index = _chunk_ids(chunk_plan)
            chunk_work.append(
                _ChunkWork(
                    document=item,
                    document_id=document_id,
                    plan=chunk_plan,
                    ids_by_index=ids_by_index,
                )
            )
            stats = stats.merge(chunk_plan.stats)

        if not persist:
            return SyncResult(
                source_id=source.id,
                stats=stats,
                complete_snapshot=complete_snapshot,
                dry_run=True,
                checkpoint=None,
                request_id=request_id,
            )

        embedding_inputs = _embedding_inputs(chunk_work)
        embedding_usage: list[EmbeddingBatchUsage] = []
        embedded = await embed_changed_chunks(
            self._embedding_provider,
            embedding_inputs,
            expected_dimension=self._embedding_dimension,
            batch_size=self._embedding_batch_size,
            usage_batches=embedding_usage,
        )
        chunk_upserts = _chunk_upserts(chunk_work, embedded)
        await chunks.upsert_many(chunk_upserts, written_at=synchronized_at)
        removed_chunk_ids = [
            chunk_id for work in chunk_work for chunk_id in work.plan.removed_chunk_ids
        ]
        await chunks.soft_delete(removed_chunk_ids, deleted_at=synchronized_at)
        await UsageService(
            self._usage_repository_factory(session),
            self._pricing,
        ).record_embedding_batches(
            request_id=request_id,
            operation="ingestion_embedding",
            batches=embedding_usage,
            created_at=synchronized_at,
        )

        unchanged_ids = [
            decision.document_id
            for decision in document_plan.decisions
            if decision.action is DocumentAction.UNCHANGED and decision.document_id is not None
        ]
        await documents.touch_seen(unchanged_ids, seen_at=synchronized_at)
        changed_ids = list(document_ids.values())
        await documents.mark_indexed(changed_ids, indexed_at=synchronized_at)
        await documents.soft_delete(missing_document_ids, deleted_at=synchronized_at)
        deleted_document_chunks = await chunks.soft_delete_for_documents(
            missing_document_ids,
            deleted_at=synchronized_at,
        )
        if deleted_document_chunks:
            stats = stats.add(chunks_deleted=deleted_document_chunks)
        duplicate_document_chunks = await chunks.soft_delete_for_documents(
            list(duplicate_document_ids.values()),
            deleted_at=synchronized_at,
        )
        if duplicate_document_chunks:
            stats = stats.add(chunks_deleted=duplicate_document_chunks)
        if checkpoint is not None:
            # This is deliberately last; transaction rollback prevents cursor advancement.
            await sources.upsert_cursor(checkpoint)
        return SyncResult(
            source_id=source.id,
            stats=stats,
            complete_snapshot=complete_snapshot,
            dry_run=False,
            checkpoint=checkpoint,
            request_id=request_id,
        )


def _document_upsert(
    source_id: UUID,
    document: PreparedDocument,
    document_id: UUID,
) -> DocumentUpsert:
    metadata = dict(document.parsed.metadata)
    metadata["deduplication_fingerprint"] = {
        "normalized_sha256": normalized_content_hash(document.parsed.content),
        "simhash64": simhash64(document.parsed.content),
    }
    repository_path = document.raw.metadata.get("repository_path")
    language = document.raw.metadata.get("language", "en")
    return DocumentUpsert(
        id=document_id,
        source_id=source_id,
        external_id=document.raw.external_id,
        document_type=document.parsed.document_type.value,
        title=document.parsed.title,
        canonical_url=(
            str(document.parsed.canonical_url) if document.parsed.canonical_url else None
        ),
        repository_path=(repository_path if isinstance(repository_path, str) else None),
        source_version=document.raw.source_version,
        language=language if isinstance(language, str) and language else "en",
        content_hash=document.content_hash,
        metadata=metadata,
    )


def _duplicate_document_upsert(
    source_id: UUID,
    duplicate: PreparedDuplicate,
    document_id: UUID,
) -> DocumentUpsert:
    """Build a non-retrievable document row with a durable dedup audit trail."""

    record = _document_upsert(source_id, duplicate.document, document_id)
    decision = duplicate.decision
    metadata = dict(record.metadata)
    metadata["deduplication"] = {
        "method": decision.method.value,
        "reason": decision.reason,
        "similarity": decision.similarity,
        "matched_external_id": decision.matched_external_id,
    }
    return replace(record, metadata=metadata)


def _document_version(
    document: PreparedDocument,
    document_id: UUID,
) -> DocumentVersionAppend:
    return DocumentVersionAppend(
        document_id=document_id,
        source_version=document.raw.source_version,
        content_hash=document.content_hash,
        raw_content=document.raw.content,
        parsed_content=document.parsed.content,
    )


def _chunk_ids(plan: ChunkSyncPlan) -> dict[int, UUID]:
    ids_by_index: dict[int, UUID] = {}
    for decision in plan.decisions:
        ids_by_index[decision.incoming.chunk_index] = decision.chunk_id or uuid4()
    return ids_by_index


def _embedding_inputs(work_items: Sequence[_ChunkWork]) -> list[tuple[tuple[UUID, int], str]]:
    inputs: list[tuple[tuple[UUID, int], str]] = []
    for work in work_items:
        drafts = {draft.chunk_index: draft for draft in work.document.chunks}
        for decision in work.plan.decisions:
            if decision.requires_embedding:
                draft = drafts[decision.incoming.chunk_index]
                inputs.append(((work.document_id, draft.chunk_index), draft.contextualized_content))
    return inputs


def _chunk_upserts(
    work_items: Sequence[_ChunkWork],
    embedded: Mapping[tuple[UUID, int], _EmbeddedVector],
) -> list[ChunkUpsert]:
    upserts: list[ChunkUpsert] = []
    for work in work_items:
        action_by_index = {
            decision.incoming.chunk_index: decision.action for decision in work.plan.decisions
        }
        for draft in work.document.chunks:
            parent_id: UUID | None = None
            if draft.parent_index is not None:
                if draft.parent_index == draft.chunk_index:
                    raise IngestionError("Chunk cannot be its own parent")
                try:
                    parent_id = work.ids_by_index[draft.parent_index]
                except KeyError as exc:
                    raise IngestionError(
                        "Chunk parent_index does not exist in its document"
                    ) from exc
            vector = embedded.get((work.document_id, draft.chunk_index))
            requires_embedding = action_by_index[draft.chunk_index] in {
                ChunkAction.CREATE,
                ChunkAction.UPDATE,
            }
            if requires_embedding and vector is None:
                raise ProviderError("A changed chunk is missing its embedding")
            metadata = dict(draft.metadata)
            metadata["content_sha256"] = draft.content_hash
            upserts.append(
                ChunkUpsert(
                    id=work.ids_by_index[draft.chunk_index],
                    document_id=work.document_id,
                    chunk_index=draft.chunk_index,
                    parent_chunk_id=parent_id,
                    document_title=work.document.parsed.title,
                    heading_path=list(draft.heading_path),
                    content=draft.content,
                    contextualized_content=draft.contextualized_content,
                    token_count=draft.token_count,
                    content_hash=normalized_content_hash(draft.contextualized_content),
                    start_offset=draft.position.start_offset,
                    end_offset=draft.position.end_offset,
                    start_line=draft.position.start_line,
                    end_line=draft.position.end_line,
                    metadata=metadata,
                    embedding=vector.values if vector else None,
                    embedding_model=vector.model if vector else None,
                    embedding_dimension=vector.dimension if vector else None,
                )
            )
    return upserts


def _build_checkpoint(
    source_id: UUID,
    spec: LoaderSpec,
    raw_documents: Sequence[RawDocument],
    synchronized_at: datetime,
) -> CursorCheckpoint | None:
    values: list[str] = []
    if spec.cursor_type == "repository_commit_sha":
        values = [
            value
            for document in raw_documents
            if isinstance((value := document.metadata.get("commit_sha")), str) and value
        ]
        if not values:
            values = [
                document.source_version for document in raw_documents if document.source_version
            ]
    elif spec.cursor_type in {"issues_updated_at", "releases_updated_at"}:
        timestamps = [document.updated_at for document in raw_documents if document.updated_at]
        values = [max(timestamps).astimezone(UTC).isoformat()] if timestamps else []
    elif spec.cursor_type == "api_snapshot_at":
        values = [synchronized_at.astimezone(UTC).isoformat()]
    if not values:
        return None
    cursor_value = max(values)
    if spec.previous_cursor is not None and spec.cursor_type.endswith("updated_at"):
        previous = datetime.fromisoformat(spec.previous_cursor.replace("Z", "+00:00"))
        current = datetime.fromisoformat(cursor_value.replace("Z", "+00:00"))
        if current <= previous:
            return None
    return CursorCheckpoint(
        source_id=source_id,
        cursor_type=spec.cursor_type,
        cursor_value=cursor_value,
        metadata={"complete_snapshot": spec.complete_snapshot},
    )


def _deduplication_threshold(source: Source) -> float:
    value = (source.config or {}).get("deduplication_threshold", 0.90)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError("deduplication_threshold must be numeric")
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ConfigurationError("deduplication_threshold must be between 0 and 1")
    return threshold


def _deduplicate_against_stored(
    batch: PreparedBatch,
    stored_fingerprints: Sequence[StoredDeduplicationFingerprint],
    *,
    threshold: float,
) -> PreparedBatch:
    """Apply the same metadata-aware policy to retained documents from prior runs."""

    if not batch.documents or not stored_fingerprints:
        return batch
    deduplicator = ContentDeduplicator(threshold)
    for fingerprint in stored_fingerprints:
        deduplicator.add_fingerprint(
            fingerprint.candidate,
            exact_hash=fingerprint.exact_hash,
            simhash=fingerprint.simhash,
        )

    retained: list[PreparedDocument] = []
    duplicates = list(batch.duplicates)
    for document in batch.documents:
        decision = deduplicator.check_and_add(DeduplicationCandidate.from_parsed(document.parsed))
        if decision.is_duplicate:
            duplicates.append(PreparedDuplicate(document, decision))
        else:
            retained.append(document)
    newly_skipped = len(duplicates) - len(batch.duplicates)
    return replace(
        batch,
        documents=tuple(retained),
        duplicates=tuple(duplicates),
        duplicates_skipped=batch.duplicates_skipped + newly_skipped,
    )


def _document_fingerprint(document: ParsedDocument) -> str:
    """Hash every stable field that can alter identity or embedding context."""

    semantic_metadata = {
        key: document.metadata[key]
        for key in _SEMANTIC_DOCUMENT_METADATA
        if key in document.metadata
    }
    material = json.dumps(
        {
            "title": document.title,
            "content": document.content,
            "canonical_url": (
                str(document.canonical_url) if document.canonical_url is not None else None
            ),
            "document_type": document.document_type.value,
            "source_type": document.source_type,
            "semantic_metadata": semantic_metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(material.encode("utf-8")).hexdigest()


def _source_format(document: RawDocument) -> str:
    source_format = document.metadata.get("format")
    if isinstance(source_format, str) and source_format:
        return source_format.casefold().lstrip(".")
    repository_path = document.metadata.get("repository_path")
    if isinstance(repository_path, str) and "." in repository_path:
        return repository_path.rsplit(".", 1)[-1].casefold()
    return "markdown"


__all__ = [
    "IngestionService",
    "PreparedBatch",
    "PreparedDocument",
    "SyncResult",
    "embed_changed_chunks",
    "prepare_documents",
]
