"""Unit coverage for the vertical ingestion orchestration path."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import ConfigurationError, ProviderError
from app.db.models.source_document import Source
from app.domain.documents import RawDocument
from app.ingestion.deduplication import DeduplicationCandidate
from app.ingestion.factory import LoaderSpec, create_loader_spec
from app.ingestion.incremental import StoredChunkState, StoredDocumentState
from app.ingestion.loaders.base import DocumentLoader
from app.providers.base import EmbeddingProvider, EmbeddingResponse, TokenUsage
from app.repositories.document_repository import StoredDeduplicationFingerprint
from app.repositories.usage_repository import UsageRecordCreate
from app.services import ingestion_service as module
from app.services.ingestion_service import IngestionService
from app.services.pricing_service import ModelPricing, PricingCatalog
from app.worker import _safe_error


class _FakeLoader(DocumentLoader):
    def __init__(self, documents: list[RawDocument]) -> None:
        self.documents = documents

    async def load(self) -> list[RawDocument]:
        return list(self.documents)


class _FakeEmbedding(EmbeddingProvider):
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    @property
    def name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return 3

    async def healthcheck(self) -> None:
        return None

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        self.calls.append(list(texts))
        if self.fail:
            raise ProviderError("synthetic embedding failure")
        return EmbeddingResponse(
            vectors=[[float(index), 0.5, 1.0] for index, _ in enumerate(texts)],
            model="fake-embedding",
            dimension=3,
            usage=TokenUsage(prompt_tokens=sum(max(1, len(text.split())) for text in texts)),
        )


@dataclass
class _Store:
    source: Source
    documents: dict[str, StoredDocumentState] = field(default_factory=dict)
    chunks: dict[UUID, dict[int, StoredChunkState]] = field(default_factory=dict)
    versions: list[Any] = field(default_factory=list)
    cursors: dict[str, str] = field(default_factory=dict)
    chunk_upserts: list[Any] = field(default_factory=list)
    duplicate_upserts: list[Any] = field(default_factory=list)
    fingerprints: dict[str, StoredDeduplicationFingerprint] = field(default_factory=dict)
    chunk_state_query_count: int = 0
    rollback_count: int = 0
    usage_records: list[UsageRecordCreate] = field(default_factory=list)
    usage_add_calls: int = 0


class _FakeSession:
    def __init__(self, store: _Store) -> None:
        self.store = store


class _SessionContext:
    def __init__(self, store: _Store, *, transactional: bool) -> None:
        self.store = store
        self.transactional = transactional
        self.snapshot: tuple[Any, ...] | None = None

    async def __aenter__(self) -> AsyncSession:
        if self.transactional:
            self.snapshot = copy.deepcopy(
                (
                    self.store.documents,
                    self.store.chunks,
                    self.store.versions,
                    self.store.cursors,
                    self.store.chunk_upserts,
                    self.store.duplicate_upserts,
                    self.store.fingerprints,
                    self.store.usage_records,
                    self.store.usage_add_calls,
                )
            )
        return cast(AsyncSession, _FakeSession(self.store))

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        if exc_type is not None and self.transactional and self.snapshot is not None:
            (
                self.store.documents,
                self.store.chunks,
                self.store.versions,
                self.store.cursors,
                self.store.chunk_upserts,
                self.store.duplicate_upserts,
                self.store.fingerprints,
                self.store.usage_records,
                self.store.usage_add_calls,
            ) = self.snapshot
            self.store.rollback_count += 1


class _FakeSessionFactory:
    def __init__(self, store: _Store) -> None:
        self.store = store

    def __call__(self) -> _SessionContext:
        return _SessionContext(self.store, transactional=False)

    def begin(self) -> _SessionContext:
        return _SessionContext(self.store, transactional=True)


class _FakeSourceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.store = cast(_FakeSession, session).store

    async def get(self, source_id: UUID) -> Source | None:
        return self.store.source if self.store.source.id == source_id else None

    async def get_cursors(self, source_id: UUID) -> list[Any]:
        return [
            SimpleNamespace(cursor_type=key, cursor_value=value)
            for key, value in self.store.cursors.items()
        ]

    async def acquire_sync_lock(self, source_id: UUID) -> None:
        return None

    async def upsert_cursor(self, checkpoint: Any) -> Any:
        self.store.cursors[checkpoint.cursor_type] = checkpoint.cursor_value
        return checkpoint


class _FakeDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.store = cast(_FakeSession, session).store

    async def list_states_for_source(self, source_id: UUID) -> list[StoredDocumentState]:
        return list(self.store.documents.values())

    async def list_deduplication_fingerprints(
        self,
        source_id: UUID,
        *,
        exclude_external_ids: tuple[str, ...] = (),
    ) -> list[StoredDeduplicationFingerprint]:
        excluded = set(exclude_external_ids)
        return [
            fingerprint
            for external_id, fingerprint in self.store.fingerprints.items()
            if external_id not in excluded
            and (state := self.store.documents.get(external_id)) is not None
            and state.deleted_at is None
        ]

    async def upsert_many(self, records: list[Any], *, seen_at: datetime) -> list[Any]:
        rows: list[Any] = []
        for record in records:
            current = self.store.documents.get(record.external_id)
            document_id = current.id if current is not None else record.id
            self.store.documents[record.external_id] = StoredDocumentState(
                id=document_id,
                source_id=record.source_id,
                external_id=record.external_id,
                content_hash=record.content_hash,
            )
            fingerprint = record.metadata["deduplication_fingerprint"]
            self.store.fingerprints[record.external_id] = StoredDeduplicationFingerprint(
                candidate=DeduplicationCandidate(
                    external_id=record.external_id,
                    content="",
                    source_type=self.store.source.source_type,
                    source_version=record.source_version,
                    document_type=record.document_type,
                    metadata=record.metadata,
                ),
                exact_hash=fingerprint["normalized_sha256"],
                simhash=fingerprint["simhash64"],
            )
            rows.append(SimpleNamespace(id=document_id, external_id=record.external_id))
        return rows

    async def append_versions(self, versions: list[Any]) -> int:
        self.store.versions.extend(versions)
        return len(versions)

    async def upsert_duplicates(self, records: list[Any], *, seen_at: datetime) -> list[Any]:
        rows: list[Any] = []
        self.store.duplicate_upserts.extend(records)
        for record in records:
            current = self.store.documents.get(record.external_id)
            document_id = current.id if current is not None else record.id
            self.store.documents[record.external_id] = StoredDocumentState(
                id=document_id,
                source_id=record.source_id,
                external_id=record.external_id,
                content_hash=record.content_hash,
                deleted_at=seen_at,
            )
            self.store.fingerprints.pop(record.external_id, None)
            rows.append(SimpleNamespace(id=document_id, external_id=record.external_id))
        return rows

    async def touch_seen(self, document_ids: list[UUID], *, seen_at: datetime) -> int:
        return len(document_ids)

    async def mark_indexed(self, document_ids: list[UUID], *, indexed_at: datetime) -> int:
        return len(document_ids)

    async def soft_delete(self, document_ids: list[UUID], *, deleted_at: datetime) -> int:
        affected = 0
        for external_id, state in list(self.store.documents.items()):
            if state.id in document_ids and state.deleted_at is None:
                self.store.documents[external_id] = state.model_copy(
                    update={"deleted_at": deleted_at}
                )
                affected += 1
        return affected


class _FakeChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.store = cast(_FakeSession, session).store

    async def list_states_for_documents(
        self,
        document_ids: list[UUID],
    ) -> dict[UUID, list[StoredChunkState]]:
        self.store.chunk_state_query_count += 1
        return {
            document_id: list(self.store.chunks.get(document_id, {}).values())
            for document_id in document_ids
        }

    async def upsert_many(self, records: list[Any], *, written_at: datetime) -> list[Any]:
        self.store.chunk_upserts.extend(records)
        for record in records:
            states = self.store.chunks.setdefault(record.document_id, {})
            current = states.get(record.chunk_index)
            chunk_id = current.id if current is not None else record.id
            states[record.chunk_index] = StoredChunkState(
                id=chunk_id,
                document_id=record.document_id,
                chunk_index=record.chunk_index,
                content_hash=record.content_hash,
            )
        return []

    async def soft_delete(self, chunk_ids: list[UUID], *, deleted_at: datetime) -> int:
        affected = 0
        for states in self.store.chunks.values():
            for index, state in list(states.items()):
                if state.id in chunk_ids and state.deleted_at is None:
                    states[index] = state.model_copy(update={"deleted_at": deleted_at})
                    affected += 1
        return affected

    async def soft_delete_for_documents(
        self, document_ids: list[UUID], *, deleted_at: datetime
    ) -> int:
        affected = 0
        for document_id in document_ids:
            for index, state in list(self.store.chunks.get(document_id, {}).items()):
                if state.deleted_at is None:
                    self.store.chunks[document_id][index] = state.model_copy(
                        update={"deleted_at": deleted_at}
                    )
                    affected += 1
        return affected


class _FakeUsageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.store = cast(_FakeSession, session).store

    async def add_many(self, records: list[UsageRecordCreate]) -> list[object]:
        self.store.usage_add_calls += 1
        self.store.usage_records.extend(records)
        return []


@pytest.fixture
def repository_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "SourceRepository", _FakeSourceRepository)
    monkeypatch.setattr(module, "DocumentRepository", _FakeDocumentRepository)
    monkeypatch.setattr(module, "ChunkRepository", _FakeChunkRepository)
    monkeypatch.setattr(module, "UsageRepository", _FakeUsageRepository)


def _source() -> Source:
    return Source(
        id=uuid4(),
        name="test-kubernetes-docs",
        source_type="github_repository",
        repository="kubernetes/website",
        branch="main",
        enabled=True,
        config={"deduplication_threshold": 1.0},
    )


def _raw(external_id: str, body: str) -> RawDocument:
    return RawDocument(
        source_type="github_repository",
        external_id=external_id,
        title=external_id,
        content=f"# {external_id}\n\n{body}\n",
        canonical_url=f"https://example.test/{external_id}",
        source_version="commit-one",
        metadata={
            "format": "md",
            "repository_path": external_id,
            "commit_sha": "commit-one",
            "document_type": "official_documentation",
        },
    )


def _service(
    store: _Store,
    provider: EmbeddingProvider | None,
) -> IngestionService:
    factory = cast(async_sessionmaker[AsyncSession], _FakeSessionFactory(store))
    return IngestionService(
        factory,
        embedding_provider=provider,
        embedding_dimension=3,
        cache_root=Path(".cache/test-ingestion"),
        clock=lambda: datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
        pricing_catalog=PricingCatalog(
            {
                "fake": {
                    "fake-embedding": ModelPricing(
                        input_per_million_tokens=1,
                        output_per_million_tokens=0,
                    )
                }
            }
        ),
    )


@pytest.mark.asyncio
async def test_repeat_sync_is_idempotent_and_only_changed_chunks_are_embedded(
    repository_fakes: None,
) -> None:
    source = _source()
    store = _Store(source=source)
    provider = _FakeEmbedding()
    loader = _FakeLoader(
        [
            _raw("deployment.md", "Deployment rollout undo behavior and revision history."),
            _raw("service.md", "Service ClusterIP networking and endpoint selection."),
        ]
    )
    spec = LoaderSpec(loader, True, "repository_commit_sha")
    service = _service(store, provider)

    first = await asyncio.wait_for(service.sync_source(source.id, loader_spec=spec), timeout=3)
    second = await service.sync_source(source.id, loader_spec=spec)

    assert first.stats.created == 2
    assert second.stats.unchanged == 2
    assert store.chunk_state_query_count == 1
    assert [len(call) for call in provider.calls] == [2]
    assert store.usage_add_calls == 1
    assert len(store.usage_records) == 1
    first_usage = store.usage_records[0]
    assert first_usage.request_id == first.request_id
    assert first_usage.operation == "ingestion_embedding"
    assert first_usage.input_text_count == 2
    assert first_usage.input_character_count == sum(len(text) for text in provider.calls[0])
    assert first_usage.prompt_tokens > 0
    assert first_usage.estimated_cost is not None
    assert first_usage.latency_ms >= 0
    assert len(store.documents) == 2
    assert len(store.versions) == 2
    assert store.cursors["repository_commit_sha"] == "commit-one"

    loader.documents[0] = _raw(
        "deployment.md",
        "Deployment rollout undo behavior changed and now describes revision limits.",
    )
    third = await service.sync_source(source.id, loader_spec=spec)

    assert third.stats.updated == 1
    assert third.stats.unchanged == 1
    assert store.chunk_state_query_count == 2
    assert [len(call) for call in provider.calls] == [2, 1]
    assert store.usage_add_calls == 2
    assert store.usage_records[-1].request_id == third.request_id
    assert store.usage_records[-1].input_text_count == 1
    assert len(store.versions) == 3

    loader.documents[1] = loader.documents[1].model_copy(update={"title": "Kubernetes Services"})
    fourth = await service.sync_source(source.id, loader_spec=spec)

    assert fourth.stats.updated == 1
    assert store.chunk_state_query_count == 3
    assert [len(call) for call in provider.calls] == [2, 1, 1]
    assert store.usage_add_calls == 3


@pytest.mark.asyncio
async def test_new_duplicate_state_is_audited_and_stale_chunks_are_excluded(
    repository_fakes: None,
) -> None:
    source = _source()
    store = _Store(source=source)
    provider = _FakeEmbedding()
    first = _raw("one.md", "First distinct Kubernetes document about rollouts.")
    second = _raw("two.md", "Second distinct Kubernetes document about services.")
    loader = _FakeLoader([first, second])
    service = _service(store, provider)
    spec = LoaderSpec(loader, True, "repository_commit_sha")

    await service.sync_source(source.id, loader_spec=spec)
    second_document_id = store.documents["two.md"].id
    assert store.chunks[second_document_id][0].deleted_at is None

    # The logical identity remains two.md, but its newly fetched content is now
    # byte-for-byte equivalent to one.md and therefore must not leave old chunks live.
    loader.documents[1] = second.model_copy(update={"content": first.content, "title": first.title})
    result = await service.sync_source(source.id, loader_spec=spec)

    assert result.stats.duplicates_skipped == 1
    assert result.stats.chunks_deleted == 1
    assert store.documents["two.md"].deleted_at is not None
    assert store.chunks[second_document_id][0].deleted_at is not None
    audit = store.duplicate_upserts[-1].metadata["deduplication"]
    assert audit["method"] == "exact_sha256"
    assert audit["matched_external_id"] == "one.md"
    assert "SHA-256" in audit["reason"]


@pytest.mark.asyncio
async def test_embedding_failure_rolls_back_versions_chunks_and_cursor(
    repository_fakes: None,
) -> None:
    source = _source()
    store = _Store(source=source)
    service = _service(store, _FakeEmbedding(fail=True))
    spec = LoaderSpec(
        _FakeLoader([_raw("deployment.md", "Deployment strategy and rollback guidance.")]),
        True,
        "repository_commit_sha",
    )

    with pytest.raises(ProviderError, match="synthetic"):
        await service.sync_source(source.id, loader_spec=spec)

    assert store.rollback_count == 1
    assert store.documents == {}
    assert store.chunks == {}
    assert store.versions == []
    assert store.cursors == {}


@pytest.mark.asyncio
async def test_dry_run_does_not_require_provider_or_mutate_state(
    repository_fakes: None,
) -> None:
    source = _source()
    store = _Store(source=source)
    service = _service(store, None)
    spec = LoaderSpec(
        _FakeLoader([_raw("api.md", "API fields and versioned kind information.")]),
        True,
        "repository_commit_sha",
    )

    result = await service.sync_source(source.id, loader_spec=spec, dry_run=True)

    assert result.dry_run is True
    assert result.stats.created == 1
    assert result.stats.chunks_created == 1
    assert store.documents == {}
    assert store.cursors == {}


@pytest.mark.asyncio
async def test_delta_sync_never_soft_deletes_missing_documents(
    repository_fakes: None,
) -> None:
    source = _source()
    store = _Store(source=source)
    provider = _FakeEmbedding()
    loader = _FakeLoader(
        [
            _raw("one.md", "First independent Kubernetes concept and its behavior."),
            _raw("two.md", "Second independent Kubernetes concept and its behavior."),
        ]
    )
    service = _service(store, provider)
    await service.sync_source(
        source.id,
        loader_spec=LoaderSpec(loader, True, "repository_commit_sha"),
    )

    loader.documents = [_raw("one.md", "First Kubernetes concept received a material update.")]
    result = await service.sync_source(
        source.id,
        loader_spec=LoaderSpec(loader, False, "issues_updated_at"),
    )

    assert result.complete_snapshot is False
    assert result.stats.deleted == 0
    assert store.documents["two.md"].deleted_at is None


@pytest.mark.asyncio
async def test_delta_sync_deduplicates_against_an_unseen_document_from_a_prior_run(
    repository_fakes: None,
) -> None:
    source = _source()
    store = _Store(source=source)
    provider = _FakeEmbedding()
    canonical = _raw("canonical.md", "Deployment rollback evidence retained across syncs.")
    other = _raw("other.md", "Service networking evidence is unrelated to rollback.")
    service = _service(store, provider)

    await service.sync_source(
        source.id,
        loader_spec=LoaderSpec(
            _FakeLoader([canonical, other]),
            True,
            "repository_commit_sha",
        ),
    )
    duplicate = canonical.model_copy(
        update={
            "external_id": "copy.md",
            "canonical_url": "https://example.test/copy.md",
        }
    )
    result = await service.sync_source(
        source.id,
        loader_spec=LoaderSpec(
            _FakeLoader([duplicate]),
            False,
            "issues_updated_at",
        ),
    )

    assert result.stats.created == 0
    assert result.stats.duplicates_skipped == 1
    assert [len(call) for call in provider.calls] == [2]
    assert store.documents["copy.md"].deleted_at is not None
    assert len(store.versions) == 3
    audit = store.duplicate_upserts[-1].metadata["deduplication"]
    assert audit["method"] == "exact_sha256"
    assert audit["matched_external_id"] == "canonical.md"


@pytest.mark.asyncio
async def test_parent_index_is_persisted_as_parent_chunk_uuid(
    repository_fakes: None,
) -> None:
    source = _source()
    store = _Store(source=source)
    provider = _FakeEmbedding()
    nested = _raw("nested.md", "placeholder").model_copy(
        update={
            "content": (
                "# Workloads\n\nDeployment overview and controller behavior.\n\n"
                "## Rollback\n\nUse rollout undo to restore a prior revision.\n"
            )
        }
    )
    service = _service(store, provider)

    await service.sync_source(
        source.id,
        loader_spec=LoaderSpec(_FakeLoader([nested]), True, "repository_commit_sha"),
    )

    assert len(store.chunk_upserts) == 2
    parent, child = store.chunk_upserts
    assert child.parent_chunk_id == parent.id
    assert child.document_id == parent.document_id


def test_worker_error_is_bounded_and_redacts_credentials() -> None:
    error = _safe_error(
        RuntimeError("token=visible-token secret-value " + "x" * 3000),
        ["secret-value"],
    )

    assert len(error) == 2000
    assert "visible-token" not in error
    assert "secret-value" not in error
    assert "[REDACTED]" in error


def test_factory_rejects_checkout_escape(tmp_path: Path) -> None:
    source = _source()
    source.config = {"checkout_subdir": "../outside"}

    with pytest.raises(ConfigurationError, match="inside"):
        create_loader_spec(source, cache_root=tmp_path)


def test_factory_validates_repository_git_timeout(tmp_path: Path) -> None:
    source = _source()
    source.config = {"git_timeout_seconds": 0}

    with pytest.raises(ConfigurationError, match="git_timeout_seconds must be a positive"):
        create_loader_spec(source, cache_root=tmp_path)
