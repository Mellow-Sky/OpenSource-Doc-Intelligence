"""Incremental synchronization decisions and queue statement guarantees."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Executable

from app.ingestion.deduplication import normalized_content_hash, simhash64
from app.ingestion.incremental import (
    ChunkAction,
    CursorCheckpoint,
    DocumentAction,
    IncomingChunkState,
    IncomingDocumentState,
    StoredChunkState,
    StoredDocumentState,
    SyncStats,
    decide_document,
    plan_chunk_sync,
    plan_document_sync,
)
from app.repositories.batching import DATABASE_WRITE_BATCH_SIZE, database_batches
from app.repositories.chunk_repository import ChunkRepository, ChunkUpsert
from app.repositories.document_repository import (
    DocumentRepository,
    DocumentUpsert,
    DocumentVersionAppend,
)
from app.repositories.ingestion_job_repository import IngestionJobRepository
from app.repositories.source_repository import SourceRepository


def _hash(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _incoming_document(source_id: UUID, external_id: str, content: str) -> IncomingDocumentState:
    return IncomingDocumentState(
        source_id=source_id,
        external_id=external_id,
        content_hash=_hash(content),
    )


def _stored_document(
    source_id: UUID,
    external_id: str,
    content: str,
    *,
    deleted_at: datetime | None = None,
) -> StoredDocumentState:
    return StoredDocumentState(
        id=uuid4(),
        source_id=source_id,
        external_id=external_id,
        content_hash=_hash(content),
        deleted_at=deleted_at,
    )


def _incoming_chunk(document_id: UUID, chunk_index: int, content: str) -> IncomingChunkState:
    return IncomingChunkState(
        document_id=document_id,
        chunk_index=chunk_index,
        content_hash=_hash(content),
    )


def _stored_chunk(
    document_id: UUID,
    chunk_index: int,
    content: str,
    *,
    deleted_at: datetime | None = None,
) -> StoredChunkState:
    return StoredChunkState(
        id=uuid4(),
        document_id=document_id,
        chunk_index=chunk_index,
        content_hash=_hash(content),
        deleted_at=deleted_at,
    )


def test_sync_stats_has_complete_schema_and_composes_immutably() -> None:
    expected_fields = {
        "scanned",
        "created",
        "updated",
        "unchanged",
        "deleted",
        "chunks_created",
        "chunks_updated",
        "chunks_deleted",
        "duplicates_skipped",
        "errors",
    }
    stats = SyncStats(scanned=2, unchanged=2).add(chunks_created=3)
    combined = stats.merge(SyncStats(deleted=1, duplicates_skipped=1, errors=1))

    assert set(combined.model_dump()) == expected_fields
    assert combined.scanned == 2
    assert combined.chunks_created == 3
    assert combined.deleted == 1
    assert SyncStats().model_dump() == dict.fromkeys(expected_fields, 0)


def test_sync_stats_rejects_invalid_increment() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SyncStats().add(errors=-1)
    with pytest.raises(ValueError, match="unknown"):
        SyncStats().add(not_a_counter=1)


def test_document_decision_uses_composite_identity_hash_and_restore_state() -> None:
    source_id = uuid4()
    incoming = _incoming_document(source_id, "docs/deployment.md", "new")

    assert decide_document(incoming, None).action is DocumentAction.CREATE
    changed = decide_document(
        incoming,
        _stored_document(source_id, "docs/deployment.md", "old"),
    )
    assert changed.action is DocumentAction.UPDATE
    assert changed.reason == "normalized content hash changed"

    restored = decide_document(
        incoming,
        _stored_document(
            source_id,
            "docs/deployment.md",
            "new",
            deleted_at=datetime(2026, 7, 1, tzinfo=UTC),
        ),
    )
    assert restored.action is DocumentAction.UPDATE
    assert "restored" in restored.reason

    with pytest.raises(ValueError, match="composite identity"):
        decide_document(incoming, _stored_document(source_id, "docs/service.md", "new"))


def test_document_snapshot_plan_soft_deletes_only_missing_active_documents() -> None:
    source_id = uuid4()
    deleted_at = datetime(2026, 7, 1, tzinfo=UTC)
    stored = [
        _stored_document(source_id, "same", "same"),
        _stored_document(source_id, "changed", "old"),
        _stored_document(source_id, "restored", "same", deleted_at=deleted_at),
        _stored_document(source_id, "missing", "missing"),
        _stored_document(source_id, "already-missing", "missing", deleted_at=deleted_at),
    ]
    incoming = [
        _incoming_document(source_id, "same", "same"),
        _incoming_document(source_id, "changed", "new"),
        _incoming_document(source_id, "restored", "same"),
        _incoming_document(source_id, "created", "created"),
    ]

    plan = plan_document_sync(source_id, incoming, stored)

    assert [decision.action for decision in plan.decisions] == [
        DocumentAction.UNCHANGED,
        DocumentAction.UPDATE,
        DocumentAction.UPDATE,
        DocumentAction.CREATE,
    ]
    assert plan.missing_document_ids == [stored[3].id]
    assert plan.stats == SyncStats(scanned=4, created=1, updated=2, unchanged=1, deleted=1)


def test_second_document_sync_is_idempotently_unchanged() -> None:
    source_id = uuid4()
    incoming = [
        _incoming_document(source_id, "a", "alpha"),
        _incoming_document(source_id, "b", "beta"),
    ]
    persisted = [StoredDocumentState(id=uuid4(), **document.model_dump()) for document in incoming]

    first_repeat = plan_document_sync(source_id, incoming, persisted)
    second_repeat = plan_document_sync(source_id, incoming, persisted)

    assert first_repeat == second_repeat
    assert first_repeat.missing_document_ids == []
    assert first_repeat.stats == SyncStats(scanned=2, unchanged=2)


def test_chunk_hash_diff_embeds_only_new_and_changed_chunks() -> None:
    document_id = uuid4()
    deleted_at = datetime(2026, 7, 1, tzinfo=UTC)
    stored = [
        _stored_chunk(document_id, 0, "same"),
        _stored_chunk(document_id, 1, "old"),
        _stored_chunk(document_id, 2, "removed"),
        _stored_chunk(document_id, 3, "restored", deleted_at=deleted_at),
    ]
    incoming = [
        _incoming_chunk(document_id, 0, "same"),
        _incoming_chunk(document_id, 1, "new"),
        _incoming_chunk(document_id, 3, "restored"),
        _incoming_chunk(document_id, 4, "created"),
    ]

    plan = plan_chunk_sync(document_id, incoming, stored)

    assert [decision.action for decision in plan.decisions] == [
        ChunkAction.UNCHANGED,
        ChunkAction.UPDATE,
        ChunkAction.RESTORE,
        ChunkAction.CREATE,
    ]
    assert plan.embedding_chunk_indices == [1, 4]
    assert plan.removed_chunk_ids == [stored[2].id]
    assert plan.stats == SyncStats(chunks_created=1, chunks_updated=2, chunks_deleted=1)


def test_second_chunk_sync_has_no_writes_or_embedding_work() -> None:
    document_id = uuid4()
    incoming = [
        _incoming_chunk(document_id, 0, "alpha"),
        _incoming_chunk(document_id, 1, "beta"),
    ]
    persisted = [StoredChunkState(id=uuid4(), **chunk.model_dump()) for chunk in incoming]

    plan = plan_chunk_sync(document_id, incoming, persisted)

    assert all(decision.action is ChunkAction.UNCHANGED for decision in plan.decisions)
    assert plan.embedding_chunk_indices == []
    assert plan.removed_chunk_ids == []
    assert plan.stats == SyncStats()


def test_chunk_prefix_insertion_reuses_shifted_embeddings_with_lcs() -> None:
    document_id = uuid4()
    stored = [
        _stored_chunk(document_id, 0, "alpha"),
        _stored_chunk(document_id, 1, "beta"),
        _stored_chunk(document_id, 2, "gamma"),
    ]
    incoming = [
        _incoming_chunk(document_id, 0, "new-prefix"),
        _incoming_chunk(document_id, 1, "alpha"),
        _incoming_chunk(document_id, 2, "beta"),
        _incoming_chunk(document_id, 3, "gamma"),
    ]

    plan = plan_chunk_sync(document_id, incoming, stored)

    assert [decision.action for decision in plan.decisions] == [
        ChunkAction.CREATE,
        ChunkAction.MOVE,
        ChunkAction.MOVE,
        ChunkAction.MOVE,
    ]
    assert plan.embedding_chunk_indices == [0]
    assert [decision.chunk_id for decision in plan.decisions[1:]] == [item.id for item in stored]
    assert plan.removed_chunk_ids == []


def test_planners_reject_duplicate_batch_keys() -> None:
    source_id = uuid4()
    duplicate_document = _incoming_document(source_id, "same", "same")
    with pytest.raises(ValueError, match="duplicate incoming external_id"):
        plan_document_sync(source_id, [duplicate_document, duplicate_document], [])

    document_id = uuid4()
    duplicate_chunk = _incoming_chunk(document_id, 0, "same")
    with pytest.raises(ValueError, match="duplicate incoming chunk_index"):
        plan_chunk_sync(document_id, [duplicate_chunk, duplicate_chunk], [])


class _EmptyScalarResult:
    def scalar_one_or_none(self) -> None:
        return None


class _RecordingSession:
    def __init__(self) -> None:
        self.statement: Executable | None = None

    async def execute(self, statement: Executable) -> _EmptyScalarResult:
        self.statement = statement
        return _EmptyScalarResult()


@pytest.mark.asyncio
async def test_job_claim_is_one_parameterized_update_with_skip_locked() -> None:
    session = _RecordingSession()
    repository = IngestionJobRepository(cast(AsyncSession, session))
    claimed_at = datetime(2026, 7, 17, 10, 0, tzinfo=UTC)

    job = await repository.claim_next(
        claimed_at=claimed_at,
        stale_before=claimed_at - timedelta(minutes=5),
    )

    assert job is None
    assert session.statement is not None
    compiled = session.statement.compile(dialect=postgresql.dialect())
    sql = str(compiled).upper()
    assert sql.startswith("UPDATE INGESTION_JOBS")
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "RETURNING INGESTION_JOBS" in sql
    assert compiled.params
    assert claimed_at.isoformat() not in str(compiled)


class _BatchResult:
    rowcount = 1

    def scalars(self) -> list[object]:
        return []

    def scalar_one(self) -> object:
        return object()


class _FingerprintRecordingSession:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self.rows = rows
        self.statements: list[Executable] = []

    async def execute(
        self, statement: Executable, parameters: object | None = None
    ) -> list[SimpleNamespace]:
        self.statements.append(statement)
        return self.rows


class _BatchRecordingSession:
    def __init__(self) -> None:
        self.statements: list[Executable] = []
        self.parameters: list[object | None] = []

    async def execute(
        self, statement: Executable, parameters: object | None = None
    ) -> _BatchResult:
        self.statements.append(statement)
        self.parameters.append(parameters)
        return _BatchResult()

    async def scalars(self, statement: Executable) -> list[object]:
        self.statements.append(statement)
        self.parameters.append(None)
        return []


@pytest.mark.asyncio
async def test_batch_upserts_construct_parameterized_sql_without_committing() -> None:
    session = _BatchRecordingSession()
    async_session = cast(AsyncSession, session)
    source_id = uuid4()
    document_id = uuid4()
    written_at = datetime(2026, 7, 17, 10, 0, tzinfo=UTC)

    await DocumentRepository(async_session).upsert_many(
        [
            DocumentUpsert(
                id=document_id,
                source_id=source_id,
                external_id="docs/deployment.md",
                document_type="official_documentation",
                title="Deployment",
                content_hash=_hash("document"),
                metadata={"branch": "main"},
            )
        ],
        seen_at=written_at,
    )
    await ChunkRepository(async_session).upsert_many(
        [
            ChunkUpsert(
                document_id=document_id,
                chunk_index=0,
                document_title="Deployment",
                heading_path=["Deployment", "Rolling back"],
                content="Use rollout undo.",
                contextualized_content="Deployment > Rolling back\nUse rollout undo.",
                token_count=5,
                content_hash=_hash("chunk"),
                start_offset=0,
                end_offset=17,
                metadata={"section": "rollback"},
            )
        ],
        written_at=written_at,
    )
    await SourceRepository(async_session).upsert_cursor(
        CursorCheckpoint(
            source_id=source_id,
            cursor_type="commit_sha",
            cursor_value="abc123",
        )
    )

    compiled_statements = [
        statement.compile(dialect=postgresql.dialect()) for statement in session.statements
    ]
    sql = [str(statement).upper() for statement in compiled_statements]
    assert sum("ON CONFLICT" in statement for statement in sql) == 2
    assert any("RANKED_CHUNKS" in statement and "UPDATE CHUNKS" in statement for statement in sql)
    assert any(statement.startswith("INSERT INTO CHUNKS") for statement in sql)
    assert all(statement.params for statement in compiled_statements)


def test_database_batches_preserve_order_and_enforce_safe_limit() -> None:
    values = list(range(DATABASE_WRITE_BATCH_SIZE * 2 + 1))

    batches = list(database_batches(values))

    assert [len(batch) for batch in batches] == [
        DATABASE_WRITE_BATCH_SIZE,
        DATABASE_WRITE_BATCH_SIZE,
        1,
    ]
    assert [item for batch in batches for item in batch] == values


@pytest.mark.asyncio
async def test_document_and_version_upserts_split_parameter_heavy_writes() -> None:
    session = _BatchRecordingSession()
    repository = DocumentRepository(cast(AsyncSession, session))
    source_id = uuid4()
    written_at = datetime(2026, 7, 17, 10, 0, tzinfo=UTC)
    record_count = DATABASE_WRITE_BATCH_SIZE + 1
    records = [
        DocumentUpsert(
            source_id=source_id,
            external_id=f"docs/{index}.md",
            document_type="official_documentation",
            title=f"Document {index}",
            content_hash=_hash(f"document-{index}"),
        )
        for index in range(record_count)
    ]

    await repository.upsert_many(records, seen_at=written_at)
    await repository.append_versions(
        [
            DocumentVersionAppend(
                document_id=record.id,
                content_hash=record.content_hash,
                raw_content=record.title,
                parsed_content=record.title,
            )
            for record in records
        ]
    )

    sql = [
        str(statement.compile(dialect=postgresql.dialect())).upper()
        for statement in session.statements
    ]
    assert sum(statement.startswith("INSERT INTO DOCUMENTS") for statement in sql) == 2
    assert sum(statement.startswith("INSERT INTO DOCUMENT_VERSIONS") for statement in sql) == 2


@pytest.mark.asyncio
async def test_chunk_upsert_batches_inserts_before_cross_batch_parent_links() -> None:
    session = _BatchRecordingSession()
    repository = ChunkRepository(cast(AsyncSession, session))
    document_id = uuid4()
    written_at = datetime(2026, 7, 17, 10, 0, tzinfo=UTC)
    parent_id = uuid4()
    record_count = DATABASE_WRITE_BATCH_SIZE + 1
    records = [
        ChunkUpsert(
            id=uuid4(),
            document_id=document_id,
            chunk_index=index,
            parent_chunk_id=parent_id if index == 0 else None,
            document_title="Batched document",
            heading_path=["Batched document"],
            content=f"chunk {index}",
            contextualized_content=f"Batched document\nchunk {index}",
            token_count=3,
            content_hash=_hash(f"chunk-{index}"),
            start_offset=index * 10,
            end_offset=index * 10 + 7,
        )
        for index in range(record_count - 1)
    ]
    records.append(
        ChunkUpsert(
            id=parent_id,
            document_id=document_id,
            chunk_index=record_count - 1,
            document_title="Batched document",
            heading_path=["Batched document"],
            content="parent chunk",
            contextualized_content="Batched document\nparent chunk",
            token_count=3,
            content_hash=_hash("parent-chunk"),
            start_offset=record_count * 10,
            end_offset=record_count * 10 + 12,
        )
    )

    await repository.upsert_many(records, written_at=written_at)

    compiled = [statement.compile(dialect=postgresql.dialect()) for statement in session.statements]
    insert_indexes = [
        index
        for index, statement in enumerate(compiled)
        if str(statement).upper().startswith("INSERT INTO CHUNKS")
    ]
    parent_update_indexes = [
        index
        for index, parameters in enumerate(session.parameters)
        if isinstance(parameters, list)
        and parameters
        and isinstance(parameters[0], dict)
        and "p_parent_chunk_id" in parameters[0]
    ]
    assert len(insert_indexes) == 2
    assert parent_update_indexes
    assert min(parent_update_indexes) > max(insert_indexes)
    for index in insert_indexes:
        parent_values = [
            value for key, value in compiled[index].params.items() if "parent_chunk_id" in key
        ]
        assert parent_values
        assert all(value is None for value in parent_values)


@pytest.mark.asyncio
async def test_large_id_write_is_split_into_bounded_statements() -> None:
    session = _BatchRecordingSession()
    chunk_ids = [uuid4() for _ in range(DATABASE_WRITE_BATCH_SIZE + 1)]

    await ChunkRepository(cast(AsyncSession, session)).soft_delete(
        chunk_ids,
        deleted_at=datetime(2026, 7, 17, 10, 0, tzinfo=UTC),
    )

    assert len(session.statements) == 2


@pytest.mark.asyncio
async def test_chunk_states_for_documents_are_grouped_from_one_query() -> None:
    first_document_id = uuid4()
    second_document_id = uuid4()
    first_chunk = _stored_chunk(first_document_id, 0, "first")
    second_chunk = _stored_chunk(first_document_id, 1, "second")
    session = _FingerprintRecordingSession(
        [
            SimpleNamespace(**first_chunk.model_dump()),
            SimpleNamespace(**second_chunk.model_dump()),
        ]
    )

    grouped = await ChunkRepository(cast(AsyncSession, session)).list_states_for_documents(
        [first_document_id, second_document_id]
    )

    assert len(session.statements) == 1
    assert grouped[first_document_id] == [first_chunk, second_chunk]
    assert grouped[second_document_id] == []
    compiled = session.statements[0].compile(dialect=postgresql.dialect())
    assert "= ANY" in str(compiled).upper()
    assert compiled.params["chunk_state_document_ids"] == [
        first_document_id,
        second_document_id,
    ]


@pytest.mark.asyncio
async def test_cross_run_fingerprints_use_one_query_and_fallback_to_latest_version() -> None:
    persisted_content = "Persisted Deployment rollback evidence."
    fallback_content = "Legacy Service networking evidence."
    exact_hash = normalized_content_hash(persisted_content)
    persisted_simhash = simhash64(persisted_content)
    session = _FingerprintRecordingSession(
        [
            SimpleNamespace(
                external_id="deployment.md",
                source_version="commit-a",
                document_type="official_documentation",
                document_metadata={
                    "deduplication_fingerprint": {
                        "normalized_sha256": exact_hash,
                        "simhash64": persisted_simhash,
                    }
                },
                source_type="github_repository",
                latest_parsed_content="this value must not replace valid metadata",
            ),
            SimpleNamespace(
                external_id="service.md",
                source_version="commit-a",
                document_type="official_documentation",
                document_metadata={"legacy": True},
                source_type="github_repository",
                latest_parsed_content=fallback_content,
            ),
        ]
    )

    fingerprints = await DocumentRepository(
        cast(AsyncSession, session)
    ).list_deduplication_fingerprints(
        uuid4(),
        exclude_external_ids=("incoming.md",),
    )

    assert len(session.statements) == 1
    compiled = session.statements[0].compile(dialect=postgresql.dialect())
    sql = str(compiled).upper()
    assert "DOCUMENT_VERSIONS" in sql
    assert "ALL" in sql
    assert compiled.params
    assert fingerprints[0].exact_hash == exact_hash
    assert fingerprints[0].simhash == persisted_simhash
    assert fingerprints[1].exact_hash == normalized_content_hash(fallback_content)
    assert fingerprints[1].simhash == simhash64(fallback_content)
