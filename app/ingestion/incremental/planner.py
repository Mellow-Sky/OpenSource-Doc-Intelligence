"""Deterministic incremental-sync decisions with no database dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from app.ingestion.incremental.models import (
    ChunkAction,
    ChunkDecision,
    ChunkSyncPlan,
    DocumentAction,
    DocumentDecision,
    DocumentSyncPlan,
    IncomingChunkState,
    IncomingDocumentState,
    StoredChunkState,
    StoredDocumentState,
    SyncStats,
)


def decide_document(
    incoming: IncomingDocumentState,
    existing: StoredDocumentState | None,
) -> DocumentDecision:
    """Classify an incoming document by its composite identity and hash."""
    if existing is None:
        return DocumentDecision(
            incoming=incoming,
            action=DocumentAction.CREATE,
            reason="source_id and external_id were not previously indexed",
        )
    if (incoming.source_id, incoming.external_id) != (
        existing.source_id,
        existing.external_id,
    ):
        raise ValueError("existing document does not match incoming composite identity")
    if existing.deleted_at is not None:
        return DocumentDecision(
            incoming=incoming,
            action=DocumentAction.UPDATE,
            document_id=existing.id,
            reason="previously deleted document reappeared and must be restored",
        )
    if incoming.content_hash != existing.content_hash:
        return DocumentDecision(
            incoming=incoming,
            action=DocumentAction.UPDATE,
            document_id=existing.id,
            reason="normalized content hash changed",
        )
    return DocumentDecision(
        incoming=incoming,
        action=DocumentAction.UNCHANGED,
        document_id=existing.id,
        reason="normalized content hash is unchanged",
    )


def plan_document_sync(
    source_id: UUID,
    incoming_documents: Sequence[IncomingDocumentState],
    stored_documents: Sequence[StoredDocumentState],
) -> DocumentSyncPlan:
    """Plan upserts and soft deletes for an authoritative source snapshot."""
    incoming_by_external: dict[str, IncomingDocumentState] = {}
    for incoming in incoming_documents:
        if incoming.source_id != source_id:
            raise ValueError("incoming document belongs to a different source")
        if incoming.external_id in incoming_by_external:
            raise ValueError(f"duplicate incoming external_id: {incoming.external_id}")
        incoming_by_external[incoming.external_id] = incoming

    stored_by_external: dict[str, StoredDocumentState] = {}
    for stored in stored_documents:
        if stored.source_id != source_id:
            raise ValueError("stored document belongs to a different source")
        if stored.external_id in stored_by_external:
            raise ValueError(f"duplicate stored external_id: {stored.external_id}")
        stored_by_external[stored.external_id] = stored

    decisions = [
        decide_document(incoming, stored_by_external.get(incoming.external_id))
        for incoming in incoming_documents
    ]
    seen = set(incoming_by_external)
    missing = [
        stored.id
        for stored in stored_documents
        if stored.deleted_at is None and stored.external_id not in seen
    ]
    stats = SyncStats(
        scanned=len(incoming_documents),
        created=sum(decision.action is DocumentAction.CREATE for decision in decisions),
        updated=sum(decision.action is DocumentAction.UPDATE for decision in decisions),
        unchanged=sum(decision.action is DocumentAction.UNCHANGED for decision in decisions),
        deleted=len(missing),
    )
    return DocumentSyncPlan(
        decisions=decisions,
        missing_document_ids=missing,
        stats=stats,
    )


def plan_chunk_sync(
    document_id: UUID,
    incoming_chunks: Sequence[IncomingChunkState],
    stored_chunks: Sequence[StoredChunkState],
) -> ChunkSyncPlan:
    """Diff stable chunk indexes and hashes, preserving unchanged embeddings."""
    incoming_by_index: dict[int, IncomingChunkState] = {}
    for incoming in incoming_chunks:
        if incoming.document_id != document_id:
            raise ValueError("incoming chunk belongs to a different document")
        if incoming.chunk_index in incoming_by_index:
            raise ValueError(f"duplicate incoming chunk_index: {incoming.chunk_index}")
        incoming_by_index[incoming.chunk_index] = incoming

    stored_by_index: dict[int, StoredChunkState] = {}
    for stored in stored_chunks:
        if stored.document_id != document_id:
            raise ValueError("stored chunk belongs to a different document")
        if stored.chunk_index in stored_by_index:
            raise ValueError(f"duplicate stored chunk_index: {stored.chunk_index}")
        stored_by_index[stored.chunk_index] = stored

    active = sorted(
        (stored for stored in stored_chunks if stored.deleted_at is None),
        key=lambda item: item.chunk_index,
    )
    incoming_ordered = list(incoming_chunks)
    decisions_by_index: dict[int, ChunkDecision] = {}
    used_stored_ids: set[UUID] = set()

    for incoming_position, stored_position in _lcs_hash_matches(incoming_ordered, active):
        incoming = incoming_ordered[incoming_position]
        existing_chunk = active[stored_position]
        used_stored_ids.add(existing_chunk.id)
        moved = incoming.chunk_index != existing_chunk.chunk_index
        decisions_by_index[incoming.chunk_index] = ChunkDecision(
            incoming=incoming,
            action=ChunkAction.MOVE if moved else ChunkAction.UNCHANGED,
            chunk_id=existing_chunk.id,
            reason=(
                "unchanged embedding input moved to a new chunk index"
                if moved
                else "chunk embedding input hash is unchanged"
            ),
        )

    deleted_by_hash: dict[str, list[StoredChunkState]] = {}
    for stored in sorted(stored_chunks, key=lambda item: item.chunk_index):
        if stored.deleted_at is not None:
            deleted_by_hash.setdefault(stored.content_hash, []).append(stored)

    for incoming in incoming_ordered:
        if incoming.chunk_index in decisions_by_index:
            continue
        restored = next(
            (
                item
                for item in deleted_by_hash.get(incoming.content_hash, [])
                if item.id not in used_stored_ids
            ),
            None,
        )
        if restored is not None:
            used_stored_ids.add(restored.id)
            decisions_by_index[incoming.chunk_index] = ChunkDecision(
                incoming=incoming,
                action=ChunkAction.RESTORE,
                chunk_id=restored.id,
                reason="unchanged deleted chunk reappeared; reuse its existing embedding",
            )

    for incoming in incoming_ordered:
        if incoming.chunk_index in decisions_by_index:
            continue
        same_index_chunk = stored_by_index.get(incoming.chunk_index)
        if (
            same_index_chunk is not None
            and same_index_chunk.deleted_at is None
            and same_index_chunk.id not in used_stored_ids
        ):
            used_stored_ids.add(same_index_chunk.id)
            decisions_by_index[incoming.chunk_index] = ChunkDecision(
                incoming=incoming,
                action=ChunkAction.UPDATE,
                chunk_id=same_index_chunk.id,
                reason="chunk embedding input changed at the same structural position",
            )
        else:
            decisions_by_index[incoming.chunk_index] = ChunkDecision(
                incoming=incoming,
                action=ChunkAction.CREATE,
                reason="chunk has no reusable embedding-input identity",
            )

    decisions = [decisions_by_index[item.chunk_index] for item in incoming_ordered]
    removed = [stored.id for stored in active if stored.id not in used_stored_ids]
    stats = SyncStats(
        chunks_created=sum(decision.action is ChunkAction.CREATE for decision in decisions),
        chunks_updated=sum(
            decision.action in {ChunkAction.UPDATE, ChunkAction.MOVE, ChunkAction.RESTORE}
            for decision in decisions
        ),
        chunks_deleted=len(removed),
    )
    return ChunkSyncPlan(decisions=decisions, removed_chunk_ids=removed, stats=stats)


def _lcs_hash_matches(
    incoming: Sequence[IncomingChunkState],
    stored: Sequence[StoredChunkState],
) -> list[tuple[int, int]]:
    """Match unchanged embedding inputs in order, including index shifts and duplicates."""

    lengths = [[0] * (len(stored) + 1) for _ in range(len(incoming) + 1)]
    for left in range(len(incoming) - 1, -1, -1):
        for right in range(len(stored) - 1, -1, -1):
            if incoming[left].content_hash == stored[right].content_hash:
                lengths[left][right] = 1 + lengths[left + 1][right + 1]
            else:
                lengths[left][right] = max(lengths[left + 1][right], lengths[left][right + 1])

    matches: list[tuple[int, int]] = []
    left = right = 0
    while left < len(incoming) and right < len(stored):
        if incoming[left].content_hash == stored[right].content_hash:
            matches.append((left, right))
            left += 1
            right += 1
        elif lengths[left + 1][right] >= lengths[left][right + 1]:
            left += 1
        else:
            right += 1
    return matches
