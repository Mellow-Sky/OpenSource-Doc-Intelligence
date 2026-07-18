"""PostgreSQL full-text and pgvector retrieval persistence adapter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import isfinite
from typing import Any, cast

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, Text, bindparam, func, or_, select
from sqlalchemy import cast as sql_cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import ColumnElement

from app.core.exceptions import RetrievalError
from app.db.models.source_document import Chunk, Document, Source
from app.domain.retrieval import QueryFilters, RetrievalCandidate

MAX_RETRIEVAL_LIMIT = 200
TRIGRAM_SCORE_WEIGHT = 0.25
SIMPLE_FTS_SCORE_WEIGHT = 0.75


class RetrievalRepository:
    """Execute ranked retrieval in one query per retrieval channel.

    The repository deliberately returns domain candidates rather than ORM objects. This
    keeps the application layer independent of SQLAlchemy and avoids follow-up queries for
    document/source metadata.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def keyword_search(
        self,
        query: str,
        *,
        filters: QueryFilters | None = None,
        limit: int = 30,
    ) -> list[RetrievalCandidate]:
        """Return weighted PostgreSQL FTS candidates with deterministic ranks."""
        normalized_query = _validate_query(query)
        validated_limit = _validate_limit(limit)
        statement = build_keyword_statement(
            filters=filters or QueryFilters(),
            limit=validated_limit,
        )
        try:
            result = await self._session.execute(
                statement,
                {"search_query": normalized_query},
            )
        except SQLAlchemyError as exc:
            raise RetrievalError("PostgreSQL keyword retrieval failed") from exc
        return _to_candidates(
            result.mappings().all(),
            score_column="keyword_score",
            rank_field="keyword_rank",
        )

    async def vector_search(
        self,
        embedding: Sequence[float],
        *,
        filters: QueryFilters | None = None,
        limit: int = 30,
        embedding_model: str | None = None,
    ) -> list[RetrievalCandidate]:
        """Return cosine-similarity candidates using the pgvector HNSW ordering."""
        normalized_embedding = _validate_embedding(embedding)
        validated_limit = _validate_limit(limit)
        statement = build_vector_statement(
            filters=filters or QueryFilters(),
            limit=validated_limit,
            embedding_dimension=len(normalized_embedding),
            embedding_model=embedding_model,
        )
        try:
            result = await self._session.execute(
                statement,
                {"query_embedding": normalized_embedding},
            )
        except SQLAlchemyError as exc:
            raise RetrievalError("PostgreSQL vector retrieval failed") from exc
        return _to_candidates(
            result.mappings().all(),
            score_column="vector_score",
            rank_field="vector_rank",
        )


def build_keyword_statement(*, filters: QueryFilters, limit: int) -> Select[Any]:
    """Build a parameterized weighted FTS statement for compilation and execution."""
    validated_limit = _validate_limit(limit)
    query_parameter = bindparam("search_query", type_=Text())
    english_query = func.websearch_to_tsquery("english", query_parameter)
    simple_query = func.plainto_tsquery("simple", query_parameter)

    english_score = func.ts_rank_cd(Chunk.search_vector, english_query, 32)
    simple_score = func.ts_rank_cd(Chunk.search_vector, simple_query, 32) * SIMPLE_FTS_SCORE_WEIGHT
    trigram_score = func.similarity(Chunk.content, query_parameter) * TRIGRAM_SCORE_WEIGHT
    score = func.greatest(english_score, simple_score, trigram_score).label("keyword_score")

    statement = (
        select(*_candidate_columns(), score)
        .select_from(Chunk)
        .join(Document, Document.id == Chunk.document_id)
        .join(Source, Source.id == Document.source_id)
        .where(
            *_active_predicates(filters),
            or_(
                Chunk.search_vector.op("@@")(english_query),
                Chunk.search_vector.op("@@")(simple_query),
                Chunk.content.op("%")(query_parameter),
            ),
        )
        .order_by(score.desc(), Chunk.id.asc())
        .limit(validated_limit)
    )
    return statement


def build_vector_statement(
    *,
    filters: QueryFilters,
    limit: int,
    embedding_dimension: int,
    embedding_model: str | None = None,
) -> Select[Any]:
    """Build a parameterized cosine-distance query compatible with HNSW indexes."""
    validated_limit = _validate_limit(limit)
    if embedding_dimension < 1:
        raise ValueError("embedding_dimension must be positive")

    embedding_parameter = bindparam(
        "query_embedding",
        type_=Vector(embedding_dimension),
    )
    cosine_distance = Chunk.embedding.cosine_distance(embedding_parameter)
    score = sql_cast(1.0 - cosine_distance, Float).label("vector_score")
    predicates = [
        *_active_predicates(filters),
        Chunk.embedding.is_not(None),
        Chunk.embedding_dimension == embedding_dimension,
    ]
    if embedding_model:
        predicates.append(Chunk.embedding_model == embedding_model)

    # Keep distance as the only ORDER BY expression. pgvector can then serve this
    # bounded query directly from the vector_cosine_ops HNSW index.
    return (
        select(*_candidate_columns(), score)
        .select_from(Chunk)
        .join(Document, Document.id == Chunk.document_id)
        .join(Source, Source.id == Document.source_id)
        .where(*predicates)
        .order_by(cosine_distance.asc())
        .limit(validated_limit)
    )


def _candidate_columns() -> tuple[Any, ...]:
    return (
        Chunk.id.label("chunk_id"),
        Chunk.document_id,
        Document.title.label("document_title"),
        Document.document_type,
        Chunk.heading_path,
        Chunk.content,
        Document.canonical_url,
        Document.metadata_.label("document_metadata"),
        Chunk.metadata_.label("chunk_metadata"),
        Document.source_id,
        Source.name.label("source_name"),
        Source.source_type,
        Chunk.start_offset,
        Chunk.end_offset,
    )


def _active_predicates(filters: QueryFilters) -> list[ColumnElement[bool]]:
    predicates: list[ColumnElement[bool]] = [
        Chunk.deleted_at.is_(None),
        Document.deleted_at.is_(None),
        Document.status == "active",
        Source.enabled.is_(True),
    ]
    if filters.source_ids:
        predicates.append(Document.source_id.in_(filters.source_ids))
    if filters.document_types:
        predicates.append(Document.document_type.in_(filters.document_types))
    if filters.versions:
        version_values = _version_variants(filters.versions)
        predicates.append(
            or_(
                Document.source_version.in_(version_values),
                _metadata_any(("version", "kubernetes_version"), version_values),
            )
        )
    if filters.api_groups:
        predicates.append(_metadata_any(("api_group", "apiGroup"), filters.api_groups))
    if filters.api_versions:
        predicates.append(
            _metadata_any(("api_version", "apiVersion", "version"), filters.api_versions)
        )
    if filters.kinds:
        predicates.append(_metadata_any(("kind",), filters.kinds))
    if filters.issue_states:
        predicates.append(_metadata_any(("state", "issue_state"), filters.issue_states))
    if filters.release_versions:
        release_values = _version_variants(filters.release_versions)
        predicates.append(
            or_(
                Document.source_version.in_(release_values),
                _metadata_any(
                    ("release_version", "tag", "version"),
                    release_values,
                ),
            )
        )
    for key, value in filters.metadata.items():
        # Query preprocessing records a group/version token such as apps/v1 as
        # an API-version list. Loaders store the canonical singular `version`
        # field, so translate that structured filter instead of requiring an
        # exact JSON array shape.
        if key == "api_versions" and isinstance(value, list):
            api_versions = [str(item) for item in value if str(item).strip()]
            if api_versions:
                predicates.append(_metadata_any(("api_version", "version"), api_versions))
            continue
        predicates.append(
            or_(
                Document.metadata_.contains({key: value}),
                Chunk.metadata_.contains({key: value}),
            )
        )
    return predicates


def _metadata_any(keys: Sequence[str], values: Sequence[str]) -> ColumnElement[bool]:
    clauses = [
        metadata_column.contains(sql_cast({key: value}, JSONB))
        for key in keys
        for value in values
        for metadata_column in (Document.metadata_, Chunk.metadata_)
    ]
    return or_(*clauses)


def _version_variants(values: Sequence[str]) -> list[str]:
    """Match configured version values with or without the conventional `v` prefix."""
    variants: list[str] = []
    for raw_value in values:
        value = raw_value.strip()
        if not value:
            continue
        for variant in (value, value[1:] if value.casefold().startswith("v") else f"v{value}"):
            if variant and variant not in variants:
                variants.append(variant)
    return variants


def _to_candidates(
    rows: Iterable[RowMapping],
    *,
    score_column: str,
    rank_field: str,
) -> list[RetrievalCandidate]:
    candidates: list[RetrievalCandidate] = []
    for rank, row in enumerate(rows, start=1):
        document_metadata = dict(cast(dict[str, Any] | None, row["document_metadata"]) or {})
        chunk_metadata = dict(cast(dict[str, Any] | None, row["chunk_metadata"]) or {})
        metadata = document_metadata | chunk_metadata
        metadata.setdefault("source_id", str(row["source_id"]))
        metadata.setdefault("source_name", str(row["source_name"]))
        metadata.setdefault("source_type", str(row["source_type"]))
        rank_values = {rank_field: rank}
        candidates.append(
            RetrievalCandidate(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                document_title=str(row["document_title"]),
                document_type=str(row["document_type"]),
                heading_path=list(cast(list[str] | None, row["heading_path"]) or []),
                content=str(row["content"]),
                canonical_url=cast(str | None, row["canonical_url"]),
                metadata=metadata,
                start_offset=int(row["start_offset"]),
                end_offset=int(row["end_offset"]),
                **rank_values,
                **{score_column: float(row[score_column])},
            )
        )
    return candidates


def _validate_query(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must not be blank")
    return normalized


def _validate_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_RETRIEVAL_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_RETRIEVAL_LIMIT}")
    return limit


def _validate_embedding(embedding: Sequence[float]) -> list[float]:
    if not embedding:
        raise ValueError("embedding must not be empty")
    values = [float(value) for value in embedding]
    if not all(isfinite(value) for value in values):
        raise ValueError("embedding values must be finite")
    return values
