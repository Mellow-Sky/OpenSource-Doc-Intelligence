"""Domain contracts for query analysis, hybrid retrieval, and result tracing."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite
from typing import Annotated, Any
from uuid import UUID

import orjson
from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

_MAX_FILTER_ITEMS = 50
_MAX_METADATA_KEYS = 20
_MAX_METADATA_LIST_ITEMS = 50
_MAX_METADATA_BYTES = 8_192
_MAX_METADATA_STRING_LENGTH = 512
FilterString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    VECTOR = "vector"


class FusionMode(StrEnum):
    RRF = "rrf"
    WEIGHTED = "weighted"


class QueryFilters(BaseModel):
    """Structured filters extracted from a query or supplied by API callers."""

    source_ids: list[UUID] = Field(default_factory=list, max_length=_MAX_FILTER_ITEMS)
    document_types: list[FilterString] = Field(
        default_factory=list,
        max_length=_MAX_FILTER_ITEMS,
    )
    versions: list[FilterString] = Field(default_factory=list, max_length=_MAX_FILTER_ITEMS)
    api_groups: list[FilterString] = Field(default_factory=list, max_length=_MAX_FILTER_ITEMS)
    api_versions: list[FilterString] = Field(default_factory=list, max_length=_MAX_FILTER_ITEMS)
    kinds: list[FilterString] = Field(default_factory=list, max_length=_MAX_FILTER_ITEMS)
    issue_states: list[FilterString] = Field(default_factory=list, max_length=_MAX_FILTER_ITEMS)
    release_versions: list[FilterString] = Field(
        default_factory=list,
        max_length=_MAX_FILTER_ITEMS,
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_filter(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Keep JSONB predicates flat and bounded before SQL is constructed."""

        if len(value) > _MAX_METADATA_KEYS:
            raise ValueError(f"metadata cannot contain more than {_MAX_METADATA_KEYS} keys")
        for key, item in value.items():
            if not key.strip() or len(key) > 128:
                raise ValueError("metadata keys must contain 1 to 128 characters")
            _validate_metadata_filter_value(item)
        if len(orjson.dumps(value)) > _MAX_METADATA_BYTES:
            raise ValueError(f"metadata cannot exceed {_MAX_METADATA_BYTES} serialized bytes")
        return value


def _validate_metadata_filter_value(value: Any) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if not value.strip() or len(value) > _MAX_METADATA_STRING_LENGTH:
            raise ValueError(
                f"metadata strings must contain 1 to {_MAX_METADATA_STRING_LENGTH} characters"
            )
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return
    if isinstance(value, list):
        if len(value) > _MAX_METADATA_LIST_ITEMS:
            raise ValueError(
                f"metadata lists cannot contain more than {_MAX_METADATA_LIST_ITEMS} items"
            )
        for item in value:
            if isinstance(item, (list, dict)):
                raise ValueError("nested metadata filters are not supported")
            _validate_metadata_filter_value(item)
        return
    raise ValueError("metadata values must be JSON scalars or flat scalar lists")


class RetrievalQuery(BaseModel):
    """Normalized query passed into the retrieval layer."""

    original: str = Field(min_length=1)
    rewritten: str = Field(min_length=1)
    language: str = "en"
    filters: QueryFilters = Field(default_factory=QueryFilters)
    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=8, ge=1, le=100)


class RetrievalCandidate(BaseModel):
    """One candidate with complete ranking provenance across retrieval stages."""

    chunk_id: UUID
    document_id: UUID
    document_title: str
    document_type: str
    heading_path: list[str] = Field(default_factory=list)
    content: str
    canonical_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    keyword_rank: int | None = Field(default=None, ge=1)
    vector_rank: int | None = Field(default=None, ge=1)
    fused_rank: int | None = Field(default=None, ge=1)
    rerank_rank: int | None = Field(default=None, ge=1)
    keyword_score: float | None = None
    vector_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    selected_for_context: bool = False
    start_offset: int = Field(default=0, ge=0)
    end_offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> RetrievalCandidate:
        if self.end_offset < self.start_offset:
            msg = "end_offset must not precede start_offset"
            raise ValueError(msg)
        return self


class RetrievalTimings(BaseModel):
    rewrite_ms: float = Field(default=0, ge=0)
    keyword_ms: float = Field(default=0, ge=0)
    vector_ms: float = Field(default=0, ge=0)
    fusion_ms: float = Field(default=0, ge=0)
    rerank_ms: float = Field(default=0, ge=0)
    total_ms: float = Field(default=0, ge=0)


class RetrievalOutcome(BaseModel):
    query: RetrievalQuery
    candidates: list[RetrievalCandidate]
    trace_candidates: list[RetrievalCandidate] = Field(default_factory=list)
    keyword_count: int = Field(ge=0)
    vector_count: int = Field(ge=0)
    reranked_count: int = Field(ge=0)
    timings: RetrievalTimings = Field(default_factory=RetrievalTimings)
    degraded_channels: list[str] = Field(default_factory=list)
    reranker_degraded: bool = False
    reranker_reason: str | None = None
    embedding_prompt_tokens: int = Field(default=0, ge=0)
    embedding_model: str | None = None


class EvidenceSufficiency(BaseModel):
    """Optional lightweight judge result for retrieval gray-zone decisions."""

    sufficient: bool
    score: float = Field(ge=0, le=1)
    reason: str = ""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    model: str | None = None


class NoAnswerDecision(BaseModel):
    """Auditable combined decision made before or after answer generation."""

    answerable: bool
    confidence: float = Field(ge=0, le=1)
    reason: str
    retrieval_confidence: float = Field(ge=0, le=1)
    evidence_sufficiency_score: float = Field(ge=0, le=1)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
