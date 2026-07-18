"""Citation, context provenance, and grounded-claim domain contracts."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.domain.chunks import Chunk


class Claim(BaseModel):
    """A factual answer claim that may require supporting evidence."""

    text: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    requires_citation: bool = True
    citation_numbers: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_offsets(self) -> Claim:
        if self.end_offset < self.start_offset:
            msg = "end_offset must not precede start_offset"
            raise ValueError(msg)
        return self


class ContextChunkReference(BaseModel):
    """Mapping from one context excerpt back to an immutable source chunk."""

    chunk_id: UUID
    document_id: UUID
    heading_path: list[str] = Field(default_factory=list)
    content: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    included_start_offset: int = Field(ge=0)
    included_end_offset: int = Field(ge=0)
    context_start_offset: int = Field(ge=0)
    context_end_offset: int = Field(ge=0)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    retrieval_score: float | None = None
    truncated: bool = False

    @model_validator(mode="after")
    def validate_offset_ranges(self) -> ContextChunkReference:
        ranges = (
            (self.start_offset, self.end_offset, "source"),
            (self.included_start_offset, self.included_end_offset, "included source"),
            (self.context_start_offset, self.context_end_offset, "context"),
        )
        for start, end, label in ranges:
            if end < start:
                msg = f"{label} end offset must not precede start offset"
                raise ValueError(msg)
        if not (
            self.start_offset
            <= self.included_start_offset
            <= self.included_end_offset
            <= self.end_offset
        ):
            msg = "included source offsets must stay within the original chunk"
            raise ValueError(msg)
        return self


class ContextSource(BaseModel):
    """One numbered, possibly merged source supplied to the answer model."""

    number: int = Field(ge=1)
    document_id: UUID
    title: str
    section: str
    url: str | None = None
    document_type: str
    content: str
    chunks: list[ContextChunkReference] = Field(min_length=1)
    token_count: int = Field(ge=0)


class BuiltContext(BaseModel):
    """Bounded prompt context plus the authoritative citation-number mapping."""

    text: str
    sources: list[ContextSource] = Field(default_factory=list)
    token_count: int = Field(ge=0)
    skipped_chunk_ids: list[UUID] = Field(default_factory=list)
    truncated: bool = False

    def source(self, number: int) -> ContextSource | None:
        """Return a numbered source without trusting model-produced identifiers."""
        return next((source for source in self.sources if source.number == number), None)


class Citation(BaseModel):
    """Public citation attached to a generated answer."""

    number: int = Field(ge=1)
    chunk_id: UUID
    document_id: UUID
    title: str
    section: str
    url: str | None = None
    quoted_text: str
    document_type: str
    score: float = Field(ge=-1, le=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    valid: bool | None = None
    validation_score: float | None = Field(default=None, ge=0, le=1)


class CitationValidation(BaseModel):
    """Validation result mapping one answer claim to its cited evidence."""

    claim: Claim
    citation: Citation | None
    supported: bool
    score: float = Field(ge=0, le=1)
    reason: str


class CitationMarkerResult(BaseModel):
    """Outcome for one literal citation marker in answer order.

    Keeping marker-level outcomes is important because the public citation list
    intentionally omits rejected and forged citations. Evaluation and audit
    code must still be able to count those markers in its precision denominator.
    """

    number: int = Field(ge=1)
    claim: Claim | None = None
    citation: Citation | None = None
    supported: bool
    score: float = Field(ge=0, le=1)
    reason: str


class CitationJudgeDecision(BaseModel):
    """Strict result expected from an optional semantic citation validator."""

    supported: bool
    score: float = Field(ge=0, le=1)
    reason: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    model: str | None = None


class CitationReport(BaseModel):
    """Parsed citations and deterministic grounding/coverage measurements."""

    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    validations: list[CitationValidation] = Field(default_factory=list)
    marker_results: list[CitationMarkerResult] = Field(default_factory=list)
    invalid_citation_numbers: list[int] = Field(default_factory=list)
    citation_marker_count: int = Field(ge=0)
    citation_precision: float = Field(ge=0, le=1)
    citation_recall: float = Field(ge=0, le=1)
    claim_coverage: float = Field(ge=0, le=1)
    citation_correctness: float = Field(ge=0, le=1)
    citation_completeness: float = Field(ge=0, le=1)
    judge_prompt_tokens: int = Field(default=0, ge=0)
    judge_completion_tokens: int = Field(default=0, ge=0)
    judge_latency_ms: float = Field(default=0, ge=0)
    judge_model: str | None = None

    @model_validator(mode="after")
    def validate_marker_results(self) -> CitationReport:
        if self.marker_results and len(self.marker_results) != self.citation_marker_count:
            msg = "marker results must align with the citation marker count"
            raise ValueError(msg)
        return self


class CitationDetail(BaseModel):
    """Read model for a cited chunk and its document-local neighbours."""

    chunk: Chunk
    previous_chunk: Chunk | None = None
    next_chunk: Chunk | None = None
    document_metadata: dict[str, Any] = Field(default_factory=dict)
    source_url: str | None = None
