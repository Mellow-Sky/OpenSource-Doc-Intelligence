"""Application-level chat result contracts independent of FastAPI transport."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.citations import Citation, CitationReport
from app.domain.retrieval import NoAnswerDecision, RetrievalOutcome


class ChatUsage(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class ChatLatency(BaseModel):
    rewrite_ms: float = Field(default=0, ge=0)
    keyword_retrieval_ms: float = Field(default=0, ge=0)
    vector_retrieval_ms: float = Field(default=0, ge=0)
    fusion_ms: float = Field(default=0, ge=0)
    rerank_ms: float = Field(default=0, ge=0)
    generation_ms: float = Field(default=0, ge=0)
    citation_validation_ms: float = Field(default=0, ge=0)
    total_ms: float = Field(default=0, ge=0)


class ChatResult(BaseModel):
    request_id: UUID
    conversation_id: UUID
    message_id: UUID
    original_query: str
    rewritten_query: str
    answer: str
    answerable: bool
    confidence: float = Field(ge=0, le=1)
    citations: list[Citation] = Field(default_factory=list)
    retrieval: RetrievalOutcome
    usage: ChatUsage = Field(default_factory=ChatUsage)
    latency: ChatLatency = Field(default_factory=ChatLatency)
    no_answer: NoAnswerDecision
    citation_report: CitationReport | None = None
