"""Validated HTTP schemas for chat, streaming, and conversation history."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.chat import ChatLatency, ChatResult, ChatUsage
from app.domain.citations import Citation
from app.domain.retrieval import QueryFilters, RetrievalMode


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=50_000)
    conversation_id: UUID | None = None
    filters: QueryFilters = Field(default_factory=QueryFilters)
    top_k: int | None = Field(default=None, ge=1, le=100)
    mode: RetrievalMode | None = None
    stream: bool = False
    debug: bool = False


class RetrievalSummary(BaseModel):
    keyword_count: int = Field(ge=0)
    vector_count: int = Field(ge=0)
    reranked_count: int = Field(ge=0)
    degraded_channels: list[str] = Field(default_factory=list)
    reranker_degraded: bool = False


class ChatResponse(BaseModel):
    request_id: UUID
    conversation_id: UUID
    message_id: UUID
    original_query: str
    rewritten_query: str
    answer: str
    answerable: bool
    confidence: float = Field(ge=0, le=1)
    citations: list[Citation]
    retrieval: RetrievalSummary
    usage: ChatUsage
    latency: ChatLatency
    no_answer_reason: str
    debug: dict[str, Any] | None = None

    @classmethod
    def from_result(cls, result: ChatResult, *, debug: bool = False) -> ChatResponse:
        details: dict[str, Any] | None = None
        if debug:
            details = {
                "retrieval_confidence": result.no_answer.retrieval_confidence,
                "evidence_sufficiency_score": result.no_answer.evidence_sufficiency_score,
                "diagnostics": result.no_answer.diagnostics,
                "citation_metrics": (
                    result.citation_report.model_dump()
                    if result.citation_report is not None
                    else None
                ),
            }
        return cls(
            request_id=result.request_id,
            conversation_id=result.conversation_id,
            message_id=result.message_id,
            original_query=result.original_query,
            rewritten_query=result.rewritten_query,
            answer=result.answer,
            answerable=result.answerable,
            confidence=result.confidence,
            citations=result.citations,
            retrieval=RetrievalSummary(
                keyword_count=result.retrieval.keyword_count,
                vector_count=result.retrieval.vector_count,
                reranked_count=result.retrieval.reranked_count,
                degraded_channels=result.retrieval.degraded_channels,
                reranker_degraded=result.retrieval.reranker_degraded,
            ),
            usage=result.usage,
            latency=result.latency,
            no_answer_reason=result.no_answer.reason,
            debug=details,
        )


class ConversationMessageResponse(BaseModel):
    id: UUID
    role: str
    original_query: str | None
    rewritten_query: str | None
    content: str
    token_usage: dict[str, Any]
    cost: float | None
    latency_ms: int | None
    created_at: datetime


class ConversationResponse(BaseModel):
    id: UUID
    user_id: str | None
    title: str | None
    created_at: datetime
    updated_at: datetime
    messages: list[ConversationMessageResponse]
