"""Detailed retrieval-run and answer-citation audit entities."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class RetrievalRun(TimestampMixin, Base):
    __tablename__ = "retrieval_runs"
    __table_args__ = (Index("ix_retrieval_runs_message", "message_id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    message_id: Mapped[UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[str] = mapped_column(Text, nullable=False)
    filters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    keyword_latency_ms: Mapped[float] = mapped_column(Float, default=0, server_default="0")
    vector_latency_ms: Mapped[float] = mapped_column(Float, default=0, server_default="0")
    rerank_latency_ms: Mapped[float] = mapped_column(Float, default=0, server_default="0")
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0, server_default="0")
    retrieved_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    reranked_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    no_answer_score: Mapped[float | None] = mapped_column(Float)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default="{}"
    )

    results: Mapped[list[RetrievalResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class RetrievalResult(Base):
    __tablename__ = "retrieval_results"
    __table_args__ = (
        UniqueConstraint("retrieval_run_id", "chunk_id", name="uq_retrieval_results_run_chunk"),
        Index("ix_retrieval_results_run_rank", "retrieval_run_id", "rerank_rank"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    retrieval_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("retrieval_runs.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    keyword_rank: Mapped[int | None] = mapped_column(Integer)
    vector_rank: Mapped[int | None] = mapped_column(Integer)
    fused_rank: Mapped[int | None] = mapped_column(Integer)
    rerank_rank: Mapped[int | None] = mapped_column(Integer)
    keyword_score: Mapped[float | None] = mapped_column(Float)
    vector_score: Mapped[float | None] = mapped_column(Float)
    fused_score: Mapped[float | None] = mapped_column(Float)
    rerank_score: Mapped[float | None] = mapped_column(Float)
    selected_for_context: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false"
    )

    run: Mapped[RetrievalRun] = relationship(back_populates="results")


class AnswerCitation(Base):
    __tablename__ = "answer_citations"
    __table_args__ = (
        UniqueConstraint(
            "message_id",
            "citation_number",
            "chunk_id",
            name="uq_answer_citations_message_number_chunk",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    message_id: Mapped[UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    citation_number: Mapped[int] = mapped_column(Integer, nullable=False)
    quoted_text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    citation_valid: Mapped[bool | None] = mapped_column(Boolean)
    validation_score: Mapped[float | None] = mapped_column(Float)
