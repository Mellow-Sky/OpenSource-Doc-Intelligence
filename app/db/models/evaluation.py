"""Evaluation dataset, run, and per-sample result records."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class EvaluationCase(TimestampMixin, Base):
    __tablename__ = "evaluation_cases"
    __table_args__ = (
        Index("ix_evaluation_cases_dataset_category", "dataset_name", "category"),
        UniqueConstraint(
            "dataset_name",
            "external_id",
            name="uq_evaluation_cases_dataset_external_id",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    conversation_history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default="[]"
    )
    reference_answer: Mapped[str] = mapped_column(Text, nullable=False)
    relevant_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    expected_citations: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    answerable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    difficulty: Mapped[str] = mapped_column(String(32), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str | None] = mapped_column(String(64))
    human_reviewed: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default="{}"
    )

    results: Mapped[list[EvaluationResult]] = relationship(back_populates="case")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (Index("ix_evaluation_runs_status_started", "status", "started_at"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    report_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="pending", server_default="pending")

    results: Mapped[list[EvaluationResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    __table_args__ = (
        Index("ix_evaluation_results_run_case", "evaluation_run_id", "evaluation_case_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    evaluation_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False
    )
    evaluation_case_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("evaluation_cases.id", ondelete="SET NULL")
    )
    case_external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    generated_answer: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_answerable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    retrieved_results: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default="[]"
    )
    citations: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    latency: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    usage: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    error: Mapped[str | None] = mapped_column(Text)

    run: Mapped[EvaluationRun] = relationship(back_populates="results")
    case: Mapped[EvaluationCase | None] = relationship(back_populates="results")
