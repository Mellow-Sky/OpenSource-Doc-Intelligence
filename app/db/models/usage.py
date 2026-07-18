"""Provider usage, latency, and nullable cost accounting records."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, Float, Index, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class UsageRecord(Base):
    __tablename__ = "usage_records"
    __table_args__ = (
        Index("ix_usage_records_request_operation", "request_id", "operation"),
        CheckConstraint(
            "input_text_count >= 0",
            name="input_text_count_nonnegative",
        ),
        CheckConstraint(
            "input_character_count >= 0",
            name="input_character_count_nonnegative",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    request_id: Mapped[UUID] = mapped_column(nullable=False, index=True)
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    input_text_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    input_character_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    estimated_cost: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
