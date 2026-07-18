"""Conversation and generated message persistence models."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Conversation(TimestampMixin, Base):
    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[str | None] = mapped_column(String(255), index=True)
    title: Mapped[str | None] = mapped_column(String(512))

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(TimestampMixin, Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    original_query: Mapped[str | None] = mapped_column(Text)
    rewritten_query: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_usage: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    cost: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    latency_ms: Mapped[int | None] = mapped_column(Integer)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
