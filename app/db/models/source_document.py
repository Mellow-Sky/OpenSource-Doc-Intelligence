"""Knowledge source, versioned document, and retrievable chunk entities."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Source(TimestampMixin, Base):
    """Configurable upstream knowledge source."""

    __tablename__ = "sources"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    base_url: Mapped[str | None] = mapped_column(Text)
    repository: Mapped[str | None] = mapped_column(String(512))
    branch: Mapped[str | None] = mapped_column(String(255))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")

    documents: Mapped[list[Document]] = relationship(back_populates="source")


class Document(TimestampMixin, Base):
    """Current state of one logical upstream document."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("source_id", "external_id", name="uq_documents_source_external"),
        Index("ix_documents_metadata_gin", "metadata", postgresql_using="gin"),
        Index("ix_documents_active", "source_id", "status", "deleted_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    source_id: Mapped[UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
    )
    external_id: Mapped[str] = mapped_column(String(1024), nullable=False)
    document_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_url: Mapped[str | None] = mapped_column(Text)
    repository_path: Mapped[str | None] = mapped_column(Text)
    source_version: Mapped[str | None] = mapped_column(String(255))
    language: Mapped[str] = mapped_column(String(32), default="en", server_default="en")
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default="{}"
    )
    status: Mapped[str] = mapped_column(String(32), default="active", server_default="active")
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    source: Mapped[Source] = relationship(back_populates="documents")
    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentVersion(Base):
    """Immutable snapshot retained for auditability and incremental recovery."""

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "content_hash", name="uq_document_versions_document_hash"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    source_version: Mapped[str | None] = mapped_column(String(255))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
    parsed_content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="versions")


class Chunk(TimestampMixin, Base):
    """Atomic retrieval unit with denormalized fields needed for ranked search."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        Index("ix_chunks_metadata_gin", "metadata", postgresql_using="gin"),
        Index("ix_chunks_search_vector_gin", "search_vector", postgresql_using="gin"),
        Index("ix_chunks_document_active", "document_id", "deleted_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_chunk_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("chunks.id", ondelete="SET NULL")
    )
    document_title: Mapped[str] = mapped_column(Text, nullable=False)
    heading_path: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    contextualized_content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    start_offset: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_offset: Mapped[int] = mapped_column(BigInteger, nullable=False)
    start_line: Mapped[int | None] = mapped_column(Integer)
    end_line: Mapped[int | None] = mapped_column(Integer)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default="{}"
    )
    search_vector: Mapped[Any | None] = mapped_column(TSVECTOR)
    # Keep the ORM adapter dimension-agnostic so deployments can migrate the
    # physical vector column to another model dimension without editing Python.
    # The initial migration intentionally provisions BGE-M3's 1024 dimensions.
    embedding: Mapped[list[float] | None] = mapped_column(Vector())
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    embedding_dimension: Mapped[int | None] = mapped_column(Integer)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    document: Mapped[Document] = relationship(back_populates="chunks")
    parent: Mapped[Chunk | None] = relationship(remote_side="Chunk.id", back_populates="children")
    children: Mapped[list[Chunk]] = relationship(back_populates="parent")
