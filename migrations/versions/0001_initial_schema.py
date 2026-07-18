"""Create the complete initial PostgreSQL and pgvector schema.

Revision ID: 0001
Revises: None
Create Date: 2026-07-17 00:00:00+00:00
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())


def _timestamps() -> list[sa.Column[object]]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    """Create extensions, application tables, constraints, and search indexes."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "sources",
        sa.Column("id", UUID, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("source_type", sa.String(64), nullable=False),
        sa.Column("base_url", sa.Text()),
        sa.Column("repository", sa.String(512)),
        sa.Column("branch", sa.String(255)),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("config", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_sources"),
        sa.UniqueConstraint("name", name="uq_sources_name"),
    )
    op.create_index("ix_sources_source_type", "sources", ["source_type"])

    op.create_table(
        "documents",
        sa.Column("id", UUID, nullable=False),
        sa.Column("source_id", UUID, nullable=False),
        sa.Column("external_id", sa.String(1024), nullable=False),
        sa.Column("document_type", sa.String(64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("canonical_url", sa.Text()),
        sa.Column("repository_path", sa.Text()),
        sa.Column("source_version", sa.String(255)),
        sa.Column("language", sa.String(32), server_default="en", nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("metadata", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("status", sa.String(32), server_default="active", nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("indexed_at", sa.DateTime(timezone=True)),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE", name="fk_documents_source_id_sources"),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.UniqueConstraint("source_id", "external_id", name="uq_documents_source_external"),
    )
    op.create_index("ix_documents_document_type", "documents", ["document_type"])
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])
    op.create_index("ix_documents_active", "documents", ["source_id", "status", "deleted_at"])
    op.create_index("ix_documents_metadata_gin", "documents", ["metadata"], postgresql_using="gin")

    op.create_table(
        "document_versions",
        sa.Column("id", UUID, nullable=False),
        sa.Column("document_id", UUID, nullable=False),
        sa.Column("source_version", sa.String(255)),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("raw_content", sa.Text(), nullable=False),
        sa.Column("parsed_content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE", name="fk_document_versions_document_id_documents"),
        sa.PrimaryKeyConstraint("id", name="pk_document_versions"),
        sa.UniqueConstraint("document_id", "content_hash", name="uq_document_versions_document_hash"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", UUID, nullable=False),
        sa.Column("document_id", UUID, nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("parent_chunk_id", UUID),
        sa.Column("document_title", sa.Text(), nullable=False),
        sa.Column("heading_path", JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("contextualized_content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("start_offset", sa.BigInteger(), nullable=False),
        sa.Column("end_offset", sa.BigInteger(), nullable=False),
        sa.Column("start_line", sa.Integer()),
        sa.Column("end_line", sa.Integer()),
        sa.Column("metadata", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("search_vector", postgresql.TSVECTOR()),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=1024)),
        sa.Column("embedding_model", sa.String(255)),
        sa.Column("embedding_dimension", sa.Integer()),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE", name="fk_chunks_document_id_documents"),
        sa.ForeignKeyConstraint(["parent_chunk_id"], ["chunks.id"], ondelete="SET NULL", name="fk_chunks_parent_chunk_id_chunks"),
        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        sa.CheckConstraint("start_offset >= 0 AND end_offset >= start_offset", name="ck_chunks_valid_offsets"),
        sa.CheckConstraint("token_count > 0", name="ck_chunks_positive_tokens"),
    )
    op.create_index("ix_chunks_content_hash", "chunks", ["content_hash"])
    op.create_index("ix_chunks_document_active", "chunks", ["document_id", "deleted_at"])
    op.create_index("ix_chunks_metadata_gin", "chunks", ["metadata"], postgresql_using="gin")
    op.create_index("ix_chunks_search_vector_gin", "chunks", ["search_vector"], postgresql_using="gin")
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw "
        "(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64) "
        "WHERE embedding IS NOT NULL"
    )
    op.execute("CREATE INDEX ix_chunks_content_trgm ON chunks USING gin (content gin_trgm_ops)")
    op.execute(
        """
        CREATE FUNCTION chunks_search_vector_update() RETURNS trigger AS $$
        BEGIN
          NEW.search_vector :=
            setweight(to_tsvector('english', coalesce(NEW.document_title, '')), 'A') ||
            setweight(to_tsvector('english', coalesce(NEW.heading_path::text, '')), 'B') ||
            setweight(to_tsvector('english', coalesce(NEW.content, '')), 'C') ||
            setweight(to_tsvector('simple', coalesce(NEW.contextualized_content, '')), 'D');
          RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    # Keep DDL statements separate: asyncpg rejects multiple commands in a
    # prepared statement even though PostgreSQL itself accepts the SQL text.
    op.execute(
        """
        CREATE TRIGGER chunks_search_vector_trigger
        BEFORE INSERT OR UPDATE OF document_title, heading_path, content, contextualized_content
        ON chunks FOR EACH ROW EXECUTE FUNCTION chunks_search_vector_update()
        """
    )

    op.create_table(
        "conversations",
        sa.Column("id", UUID, nullable=False),
        sa.Column("user_id", sa.String(255)),
        sa.Column("title", sa.String(512)),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_conversations"),
    )
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])

    op.create_table(
        "messages",
        sa.Column("id", UUID, nullable=False),
        sa.Column("conversation_id", UUID, nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("original_query", sa.Text()),
        sa.Column("rewritten_query", sa.Text()),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_usage", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("cost", sa.Numeric(18, 8)),
        sa.Column("latency_ms", sa.Integer()),
        *_timestamps(),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE", name="fk_messages_conversation_id_conversations"),
        sa.PrimaryKeyConstraint("id", name="pk_messages"),
    )
    op.create_index("ix_messages_conversation_created", "messages", ["conversation_id", "created_at"])

    op.create_table(
        "retrieval_runs",
        sa.Column("id", UUID, nullable=False),
        sa.Column("message_id", UUID, nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("rewritten_query", sa.Text(), nullable=False),
        sa.Column("filters", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("keyword_latency_ms", sa.Float(), server_default="0", nullable=False),
        sa.Column("vector_latency_ms", sa.Float(), server_default="0", nullable=False),
        sa.Column("rerank_latency_ms", sa.Float(), server_default="0", nullable=False),
        sa.Column("total_latency_ms", sa.Float(), server_default="0", nullable=False),
        sa.Column("retrieved_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("reranked_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("no_answer_score", sa.Float()),
        sa.Column("metadata", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE", name="fk_retrieval_runs_message_id_messages"),
        sa.PrimaryKeyConstraint("id", name="pk_retrieval_runs"),
    )
    op.create_index("ix_retrieval_runs_message", "retrieval_runs", ["message_id"])

    op.create_table(
        "retrieval_results",
        sa.Column("id", UUID, nullable=False),
        sa.Column("retrieval_run_id", UUID, nullable=False),
        sa.Column("chunk_id", UUID, nullable=False),
        sa.Column("keyword_rank", sa.Integer()),
        sa.Column("vector_rank", sa.Integer()),
        sa.Column("fused_rank", sa.Integer()),
        sa.Column("rerank_rank", sa.Integer()),
        sa.Column("keyword_score", sa.Float()),
        sa.Column("vector_score", sa.Float()),
        sa.Column("fused_score", sa.Float()),
        sa.Column("rerank_score", sa.Float()),
        sa.Column("selected_for_context", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.ForeignKeyConstraint(["retrieval_run_id"], ["retrieval_runs.id"], ondelete="CASCADE", name="fk_retrieval_results_retrieval_run_id_retrieval_runs"),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE", name="fk_retrieval_results_chunk_id_chunks"),
        sa.PrimaryKeyConstraint("id", name="pk_retrieval_results"),
        sa.UniqueConstraint("retrieval_run_id", "chunk_id", name="uq_retrieval_results_run_chunk"),
    )
    op.create_index("ix_retrieval_results_run_rank", "retrieval_results", ["retrieval_run_id", "rerank_rank"])

    op.create_table(
        "answer_citations",
        sa.Column("id", UUID, nullable=False),
        sa.Column("message_id", UUID, nullable=False),
        sa.Column("chunk_id", UUID, nullable=False),
        sa.Column("citation_number", sa.Integer(), nullable=False),
        sa.Column("quoted_text", sa.Text(), nullable=False),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("citation_valid", sa.Boolean()),
        sa.Column("validation_score", sa.Float()),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE", name="fk_answer_citations_message_id_messages"),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE", name="fk_answer_citations_chunk_id_chunks"),
        sa.PrimaryKeyConstraint("id", name="pk_answer_citations"),
        sa.UniqueConstraint(
            "message_id",
            "citation_number",
            "chunk_id",
            name="uq_answer_citations_message_number_chunk",
        ),
    )

    op.create_table(
        "usage_records",
        sa.Column("id", UUID, nullable=False),
        sa.Column("request_id", UUID, nullable=False),
        sa.Column("operation", sa.String(64), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("completion_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("total_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("estimated_cost", sa.Numeric(18, 8)),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_usage_records"),
    )
    op.create_index("ix_usage_records_request_id", "usage_records", ["request_id"])
    op.create_index("ix_usage_records_request_operation", "usage_records", ["request_id", "operation"])

    op.create_table(
        "evaluation_cases",
        sa.Column("id", UUID, nullable=False),
        sa.Column("external_id", sa.String(255)),
        sa.Column("dataset_name", sa.String(255), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("conversation_history", JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("reference_answer", sa.Text(), nullable=False),
        sa.Column("relevant_chunk_ids", JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("expected_citations", JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("answerable", sa.Boolean(), nullable=False),
        sa.Column("difficulty", sa.String(32), nullable=False),
        sa.Column("category", sa.String(64), nullable=False),
        sa.Column("metadata", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_evaluation_cases"),
    )
    op.create_index("ix_evaluation_cases_dataset_name", "evaluation_cases", ["dataset_name"])
    op.create_index("ix_evaluation_cases_dataset_category", "evaluation_cases", ["dataset_name", "category"])

    op.create_table(
        "evaluation_runs",
        sa.Column("id", UUID, nullable=False),
        sa.Column("dataset_name", sa.String(255), nullable=False),
        sa.Column("config_snapshot", JSONB, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("summary", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("report_path", sa.Text()),
        sa.Column("status", sa.String(32), server_default="pending", nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_evaluation_runs"),
    )
    op.create_index(
        "ix_evaluation_runs_status_started",
        "evaluation_runs",
        ["status", "started_at"],
    )

    op.create_table(
        "evaluation_results",
        sa.Column("id", UUID, nullable=False),
        sa.Column("evaluation_run_id", UUID, nullable=False),
        sa.Column("evaluation_case_id", UUID),
        sa.Column("case_external_id", sa.String(255), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("generated_answer", sa.Text(), nullable=False),
        sa.Column("predicted_answerable", sa.Boolean(), nullable=False),
        sa.Column("retrieved_results", JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("citations", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("metrics", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("latency", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("usage", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("error", sa.Text()),
        sa.ForeignKeyConstraint(["evaluation_run_id"], ["evaluation_runs.id"], ondelete="CASCADE", name="fk_evaluation_results_evaluation_run_id_evaluation_runs"),
        sa.ForeignKeyConstraint(["evaluation_case_id"], ["evaluation_cases.id"], ondelete="SET NULL", name="fk_evaluation_results_evaluation_case_id_evaluation_cases"),
        sa.PrimaryKeyConstraint("id", name="pk_evaluation_results"),
    )
    op.create_index("ix_evaluation_results_run_case", "evaluation_results", ["evaluation_run_id", "evaluation_case_id"])

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", UUID, nullable=False),
        sa.Column("source_id", UUID, nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), server_default="pending", nullable=False),
        sa.Column("requested_by", sa.String(255)),
        sa.Column("options", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("stats", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE", name="fk_ingestion_jobs_source_id_sources"),
        sa.PrimaryKeyConstraint("id", name="pk_ingestion_jobs"),
        sa.UniqueConstraint("idempotency_key", name="uq_ingestion_jobs_idempotency_key"),
    )
    op.create_index("ix_ingestion_jobs_status_created", "ingestion_jobs", ["status", "created_at"])

    op.create_table(
        "sync_cursors",
        sa.Column("id", UUID, nullable=False),
        sa.Column("source_id", UUID, nullable=False),
        sa.Column("cursor_type", sa.String(64), nullable=False),
        sa.Column("cursor_value", sa.Text(), nullable=False),
        sa.Column("metadata", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE", name="fk_sync_cursors_source_id_sources"),
        sa.PrimaryKeyConstraint("id", name="pk_sync_cursors"),
        sa.UniqueConstraint("source_id", "cursor_type", name="uq_sync_cursors_source_type"),
    )


def downgrade() -> None:
    """Drop the initial application schema while retaining shared extensions."""
    op.drop_table("sync_cursors")
    op.drop_table("ingestion_jobs")
    op.drop_index("ix_evaluation_results_run_case", table_name="evaluation_results")
    op.drop_table("evaluation_results")
    op.drop_table("evaluation_runs")
    op.drop_index("ix_evaluation_cases_dataset_category", table_name="evaluation_cases")
    op.drop_index("ix_evaluation_cases_dataset_name", table_name="evaluation_cases")
    op.drop_table("evaluation_cases")
    op.drop_index("ix_usage_records_request_operation", table_name="usage_records")
    op.drop_index("ix_usage_records_request_id", table_name="usage_records")
    op.drop_table("usage_records")
    op.drop_table("answer_citations")
    op.drop_index("ix_retrieval_results_run_rank", table_name="retrieval_results")
    op.drop_table("retrieval_results")
    op.drop_index("ix_retrieval_runs_message", table_name="retrieval_runs")
    op.drop_table("retrieval_runs")
    op.drop_index("ix_messages_conversation_created", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_table("conversations")
    op.execute("DROP TRIGGER IF EXISTS chunks_search_vector_trigger ON chunks")
    op.execute("DROP FUNCTION IF EXISTS chunks_search_vector_update()")
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_trgm")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.drop_index("ix_chunks_search_vector_gin", table_name="chunks")
    op.drop_index("ix_chunks_metadata_gin", table_name="chunks")
    op.drop_index("ix_chunks_document_active", table_name="chunks")
    op.drop_index("ix_chunks_content_hash", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("document_versions")
    op.drop_index("ix_documents_metadata_gin", table_name="documents")
    op.drop_index("ix_documents_active", table_name="documents")
    op.drop_index("ix_documents_content_hash", table_name="documents")
    op.drop_index("ix_documents_document_type", table_name="documents")
    op.drop_table("documents")
    op.drop_index("ix_sources_source_type", table_name="sources")
    op.drop_table("sources")
