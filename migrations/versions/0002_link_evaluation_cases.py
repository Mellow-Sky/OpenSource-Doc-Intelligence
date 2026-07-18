"""Persist stable evaluation cases and link historical result snapshots.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-18 00:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add complete case provenance, stable identity, and historical FK links."""
    op.add_column(
        "evaluation_cases",
        sa.Column("source_type", sa.String(64), nullable=True),
    )
    op.add_column(
        "evaluation_cases",
        sa.Column(
            "human_reviewed",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )

    # Preserve manually-created legacy rows while making every future identity addressable.
    op.execute(
        """
        UPDATE evaluation_cases
        SET external_id = 'legacy-' || id::text
        WHERE external_id IS NULL
        """
    )

    # Older workers embedded a validated EvaluationCase snapshot in result metrics. Backfill
    # the latest snapshot per (dataset, external id) before enforcing the unique identity.
    op.execute(
        """
        WITH snapshots AS (
            SELECT DISTINCT ON (runs.dataset_name, results.case_external_id)
                runs.dataset_name,
                results.case_external_id AS external_id,
                results.question AS result_question,
                results.predicted_answerable,
                results.metrics #> '{_evaluation,case}' AS case_data
            FROM evaluation_results AS results
            JOIN evaluation_runs AS runs ON runs.id = results.evaluation_run_id
            WHERE jsonb_typeof(results.metrics #> '{_evaluation,case}') = 'object'
            ORDER BY
                runs.dataset_name,
                results.case_external_id,
                runs.started_at DESC,
                results.id DESC
        )
        INSERT INTO evaluation_cases (
            id,
            external_id,
            dataset_name,
            question,
            conversation_history,
            reference_answer,
            relevant_chunk_ids,
            expected_citations,
            answerable,
            difficulty,
            category,
            source_type,
            human_reviewed,
            metadata,
            created_at,
            updated_at
        )
        SELECT
            gen_random_uuid(),
            snapshots.external_id,
            snapshots.dataset_name,
            COALESCE(NULLIF(snapshots.case_data ->> 'question', ''), snapshots.result_question),
            CASE
                WHEN jsonb_typeof(snapshots.case_data -> 'conversation_history') = 'array'
                THEN snapshots.case_data -> 'conversation_history'
                ELSE '[]'::jsonb
            END,
            COALESCE(snapshots.case_data ->> 'reference_answer', ''),
            CASE
                WHEN jsonb_typeof(snapshots.case_data -> 'relevant_chunk_ids') = 'array'
                THEN snapshots.case_data -> 'relevant_chunk_ids'
                ELSE '[]'::jsonb
            END,
            CASE
                WHEN jsonb_typeof(snapshots.case_data -> 'expected_citations') = 'array'
                THEN snapshots.case_data -> 'expected_citations'
                ELSE '[]'::jsonb
            END,
            CASE
                WHEN jsonb_typeof(snapshots.case_data -> 'answerable') = 'boolean'
                THEN (snapshots.case_data ->> 'answerable')::boolean
                ELSE snapshots.predicted_answerable
            END,
            COALESCE(NULLIF(snapshots.case_data ->> 'difficulty', ''), 'medium'),
            COALESCE(NULLIF(snapshots.case_data ->> 'category', ''), 'unknown'),
            NULLIF(snapshots.case_data ->> 'source_type', ''),
            CASE
                WHEN jsonb_typeof(snapshots.case_data -> 'human_reviewed') = 'boolean'
                THEN (snapshots.case_data ->> 'human_reviewed')::boolean
                ELSE false
            END,
            CASE
                WHEN jsonb_typeof(snapshots.case_data -> 'metadata') = 'object'
                THEN snapshots.case_data -> 'metadata'
                ELSE '{}'::jsonb
            END,
            now(),
            now()
        FROM snapshots
        WHERE NOT EXISTS (
            SELECT 1
            FROM evaluation_cases AS existing
            WHERE existing.dataset_name = snapshots.dataset_name
              AND existing.external_id = snapshots.external_id
        )
        """
    )

    op.alter_column(
        "evaluation_cases",
        "external_id",
        existing_type=sa.String(255),
        nullable=False,
    )
    op.create_unique_constraint(
        "uq_evaluation_cases_dataset_external_id",
        "evaluation_cases",
        ["dataset_name", "external_id"],
    )
    op.execute(
        """
        UPDATE evaluation_results AS results
        SET evaluation_case_id = cases.id
        FROM evaluation_runs AS runs, evaluation_cases AS cases
        WHERE runs.id = results.evaluation_run_id
          AND cases.dataset_name = runs.dataset_name
          AND cases.external_id = results.case_external_id
          AND results.evaluation_case_id IS NULL
        """
    )


def downgrade() -> None:
    """Remove the new identity constraint and provenance columns without deleting data."""
    op.drop_constraint(
        "uq_evaluation_cases_dataset_external_id",
        "evaluation_cases",
        type_="unique",
    )
    op.alter_column(
        "evaluation_cases",
        "external_id",
        existing_type=sa.String(255),
        nullable=True,
    )
    op.drop_column("evaluation_cases", "human_reviewed")
    op.drop_column("evaluation_cases", "source_type")
