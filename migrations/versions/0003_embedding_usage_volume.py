"""Persist embedding input volume alongside token and latency usage.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-18 00:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add non-negative input cardinality and character-volume counters."""

    op.add_column(
        "usage_records",
        sa.Column("input_text_count", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column(
        "usage_records",
        sa.Column("input_character_count", sa.Integer(), server_default="0", nullable=False),
    )
    op.create_check_constraint(
        op.f("ck_usage_records_input_text_count_nonnegative"),
        "usage_records",
        "input_text_count >= 0",
    )
    op.create_check_constraint(
        op.f("ck_usage_records_input_character_count_nonnegative"),
        "usage_records",
        "input_character_count >= 0",
    )


def downgrade() -> None:
    """Remove embedding input-volume counters."""

    op.drop_constraint(
        op.f("ck_usage_records_input_character_count_nonnegative"),
        "usage_records",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_usage_records_input_text_count_nonnegative"),
        "usage_records",
        type_="check",
    )
    op.drop_column("usage_records", "input_character_count")
    op.drop_column("usage_records", "input_text_count")
