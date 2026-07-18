from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_alembic_migration_graph_has_one_linear_head() -> None:
    """Keep deployments and E2E schema checks unambiguous as migrations grow."""

    root = Path(__file__).resolve().parents[2]
    script = ScriptDirectory.from_config(Config(str(root / "alembic.ini")))

    heads = script.get_heads()
    bases = script.get_bases()
    branch_points = [
        revision.revision for revision in script.walk_revisions() if revision.is_branch_point
    ]

    assert len(heads) == 1, f"expected one Alembic head, found {heads}"
    assert len(bases) == 1, f"expected one Alembic base, found {bases}"
    assert branch_points == []
