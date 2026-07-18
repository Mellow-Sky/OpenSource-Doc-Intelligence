"""Generic source configuration remains replaceable without code changes."""

from pathlib import Path

from app.schemas.ingestion import load_source_configs


def test_default_kubernetes_sources_are_valid() -> None:
    path = Path(__file__).parents[2] / "config" / "sources.yaml"
    sources = load_source_configs(path)

    assert len(sources) == 5
    assert {source.source_type for source in sources} >= {
        "github_repository",
        "github_issues",
        "release_notes",
        "kubernetes_api_reference",
    }
    assert all(source.enabled for source in sources)
