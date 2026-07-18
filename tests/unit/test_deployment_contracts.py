from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def test_compose_declares_migration_api_and_both_durable_workers() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert services["postgres"]["image"] == "pgvector/pgvector:pg16"
    assert services["migrate"]["command"] == ["alembic", "upgrade", "head"]
    assert services["api"]["depends_on"]["migrate"]["condition"] == (
        "service_completed_successfully"
    )
    assert services["worker"]["command"] == ["python", "-m", "app.worker"]
    assert services["evaluation-worker"]["command"] == [
        "python",
        "-m",
        "app.evaluation_worker",
    ]
    assert set(compose["volumes"]) >= {
        "postgres-data",
        "ingestion-cache",
        "model-cache",
        "evaluation-reports",
    }


def test_compose_database_url_is_overridable_and_has_matching_defaults() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))

    assert compose["x-app-environment"]["DATABASE_URL"] == (
        "${DATABASE_URL:-postgresql+asyncpg://rag:rag@postgres:5432/rag}"
    )
    assert compose["services"]["postgres"]["environment"]["POSTGRES_PASSWORD"] == (
        "${POSTGRES_PASSWORD:-rag}"
    )


def test_dockerfile_is_locked_non_root_and_contains_runtime_assets() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.count("FROM ") >= 2
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "COPY --chown=odi:odi evaluation/datasets ./evaluation/datasets" in dockerfile
    assert "USER odi" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "org.opencontainers.image.revision=${GIT_COMMIT}" in dockerfile
    assert "COPY .env" not in dockerfile
    assert ".env" in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()


def test_makefile_exposes_every_documented_acceptance_target() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-z][a-z-]*):", makefile, flags=re.MULTILINE))

    assert targets >= {
        "install",
        "dev",
        "test",
        "test-unit",
        "test-integration",
        "lint",
        "format",
        "typecheck",
        "migrate",
        "migration",
        "ingest",
        "evaluate",
        "benchmark",
        "docker-up",
        "docker-down",
        "clean",
    }
    assert "git rev-parse --verify HEAD" in makefile


def test_ci_runs_quality_migration_tests_and_container_build() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    quality_commands = {step["run"] for step in jobs["quality-and-tests"]["steps"] if "run" in step}
    container_commands = {step["run"] for step in jobs["container-build"]["steps"] if "run" in step}

    assert "uv run ruff check ." in quality_commands
    assert "uv run mypy app" in quality_commands
    assert "uv run alembic upgrade head" in quality_commands
    assert "uv run pytest" in quality_commands
    assert "docker compose config --quiet" in container_commands
    assert "docker compose build" in container_commands

    build_step = next(
        step
        for step in jobs["container-build"]["steps"]
        if step.get("run") == "docker compose build"
    )
    assert build_step["env"] == {
        "GIT_COMMIT": "${{ github.sha }}",
        "INSTALL_LOCAL_MODELS": "false",
    }
    assert any(
        "import app.main, evaluation" in command and "alembic heads" in command
        for command in container_commands
    )
