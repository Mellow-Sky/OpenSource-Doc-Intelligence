from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.evaluation_worker import _safe_error
from app.services.evaluation_execution_service import (
    _experiment_settings,
    _safe_dataset_path,
    _safe_snapshot,
    _snapshot_parts,
)


def test_evaluation_snapshot_and_overrides_are_allowlisted() -> None:
    request, overrides = _snapshot_parts(
        {
            "request": {
                "dataset_path": "evaluation/datasets/test.jsonl",
                "experiment_name": "keyword",
            },
            "overrides": {
                "retrieval_mode": "keyword",
                "enable_reranker": False,
                "experiment_top_k": 5,
            },
        }
    )
    settings, top_k = _experiment_settings(Settings(), overrides)

    assert request["experiment_name"] == "keyword"
    assert settings.retrieval_mode == "keyword"
    assert settings.enable_reranker is False
    assert top_k == 5

    with pytest.raises(ConfigurationError, match="not allowed"):
        _experiment_settings(Settings(), {"database_url": "postgresql://attacker"})


def test_evaluation_dataset_path_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "evaluation" / "datasets"
    root.mkdir(parents=True)
    dataset = root / "valid.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")

    assert _safe_dataset_path(str(dataset), root) == dataset.resolve()
    with pytest.raises(ConfigurationError, match="invalid"):
        _safe_dataset_path(str(tmp_path / "outside.jsonl"), root)


def test_evaluation_worker_error_redacts_configured_and_inline_credentials() -> None:
    error = RuntimeError(
        "Bearer exposed-token at https://user:password@example.test and api_key=known"
    )

    sanitized = _safe_error(error, ["known"])

    assert "exposed-token" not in sanitized
    assert "user:password" not in sanitized
    assert "known" not in sanitized


def test_execution_snapshot_never_persists_database_password() -> None:
    settings = Settings(database_url="postgresql+asyncpg://rag:database-secret@postgres:5432/rag")

    snapshot = _safe_snapshot(settings, 8)

    assert "database-secret" not in snapshot["database_url"]
    assert "***" in snapshot["database_url"]
