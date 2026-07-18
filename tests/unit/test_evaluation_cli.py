from __future__ import annotations

from argparse import Namespace

import pytest

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from scripts.run_evaluation import (
    Experiment,
    _apply_overrides,
    _experiments,
    _safe_snapshot,
    parse_args,
)


def test_compare_mode_contains_retrieval_rewrite_and_top_k_baselines() -> None:
    args = Namespace(experiment_config=None, compare=True, experiment_name="unused")

    experiments = _experiments(args, Settings())

    assert [experiment.name for experiment in experiments] == [
        "keyword-only",
        "vector-only",
        "hybrid",
        "hybrid-reranked",
        "raw-query",
        "rewritten-query",
        "top-k-5",
        "top-k-10",
        "top-k-20",
    ]


def test_experiment_overrides_are_validated_and_unknown_keys_rejected() -> None:
    updated = _apply_overrides(
        Settings(),
        {"retrieval_mode": "keyword", "enable_reranker": False},
    )
    assert updated.retrieval_mode == "keyword"
    assert updated.enable_reranker is False

    with pytest.raises(ConfigurationError, match="Unknown experiment settings"):
        _apply_overrides(Settings(), {"typo_setting": True})


def test_config_snapshot_redacts_secrets_and_database_password() -> None:
    settings = Settings(
        github_token="sensitive",
        llm_api_key="sensitive",
        database_url="postgresql+asyncpg://rag:database-secret@postgres:5432/rag",
        chunk_target_tokens=500,
    )

    snapshot = _safe_snapshot(settings, Experiment(name="test"))

    assert snapshot["github_token"] == "[REDACTED]"
    assert snapshot["llm_api_key"] == "[REDACTED]"
    assert "database-secret" not in snapshot["database_url"]
    assert "***" in snapshot["database_url"]
    assert snapshot["chunk_target_tokens"] == 500


def test_cli_help_and_required_defaults_are_real_paths() -> None:
    args = parse_args([])
    assert str(args.dataset).endswith("evaluation/datasets/kubernetes_eval.jsonl")
    assert str(args.output).endswith("evaluation/reports/latest")
