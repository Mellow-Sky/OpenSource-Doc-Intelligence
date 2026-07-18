"""Configuration validation tests."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_have_enterprise_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_dimension == 1024
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.keyword_top_k == settings.vector_top_k == 30
    assert settings.chunk_target_tokens < settings.chunk_max_tokens
    assert settings.chunk_tokenizer_provider == "auto"
    assert settings.chunk_tokenizer_model is None
    assert settings.chunk_tokenizer_allow_regex_fallback is False
    assert settings.ingestion_max_outstanding_jobs == 100
    assert settings.evaluation_max_outstanding_runs == 10
    assert settings.ingestion_max_concurrent_jobs == 1
    assert settings.evaluation_max_concurrent_runs == 1
    assert settings.queue_retry_after_seconds == 5


def test_chunk_overlap_must_be_smaller_than_target() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        Settings(_env_file=None, chunk_target_tokens=100, chunk_overlap_tokens=100)


def test_repository_requires_owner_and_name() -> None:
    with pytest.raises(ValidationError, match="owner/name"):
        Settings(_env_file=None, github_repository="kubernetes")


def test_embedding_dimension_must_match_migrated_vector_column() -> None:
    with pytest.raises(ValidationError, match="DATABASE_VECTOR_DIMENSION"):
        Settings(
            _env_file=None,
            embedding_dimension=1536,
            database_vector_dimension=1024,
        )

    settings = Settings(
        _env_file=None,
        embedding_dimension=1536,
        database_vector_dimension=1536,
    )
    assert settings.embedding_dimension == 1536


def test_model_provider_urls_cannot_embed_credentials() -> None:
    with pytest.raises(ValidationError, match="must not contain credentials"):
        Settings(
            _env_file=None,
            llm_base_url="https://user:secret@example.test/v1",
        )

    settings = Settings(_env_file=None, llm_base_url="https://example.test/v1/")
    assert settings.llm_base_url == "https://example.test/v1"


def test_git_commit_must_be_a_hexadecimal_revision() -> None:
    assert Settings(_env_file=None, git_commit="ABCDEF012345").git_commit == "abcdef012345"
    with pytest.raises(ValidationError, match="GIT_COMMIT"):
        Settings(_env_file=None, git_commit="latest")


def test_reranker_endpoint_healthcheck_requires_a_safe_relative_resource() -> None:
    with pytest.raises(ValidationError, match="RERANKER_HEALTHCHECK_RESOURCE"):
        Settings(_env_file=None, reranker_healthcheck_mode="endpoint")
    with pytest.raises(ValidationError, match="safe relative path"):
        Settings(
            _env_file=None,
            reranker_healthcheck_mode="endpoint",
            reranker_healthcheck_resource="https://attacker.example/steal-key",
        )

    settings = Settings(
        _env_file=None,
        reranker_healthcheck_mode="endpoint",
        reranker_healthcheck_resource="healthz",
    )
    assert settings.reranker_healthcheck_resource == "healthz"
