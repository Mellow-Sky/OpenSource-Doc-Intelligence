"""Typed environment configuration for the application."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
        env_ignore_empty=True,
    )

    app_env: Literal["development", "test", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    git_commit: str | None = None
    api_key: SecretStr | None = None
    admin_api_key: SecretStr | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    database_pool_size: int = Field(default=10, ge=1, le=100)
    database_max_overflow: int = Field(default=20, ge=0, le=200)

    github_token: SecretStr | None = None
    github_repository: str = "kubernetes/kubernetes"
    github_branch: str = "master"

    llm_provider: str = "openai_compatible"
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    llm_api_version: str | None = None
    llm_deployment: str | None = None
    llm_tokenizer_provider: Literal["auto", "huggingface", "regex"] = "auto"
    llm_tokenizer_model: str | None = None
    llm_tokenizer_allow_regex_fallback: bool = False
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    llm_max_concurrency: int = Field(default=8, ge=1, le=64)
    llm_healthcheck_mode: Literal["auto", "catalog", "inference"] = "auto"
    answer_max_tokens: int = Field(default=1200, ge=64, le=8192)
    query_rewrite_max_tokens: int = Field(default=256, ge=32, le=2048)
    query_rewrite_history_turns: int = Field(default=4, ge=1, le=20)
    judge_max_tokens: int = Field(default=256, ge=32, le=2048)
    judge_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    judge_provider: str | None = None
    judge_base_url: str | None = None
    judge_api_key: SecretStr | None = None
    judge_model: str | None = None
    judge_api_version: str | None = None
    judge_deployment: str | None = None

    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-m3"
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_api_version: str | None = None
    embedding_deployment: str | None = None
    embedding_dimension: int = Field(default=1024, ge=1, le=65535)
    database_vector_dimension: int = Field(default=1024, ge=1, le=65535)
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    embedding_max_concurrency: int = Field(default=2, ge=1, le=32)
    embedding_healthcheck_mode: Literal["auto", "catalog", "inference"] = "auto"

    reranker_provider: str = "local"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_base_url: str | None = None
    reranker_api_key: SecretStr | None = None
    reranker_batch_size: int = Field(default=16, ge=1, le=256)
    reranker_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    reranker_max_concurrency: int = Field(default=2, ge=1, le=32)
    reranker_healthcheck_mode: Literal["auto", "inference", "endpoint"] = "auto"
    reranker_healthcheck_resource: str | None = None
    reranker_score_threshold: float | None = None
    provider_max_retries: int = Field(default=3, ge=0, le=10)

    keyword_top_k: int = Field(default=30, ge=1, le=200)
    vector_top_k: int = Field(default=30, ge=1, le=200)
    rerank_top_k: int = Field(default=8, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    retrieval_mode: Literal["hybrid", "keyword", "vector"] = "hybrid"
    fusion_mode: Literal["rrf", "weighted"] = "rrf"
    keyword_fusion_weight: float = Field(default=0.5, ge=0)
    vector_fusion_weight: float = Field(default=0.5, ge=0)

    chunk_target_tokens: int = Field(default=500, ge=32)
    chunk_max_tokens: int = Field(default=800, ge=64)
    chunk_overlap_tokens: int = Field(default=80, ge=0)
    chunk_min_tokens: int = Field(default=80, ge=1)
    chunk_tokenizer_provider: Literal["auto", "huggingface", "regex"] = "auto"
    chunk_tokenizer_model: str | None = None
    chunk_tokenizer_allow_regex_fallback: bool = False

    no_answer_top1_threshold: float | None = Field(default=None, ge=-1, le=1)
    no_answer_avg_threshold: float | None = Field(default=None, ge=-1, le=1)
    no_answer_margin_threshold: float | None = Field(default=None, ge=0, le=2)
    no_answer_top_k: int = Field(default=3, ge=1, le=20)
    no_answer_topic_overlap_threshold: float = Field(default=0.05, ge=0, le=1)
    no_answer_gray_zone_lower: float = Field(default=0.35, ge=0, le=1)
    no_answer_gray_zone_upper: float = Field(default=0.65, ge=0, le=1)
    evidence_sufficiency_threshold: float = Field(default=0.60, ge=0, le=1)
    citation_coverage_threshold: float = Field(default=0.60, ge=0, le=1)
    max_context_tokens: int = Field(default=6000, ge=256)

    enable_query_rewrite: bool = True
    enable_reranker: bool = True
    enable_citation_validation: bool = True
    enable_telemetry: bool = False
    otel_service_name: str = "opensource-doc-intelligence"
    otel_exporter_otlp_endpoint: str | None = None

    pricing_config_path: Path = Path("config/pricing.yaml")
    prompt_directory: Path = Path("prompts")
    ingestion_worker_poll_seconds: float = Field(default=2.0, ge=0.1, le=60)
    evaluation_worker_poll_seconds: float = Field(default=2.0, ge=0.1, le=60)
    ingestion_max_outstanding_jobs: int = Field(default=100, ge=1, le=100_000)
    evaluation_max_outstanding_runs: int = Field(default=10, ge=1, le=10_000)
    ingestion_max_concurrent_jobs: int = Field(default=1, ge=1, le=128)
    evaluation_max_concurrent_runs: int = Field(default=1, ge=1, le=128)
    queue_retry_after_seconds: int = Field(default=5, ge=1, le=3600)
    evaluation_concurrency: int = Field(default=1, ge=1, le=16)
    evaluation_heartbeat_seconds: float = Field(default=30.0, ge=1, le=3600)
    evaluation_stale_seconds: int = Field(default=86_400, ge=300, le=604_800)
    max_query_length: int = Field(default=4000, ge=1, le=50000)

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Normalize and validate the configured log level."""
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            msg = f"Unsupported LOG_LEVEL: {value}"
            raise ValueError(msg)
        return normalized

    @field_validator("reranker_score_threshold")
    @classmethod
    def validate_reranker_score_threshold(cls, value: float | None) -> float | None:
        """Reject NaN and infinity while allowing provider-specific score ranges."""
        if value is not None and not math.isfinite(value):
            msg = "RERANKER_SCORE_THRESHOLD must be finite"
            raise ValueError(msg)
        return value

    @field_validator("git_commit")
    @classmethod
    def validate_git_commit(cls, value: str | None) -> str | None:
        """Accept an immutable hexadecimal VCS revision injected at build time."""

        if value is None:
            return None
        normalized = value.strip().casefold()
        if not re.fullmatch(r"[0-9a-f]{7,64}", normalized):
            raise ValueError("GIT_COMMIT must be a 7 to 64 character hexadecimal revision")
        return normalized

    @field_validator("github_repository")
    @classmethod
    def validate_repository(cls, value: str) -> str:
        """Require GitHub repositories in owner/name form."""
        if value.count("/") != 1 or any(not part.strip() for part in value.split("/")):
            msg = "GITHUB_REPOSITORY must use owner/name format"
            raise ValueError(msg)
        return value

    @field_validator(
        "llm_base_url",
        "judge_base_url",
        "embedding_base_url",
        "reranker_base_url",
    )
    @classmethod
    def validate_model_endpoint(cls, value: str | None) -> str | None:
        """Require HTTP(S) model endpoints and prohibit credentials in URLs."""

        if value is None:
            return None
        normalized = value.strip()
        parts = urlsplit(normalized)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("model provider base URLs must be absolute HTTP(S) URLs")
        if parts.username is not None or parts.password is not None:
            raise ValueError("model provider base URLs must not contain credentials")
        return normalized.rstrip("/")

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> Settings:
        """Validate relationships between chunking settings."""
        if self.chunk_target_tokens > self.chunk_max_tokens:
            msg = "CHUNK_TARGET_TOKENS cannot exceed CHUNK_MAX_TOKENS"
            raise ValueError(msg)
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            msg = "CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TARGET_TOKENS"
            raise ValueError(msg)
        if self.chunk_min_tokens > self.chunk_target_tokens:
            msg = "CHUNK_MIN_TOKENS cannot exceed CHUNK_TARGET_TOKENS"
            raise ValueError(msg)
        if self.rerank_top_k > max(self.keyword_top_k, self.vector_top_k):
            msg = "RERANK_TOP_K cannot exceed both retrieval candidate limits"
            raise ValueError(msg)
        if self.embedding_dimension != self.database_vector_dimension:
            msg = (
                "EMBEDDING_DIMENSION must match DATABASE_VECTOR_DIMENSION; "
                "migrate the pgvector column before changing both values"
            )
            raise ValueError(msg)
        if self.keyword_fusion_weight + self.vector_fusion_weight <= 0:
            msg = "At least one retrieval fusion weight must be positive"
            raise ValueError(msg)
        if self.no_answer_gray_zone_lower >= self.no_answer_gray_zone_upper:
            msg = "NO_ANSWER_GRAY_ZONE_LOWER must be smaller than the upper bound"
            raise ValueError(msg)
        if self.evaluation_heartbeat_seconds >= self.evaluation_stale_seconds:
            msg = "EVALUATION_HEARTBEAT_SECONDS must be smaller than EVALUATION_STALE_SECONDS"
            raise ValueError(msg)
        if self.reranker_healthcheck_mode == "endpoint":
            resource = (self.reranker_healthcheck_resource or "").strip("/")
            if not resource:
                msg = (
                    "RERANKER_HEALTHCHECK_RESOURCE is required when "
                    "RERANKER_HEALTHCHECK_MODE=endpoint"
                )
                raise ValueError(msg)
            if "://" in resource or resource.startswith("..") or "?" in resource:
                msg = "RERANKER_HEALTHCHECK_RESOURCE must be a safe relative path"
                raise ValueError(msg)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""
    return Settings()
