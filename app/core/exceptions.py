"""Application exception hierarchy exposed through a stable API error contract."""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for controlled application failures."""

    code = "INTERNAL_ERROR"
    status_code = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(AppError):
    code = "VALIDATION_FAILED"
    status_code = 422


class AuthenticationError(AppError):
    code = "AUTHENTICATION_FAILED"
    status_code = 401


class ProviderError(AppError):
    code = "PROVIDER_FAILED"
    status_code = 502


class RetrievalError(AppError):
    code = "RETRIEVAL_FAILED"
    status_code = 500


class IngestionError(AppError):
    code = "INGESTION_FAILED"
    status_code = 500


class DatabaseError(AppError):
    code = "DATABASE_FAILED"
    status_code = 503


class RateLimitError(AppError):
    code = "RATE_LIMITED"
    status_code = 429

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.retry_after_seconds = retry_after_seconds


class ConfigurationError(AppError):
    code = "CONFIGURATION_INVALID"
    status_code = 500


class NotFoundError(AppError):
    code = "NOT_FOUND"
    status_code = 404
