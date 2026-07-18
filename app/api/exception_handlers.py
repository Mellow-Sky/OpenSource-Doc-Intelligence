"""Stable JSON error responses and request correlation middleware."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.exceptions import AppError, RateLimitError

logger = structlog.get_logger(__name__)


def request_id_from(request: Request) -> str:
    """Read the middleware-generated request ID without trusting client values blindly."""
    return str(getattr(request.state, "request_id", "unknown"))


def error_payload(
    *, code: str, message: str, request_id: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "details": details or {},
        }
    }


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a UUID request ID to response headers, logs, and controlled errors."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if 0 < len(supplied) <= 128 else str(uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        # A route may normalize the correlation identifier further (chat uses a
        # UUID because the same value is persisted in usage_records).
        response.headers["X-Request-ID"] = request_id_from(request)
        return response


async def app_error_handler(request: Request, exc: AppError) -> ORJSONResponse:
    logger.warning("controlled_request_error", code=exc.code, status_code=exc.status_code)
    headers: dict[str, str] = {}
    if isinstance(exc, RateLimitError) and exc.retry_after_seconds is not None:
        headers["Retry-After"] = str(exc.retry_after_seconds)
    return ORJSONResponse(
        status_code=exc.status_code,
        headers=headers,
        content=error_payload(
            code=exc.code,
            message=exc.message,
            request_id=request_id_from(request),
            details=exc.details,
        ),
    )


async def request_validation_handler(
    request: Request, exc: RequestValidationError
) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=422,
        content=error_payload(
            code="VALIDATION_FAILED",
            message="Request validation failed",
            request_id=request_id_from(request),
            # Pydantic validator errors can carry a ValueError object in ``ctx``.
            # Normalize it before ORJSON serialization while keeping the public detail.
            details={"errors": jsonable_encoder(exc.errors())},
        ),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> ORJSONResponse:
    logger.exception("unhandled_request_error", exception_type=type(exc).__name__)
    return ORJSONResponse(
        status_code=500,
        content=error_payload(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred",
            request_id=request_id_from(request),
        ),
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Register the controlled error contract without exposing tracebacks."""
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, request_validation_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_error_handler)
