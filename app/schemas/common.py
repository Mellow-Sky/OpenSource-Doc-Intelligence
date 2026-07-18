"""Schemas shared by multiple API routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadinessCheck(BaseModel):
    ready: bool
    latency_ms: float = Field(ge=0)
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, ReadinessCheck]
