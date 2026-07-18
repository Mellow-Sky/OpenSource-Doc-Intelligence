"""Prometheus exposition endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["metrics"])


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Expose process, Python runtime, and normalized HTTP request metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
