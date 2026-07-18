"""Prometheus exposition and normalized-label instrumentation tests."""

import httpx
import pytest
from fastapi import FastAPI, Response

from app.api.routes.metrics import router
from app.core.metrics import PrometheusMiddleware


@pytest.mark.asyncio
async def test_metrics_exposes_normalized_http_route_and_runtime_metrics() -> None:
    app = FastAPI()
    app.add_middleware(PrometheusMiddleware)
    app.include_router(router)

    @app.get("/items/{item_id}", status_code=204)
    async def item(item_id: str) -> Response:
        return Response(status_code=204)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/items/secret-one")
        await client.get("/items/secret-two")
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "python_info" in body
    assert "odi_http_requests_total" in body
    assert 'route="/items/{item_id}"' in body
    assert "secret-one" not in body
    assert "secret-two" not in body
    assert "odi_http_request_duration_seconds_bucket" in body
    assert "odi_http_response_size_bytes_bucket" in body
