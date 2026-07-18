"""Liveness API behavior independent of downstream readiness."""

import httpx
import pytest

from app.core.config import Settings
from app.main import create_app


@pytest.mark.asyncio
async def test_health_endpoint_returns_version_and_request_id() -> None:
    settings = Settings(_env_file=None, app_env="test")
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["x-request-id"]
