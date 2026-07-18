from __future__ import annotations

from fastapi import FastAPI

from app.core.config import Settings
from app.core.telemetry import configure_telemetry


def test_telemetry_is_dependency_free_when_disabled() -> None:
    app = FastAPI()
    assert configure_telemetry(app, Settings(enable_telemetry=False)) is None
