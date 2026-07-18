"""HTTP contract and report-path security tests for offline evaluation jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from app.api.exception_handlers import RequestIDMiddleware, install_exception_handlers
from app.api.routes.evaluation import (
    get_evaluation_report_root,
    get_evaluation_service,
    router,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import NotFoundError, RateLimitError
from app.db.models.evaluation import EvaluationRun
from app.schemas.evaluation import EvaluationRunCreateRequest
from app.services.evaluation_service import EvaluationRunView, EvaluationService


@pytest.fixture
def inline_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the tiny fixture read inline in restricted CI without changing production code."""

    async def run_inline(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)


class _FakeEvaluationService:
    def __init__(self, run: EvaluationRun) -> None:
        self.run = run
        self.enqueue_calls: list[EvaluationRunCreateRequest] = []
        self.missing = False
        self.rate_limited = False

    async def enqueue(self, request: EvaluationRunCreateRequest) -> EvaluationRunView:
        if self.rate_limited:
            raise RateLimitError(
                "Evaluation queue capacity exceeded",
                details={"queue": "evaluation", "limit": 1},
                retry_after_seconds=11,
            )
        self.enqueue_calls.append(request)
        return EvaluationRunView(run=self.run, result_count=0)

    async def get(self, run_id: UUID) -> EvaluationRunView:
        if self.missing or run_id != self.run.id:
            raise NotFoundError("Evaluation run was not found")
        return EvaluationRunView(run=self.run, result_count=3)


def _run(*, report_path: str | None = None, status: str = "pending") -> EvaluationRun:
    return EvaluationRun(
        id=uuid4(),
        dataset_name="kubernetes_eval",
        config_snapshot={
            "request": {
                "dataset_path": "evaluation/datasets/kubernetes_eval.jsonl",
                "experiment_name": "hybrid-rerank",
            },
            "overrides": {"retrieval_mode": "hybrid"},
        },
        started_at=datetime(2026, 7, 18, 4, tzinfo=UTC),
        finished_at=(datetime(2026, 7, 18, 5, tzinfo=UTC) if status == "succeeded" else None),
        summary={"recall_at_5": 0.9} if status == "succeeded" else {},
        report_path=report_path,
        status=status,
    )


def _app(service: _FakeEvaluationService, report_root: Path) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    install_exception_handlers(app)
    app.include_router(router)
    settings = Settings(_env_file=None, app_env="test", admin_api_key="admin-secret")

    async def settings_override() -> Settings:
        return settings

    async def service_override() -> EvaluationService:
        return cast(EvaluationService, service)

    async def report_root_override() -> Path:
        return report_root

    app.dependency_overrides[get_settings] = settings_override
    app.dependency_overrides[get_evaluation_service] = service_override
    app.dependency_overrides[get_evaluation_report_root] = report_root_override
    return app


@pytest.mark.asyncio
async def test_create_evaluation_returns_pending_202_without_running_cases(
    tmp_path: Path,
) -> None:
    service = _FakeEvaluationService(_run())
    transport = httpx.ASGITransport(app=_app(service, tmp_path / "evaluation/reports"))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/evaluations",
            headers={"x-admin-api-key": "admin-secret"},
            json={
                "dataset_name": "kubernetes_eval",
                "dataset_path": "kubernetes_eval.jsonl",
                "experiment_name": "hybrid-rerank",
                "config_snapshot": {"retrieval_mode": "hybrid"},
            },
        )

    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    assert response.json()["result_count"] == 0
    assert response.json()["report_available"] is False
    assert len(service.enqueue_calls) == 1
    assert service.enqueue_calls[0].dataset_path == ("evaluation/datasets/kubernetes_eval.jsonl")


@pytest.mark.asyncio
async def test_create_evaluation_returns_429_and_retry_after_when_queue_is_full(
    tmp_path: Path,
) -> None:
    service = _FakeEvaluationService(_run())
    service.rate_limited = True
    transport = httpx.ASGITransport(app=_app(service, tmp_path / "evaluation/reports"))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/evaluations",
            headers={
                "x-admin-api-key": "admin-secret",
                "x-request-id": "queue-full",
            },
            json={"dataset_name": "kubernetes_eval", "dataset_path": "kubernetes_eval.jsonl"},
        )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "11"
    assert response.json()["error"] == {
        "code": "RATE_LIMITED",
        "message": "Evaluation queue capacity exceeded",
        "request_id": "queue-full",
        "details": {"queue": "evaluation", "limit": 1},
    }


@pytest.mark.asyncio
async def test_polling_returns_summary_and_persisted_result_count(tmp_path: Path) -> None:
    run = _run(status="succeeded", report_path="evaluation/reports/run/report.json")
    service = _FakeEvaluationService(run)
    transport = httpx.ASGITransport(app=_app(service, tmp_path / "evaluation/reports"))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/api/v1/evaluations/{run.id}",
            headers={"x-admin-api-key": "admin-secret"},
        )

    assert response.status_code == 200
    assert response.json()["result_count"] == 3
    assert response.json()["summary"] == {"recall_at_5": 0.9}
    assert response.json()["report_available"] is True


@pytest.mark.asyncio
async def test_report_endpoint_serves_completed_report_beneath_root(
    tmp_path: Path,
    inline_to_thread: None,
) -> None:
    report_root = tmp_path / "evaluation/reports"
    report_directory = report_root / "run-1"
    report_directory.mkdir(parents=True)
    report_file = report_directory / "report.json"
    report_file.write_text('{"recall_at_5": 0.9}\n', encoding="utf-8")
    run = _run(status="succeeded", report_path=str(report_directory))
    transport = httpx.ASGITransport(app=_app(_FakeEvaluationService(run), report_root))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/api/v1/evaluations/{run.id}/report",
            headers={"x-admin-api-key": "admin-secret"},
        )

    assert response.status_code == 200
    assert response.json() == {"recall_at_5": 0.9}
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
async def test_report_endpoint_rejects_path_traversal_even_if_target_exists(
    tmp_path: Path,
    inline_to_thread: None,
) -> None:
    report_root = tmp_path / "evaluation/reports"
    report_root.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret": true}', encoding="utf-8")
    run = _run(status="succeeded", report_path=str(outside))
    transport = httpx.ASGITransport(app=_app(_FakeEvaluationService(run), report_root))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/api/v1/evaluations/{run.id}/report",
            headers={
                "x-admin-api-key": "admin-secret",
                "x-request-id": "path-request",
            },
        )

    assert response.status_code == 404
    assert response.json()["error"] == {
        "code": "NOT_FOUND",
        "message": "Evaluation report path is invalid",
        "request_id": "path-request",
        "details": {},
    }


@pytest.mark.asyncio
async def test_evaluation_endpoints_require_admin_auth_and_reject_dataset_traversal(
    tmp_path: Path,
) -> None:
    service = _FakeEvaluationService(_run())
    transport = httpx.ASGITransport(app=_app(service, tmp_path / "evaluation/reports"))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthorized = await client.post(
            "/api/v1/evaluations",
            json={"dataset_name": "test", "dataset_path": "dataset.jsonl"},
        )
        traversal = await client.post(
            "/api/v1/evaluations",
            headers={"x-admin-api-key": "admin-secret"},
            json={"dataset_name": "test", "dataset_path": "../secret.jsonl"},
        )

    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "AUTHENTICATION_FAILED"
    assert traversal.status_code == 422
    assert traversal.json()["error"]["code"] == "VALIDATION_FAILED"
    assert service.enqueue_calls == []
