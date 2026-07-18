"""Administrative endpoints for durable, asynchronous offline evaluations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated
from uuid import UUID

import orjson
from fastapi import APIRouter, Depends, status
from fastapi.responses import ORJSONResponse

from app.api.dependencies import get_container
from app.container import AppContainer
from app.core.exceptions import NotFoundError
from app.core.security import require_admin_api_key
from app.domain.evaluation import EvaluationRunStatus
from app.schemas.evaluation import EvaluationRunCreateRequest, EvaluationRunResponse
from app.services.evaluation_service import EvaluationRunView, EvaluationService

REPORT_ROOT = Path("evaluation/reports")

router = APIRouter(
    prefix="/api/v1/evaluations",
    tags=["evaluation"],
    dependencies=[Depends(require_admin_api_key)],
)


def get_evaluation_service(
    container: Annotated[AppContainer, Depends(get_container)],
) -> EvaluationService:
    """Build the request-independent queue service around the process database."""
    return EvaluationService(
        container.database.session_factory,
        max_outstanding_runs=container.settings.evaluation_max_outstanding_runs,
        retry_after_seconds=container.settings.queue_retry_after_seconds,
    )


async def get_evaluation_report_root() -> Path:
    """Return the only directory from which report files may be served."""
    return REPORT_ROOT


@router.post("", response_model=EvaluationRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_evaluation(
    payload: EvaluationRunCreateRequest,
    service: Annotated[EvaluationService, Depends(get_evaluation_service)],
) -> EvaluationRunResponse:
    """Enqueue a pending evaluation; execution never blocks this HTTP request."""
    return _run_response(await service.enqueue(payload))


@router.get("/{run_id}", response_model=EvaluationRunResponse)
async def get_evaluation(
    run_id: UUID,
    service: Annotated[EvaluationService, Depends(get_evaluation_service)],
) -> EvaluationRunResponse:
    """Poll status, aggregate summary, and persisted result count."""
    return _run_response(await service.get(run_id))


@router.get("/{run_id}/report", response_class=ORJSONResponse)
async def get_evaluation_report(
    run_id: UUID,
    service: Annotated[EvaluationService, Depends(get_evaluation_service)],
    report_root: Annotated[Path, Depends(get_evaluation_report_root)],
) -> ORJSONResponse:
    """Serve a completed report while preventing path and symlink traversal."""
    view = await service.get(run_id)
    run = view.run
    if run.status != EvaluationRunStatus.SUCCEEDED.value or not run.report_path:
        raise NotFoundError("Evaluation report is not available")
    # Resolution, stat calls, and reading are all offloaded from the async request loop.
    content = await asyncio.to_thread(
        read_report_bytes,
        run.report_path,
        report_root=report_root,
    )
    try:
        report = orjson.loads(content)
    except orjson.JSONDecodeError as exc:
        raise NotFoundError("Evaluation report is invalid") from exc
    return ORJSONResponse(
        content=report,
        headers={"Cache-Control": "no-store"},
    )


def resolve_report_file(stored_path: str, *, report_root: Path = REPORT_ROOT) -> Path:
    """Resolve a DB path beneath the approved report root and require a real report file."""
    root = report_root.resolve(strict=False)
    requested = Path(stored_path)
    raw_candidates = (
        [requested] if requested.is_absolute() else [Path.cwd() / requested, root / requested]
    )
    candidate = next(
        (
            path.resolve(strict=False)
            for path in raw_candidates
            if path.resolve(strict=False).is_relative_to(root)
        ),
        None,
    )
    if candidate is None:
        raise NotFoundError("Evaluation report path is invalid")
    if candidate.is_dir():
        candidate = (candidate / "report.json").resolve(strict=False)
    elif candidate.name != "report.json":
        candidate = (candidate.parent / "report.json").resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise NotFoundError("Evaluation report path is invalid")
    if candidate.name != "report.json" or not candidate.is_file():
        raise NotFoundError("Evaluation report is not available")
    return candidate


def read_report_bytes(stored_path: str, *, report_root: Path = REPORT_ROOT) -> bytes:
    """Resolve and read a report in one offloadable filesystem operation."""
    return resolve_report_file(stored_path, report_root=report_root).read_bytes()


def _run_response(view: EvaluationRunView) -> EvaluationRunResponse:
    run = view.run
    return EvaluationRunResponse(
        id=run.id,
        dataset_name=run.dataset_name,
        status=EvaluationRunStatus(run.status),
        config_snapshot=dict(run.config_snapshot or {}),
        started_at=run.started_at,
        finished_at=run.finished_at,
        summary=dict(run.summary or {}),
        result_count=view.result_count,
        report_available=(
            run.status == EvaluationRunStatus.SUCCEEDED.value and bool(run.report_path)
        ),
    )
