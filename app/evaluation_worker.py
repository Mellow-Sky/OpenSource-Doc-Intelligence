"""Durable PostgreSQL-backed evaluation worker with bounded recovery and graceful shutdown."""

from __future__ import annotations

import argparse
import asyncio
import re
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType
from uuid import UUID

import structlog

from app.container import AppContainer
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.models.evaluation import EvaluationRun
from app.db.session import Database
from app.repositories.evaluation_repository import EvaluationRepository
from app.services.evaluation_execution_service import EvaluationExecutionService
from app.services.evaluation_service import EvaluationLeaseLostError, EvaluationService
from evaluation.models import EvaluationRunReport

_MAX_ERROR_CHARACTERS = 4_000
_AUTH_PATTERN = re.compile(r"(?i)(bearer\s+|(?:api[_ -]?key|token)\s*[=:]\s*)\S+")
_URL_CREDENTIAL_PATTERN = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)


class EvaluationWorker:
    """Claim one run at a time and persist complete per-case/report output."""

    def __init__(
        self,
        container: AppContainer,
        *,
        dataset_root: Path = Path("evaluation/datasets"),
        report_root: Path = Path("evaluation/reports"),
    ) -> None:
        self._container = container
        self._report_root = report_root
        self._executor = EvaluationExecutionService(
            container.settings,
            dataset_root=dataset_root,
            report_root=report_root,
        )
        self._persistence = EvaluationService(
            container.database.session_factory,
            max_outstanding_runs=container.settings.evaluation_max_outstanding_runs,
            retry_after_seconds=container.settings.queue_retry_after_seconds,
        )
        self._stop = asyncio.Event()
        self._logger = structlog.get_logger(__name__)

    def request_stop(self) -> None:
        """Stop polling after the current evaluation reaches a terminal state."""
        self._stop.set()

    async def run(self, *, once: bool = False) -> None:
        """Poll until stopped; once mode performs at most one claim."""
        while not self._stop.is_set():
            claimed = await self.run_once()
            if once:
                return
            if claimed:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self._container.settings.evaluation_worker_poll_seconds,
                )
            except TimeoutError:
                continue

    async def run_once(self) -> bool:
        """Claim, execute, and finish at most one queued evaluation."""
        claimed_at = datetime.now(UTC)
        stale_before = claimed_at - timedelta(
            seconds=self._container.settings.evaluation_stale_seconds
        )
        async with self._container.database.session_factory.begin() as session:
            run = await EvaluationRepository(session).claim_next(
                claimed_at=claimed_at,
                stale_before=stale_before,
                max_running=self._container.settings.evaluation_max_concurrent_runs,
            )
        if run is None:
            return False
        lease_started_at = run.started_at
        if lease_started_at is None:
            raise RuntimeError("claimed evaluation run did not receive a lease timestamp")
        lease_output = (
            self._report_root
            / str(run.id)
            / lease_started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        )
        self._logger.info("evaluation_run_started", run_id=str(run.id))
        try:
            report = await self._execute_with_heartbeat(
                run,
                lease_started_at=lease_started_at,
                output_directory=lease_output,
            )
            report_path = lease_output / "report.json"
            await self._persistence.complete(
                run,
                report,
                report_path=report_path,
                lease_started_at=lease_started_at,
            )
        except EvaluationLeaseLostError:
            self._logger.warning("evaluation_run_lease_lost", run_id=str(run.id))
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = _safe_error(exc, self._configured_secrets())
            async with self._container.database.session_factory.begin() as session:
                updated = await EvaluationRepository(session).mark_failed(
                    run.id,
                    error=error,
                    lease_started_at=lease_started_at,
                )
            if updated:
                self._logger.error("evaluation_run_failed", run_id=str(run.id), error=error)
            else:
                self._logger.warning(
                    "evaluation_run_failure_cas_rejected",
                    run_id=str(run.id),
                )
            return True
        self._logger.info(
            "evaluation_run_succeeded",
            run_id=str(run.id),
            case_count=report.dataset_size,
        )
        return True

    async def _execute_with_heartbeat(
        self,
        run: EvaluationRun,
        *,
        lease_started_at: datetime,
        output_directory: Path,
    ) -> EvaluationRunReport:
        """Run slow evaluation while periodically proving ownership in PostgreSQL."""

        execution_task = asyncio.create_task(
            self._executor.execute(run, output_directory=output_directory)
        )
        heartbeat_stop = asyncio.Event()
        heartbeat_task = asyncio.create_task(
            self._heartbeat(run.id, lease_started_at, heartbeat_stop)
        )
        done, _ = await asyncio.wait(
            {execution_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task in done:
            try:
                still_owned = heartbeat_task.result()
            except Exception:
                execution_task.cancel()
                await asyncio.gather(execution_task, return_exceptions=True)
                raise
            if not still_owned:
                execution_task.cancel()
                await asyncio.gather(execution_task, return_exceptions=True)
                raise EvaluationLeaseLostError("evaluation heartbeat CAS was rejected")
        try:
            return await execution_task
        finally:
            heartbeat_stop.set()
            await heartbeat_task

    async def _heartbeat(
        self,
        run_id: UUID,
        lease_started_at: datetime,
        stop: asyncio.Event,
    ) -> bool:
        while not stop.is_set():
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=self._container.settings.evaluation_heartbeat_seconds,
                )
                return True
            except TimeoutError:
                async with self._container.database.session_factory.begin() as session:
                    owned = await EvaluationRepository(session).heartbeat(
                        run_id,
                        lease_started_at=lease_started_at,
                        heartbeat_at=datetime.now(UTC),
                    )
                if not owned:
                    return False
        return True

    def _configured_secrets(self) -> list[str]:
        settings = self._container.settings
        protected = (
            settings.github_token,
            settings.llm_api_key,
            settings.embedding_api_key,
            settings.reranker_api_key,
            settings.judge_api_key,
            settings.api_key,
            settings.admin_api_key,
        )
        return [value.get_secret_value() for value in protected if value is not None]


def _safe_error(exc: Exception, secrets: list[str]) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = _AUTH_PATTERN.sub(r"\1[REDACTED]", message)
    message = _URL_CREDENTIAL_PATTERN.sub(r"\1[REDACTED]@", message)
    return message.replace("\n", " ")[:_MAX_ERROR_CHARACTERS]


def _install_signal_handlers(worker: EvaluationWorker) -> None:
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, worker.request_stop)
        except NotImplementedError:
            signal.signal(
                signum,
                lambda _signal, _frame: _fallback_stop(worker, _signal, _frame),
            )


def _fallback_stop(
    worker: EvaluationWorker,
    _signum: int,
    _frame: FrameType | None,
) -> None:
    worker.request_stop()


async def run_worker(*, once: bool = False) -> None:
    """Build infrastructure and execute the evaluation polling loop."""
    settings = get_settings()
    configure_logging(settings.log_level)
    container = AppContainer(settings=settings, database=Database(settings))
    worker = EvaluationWorker(container)
    _install_signal_handlers(worker)
    try:
        await worker.run(once=once)
    finally:
        await container.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Attempt one run and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_worker(once=args.once))


if __name__ == "__main__":
    main()
