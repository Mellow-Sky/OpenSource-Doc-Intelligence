"""Lease-based ingestion worker with graceful shutdown and bounded failures."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import Any
from uuid import UUID

import structlog

from app.container import AppContainer
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import Database
from app.ingestion.chunkers import ChunkingConfig, TokenCounter, create_chunk_token_counter
from app.ingestion.incremental import SyncStats
from app.providers.factory import create_embedding_provider
from app.repositories.ingestion_job_repository import IngestionJobRepository
from app.services.ingestion_service import IngestionService, SyncResult
from app.services.pricing_service import PricingCatalog

_DEFAULT_LEASE_SECONDS = 300.0
_DEFAULT_HEARTBEAT_SECONDS = 30.0
_MAX_ERROR_CHARACTERS = 2000
_AUTH_PATTERN = re.compile(r"(?i)(bearer\s+|(?:api[_ -]?key|token)\s*[=:]\s*)\S+")
_URL_CREDENTIAL_PATTERN = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)


class LeaseLostError(RuntimeError):
    """Raised when another worker has taken ownership of the durable job."""


class IngestionWorker:
    """Claim jobs in short transactions and execute slow synchronization separately."""

    def __init__(
        self,
        container: AppContainer,
        *,
        cache_root: Path | None = None,
        lease_seconds: float = _DEFAULT_LEASE_SECONDS,
        heartbeat_seconds: float = _DEFAULT_HEARTBEAT_SECONDS,
        token_counter: TokenCounter,
    ) -> None:
        if lease_seconds <= 0 or heartbeat_seconds <= 0:
            raise ValueError("lease and heartbeat durations must be positive")
        if heartbeat_seconds >= lease_seconds:
            raise ValueError("heartbeat duration must be shorter than the lease")
        self._container = container
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._stop = asyncio.Event()
        self._logger = structlog.get_logger(__name__)
        self._service = IngestionService(
            container.database.session_factory,
            embedding_provider=container.embedding_provider,
            embedding_dimension=container.settings.embedding_dimension,
            embedding_batch_size=container.settings.embedding_batch_size,
            cache_root=cache_root
            or Path(os.environ.get("INGESTION_CACHE_DIR", ".cache/ingestion")),
            github_token=container.settings.github_token,
            chunking_config=_chunking_config(container),
            token_counter=token_counter,
            pricing_catalog=PricingCatalog.from_file(
                container.settings.pricing_config_path,
            ),
        )

    def request_stop(self) -> None:
        """Stop polling after the currently leased job reaches a terminal state."""

        self._stop.set()

    async def run(self, *, once: bool = False) -> None:
        """Poll until signalled; ``once`` performs at most one claim."""

        while not self._stop.is_set():
            claimed = await self.run_once()
            if once:
                return
            if claimed:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self._container.settings.ingestion_worker_poll_seconds,
                )
            except TimeoutError:
                continue

    async def run_once(self) -> bool:
        """Claim and finish at most one job, returning whether a claim occurred."""

        claimed_at = datetime.now(UTC)
        async with self._container.database.session_factory.begin() as session:
            job = await IngestionJobRepository(session).claim_next(
                claimed_at=claimed_at,
                stale_before=claimed_at - timedelta(seconds=self._lease_seconds),
                max_running=self._container.settings.ingestion_max_concurrent_jobs,
            )
        if job is None:
            return False
        lease_started_at = job.started_at
        if lease_started_at is None:
            raise RuntimeError("claimed ingestion job did not receive a lease timestamp")

        self._logger.info(
            "ingestion_job_started",
            job_id=str(job.id),
            source_id=str(job.source_id),
        )
        try:
            result = await self._synchronize_with_heartbeat(
                job.id,
                job.source_id,
                lease_started_at,
                dict(job.options or {}),
            )
        except LeaseLostError:
            self._logger.warning("ingestion_job_lease_lost", job_id=str(job.id))
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._mark_failed(job.id, lease_started_at, exc)
            return True

        finished_at = datetime.now(UTC)
        async with self._container.database.session_factory.begin() as session:
            updated = await IngestionJobRepository(session).mark_succeeded(
                job.id,
                lease_started_at=lease_started_at,
                stats=result.stats,
                finished_at=finished_at,
            )
            if not updated:
                self._logger.warning(
                    "ingestion_job_success_cas_rejected",
                    job_id=str(job.id),
                )
                return True
        self._logger.info(
            "ingestion_job_succeeded",
            job_id=str(job.id),
            stats=result.stats.model_dump(),
        )
        return True

    async def _synchronize_with_heartbeat(
        self,
        job_id: UUID,
        source_id: UUID,
        lease_started_at: datetime,
        options: dict[str, Any],
    ) -> SyncResult:
        heartbeat_stop = asyncio.Event()
        sync_task = asyncio.create_task(
            self._service.sync_source(
                source_id,
                dry_run=_option_bool(options, "dry_run", default=False),
                allow_delete_missing=_option_bool(
                    options,
                    "allow_delete_missing",
                    default=True,
                ),
                request_id=job_id,
            )
        )
        heartbeat_task = asyncio.create_task(
            self._heartbeat(job_id, lease_started_at, heartbeat_stop)
        )
        done, _ = await asyncio.wait(
            {sync_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task in done:
            try:
                still_owned = heartbeat_task.result()
            except Exception:
                sync_task.cancel()
                await asyncio.gather(sync_task, return_exceptions=True)
                raise
            if not still_owned:
                sync_task.cancel()
                await asyncio.gather(sync_task, return_exceptions=True)
                raise LeaseLostError("heartbeat CAS rejected for expired ingestion lease")
        try:
            return await sync_task
        finally:
            heartbeat_stop.set()
            await heartbeat_task

    async def _heartbeat(
        self,
        job_id: UUID,
        lease_started_at: datetime,
        stop: asyncio.Event,
    ) -> bool:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._heartbeat_seconds)
                return True
            except TimeoutError:
                heartbeat_at = datetime.now(UTC)
                async with self._container.database.session_factory.begin() as session:
                    owned = await IngestionJobRepository(session).heartbeat(
                        job_id,
                        lease_started_at=lease_started_at,
                        heartbeat_at=heartbeat_at,
                    )
                if not owned:
                    return False
        return True

    async def _mark_failed(
        self,
        job_id: UUID,
        lease_started_at: datetime,
        exc: Exception,
    ) -> None:
        error = _safe_error(exc, self._configured_secrets())
        stats = SyncStats(errors=1)
        finished_at = datetime.now(UTC)
        async with self._container.database.session_factory.begin() as session:
            updated = await IngestionJobRepository(session).mark_failed(
                job_id,
                lease_started_at=lease_started_at,
                error=error,
                stats=stats,
                finished_at=finished_at,
            )
        if updated:
            self._logger.error("ingestion_job_failed", job_id=str(job_id), error=error)
        else:
            self._logger.warning("ingestion_job_failure_cas_rejected", job_id=str(job_id))

    def _configured_secrets(self) -> list[str]:
        settings = self._container.settings
        protected = (
            settings.github_token,
            settings.llm_api_key,
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
    return message[:_MAX_ERROR_CHARACTERS]


def _option_bool(options: dict[str, Any], key: str, *, default: bool) -> bool:
    value = options.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"ingestion job option {key} must be boolean")
    return value


def _chunking_config(container: AppContainer) -> ChunkingConfig:
    settings = container.settings
    return ChunkingConfig(
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_tokens=settings.chunk_min_tokens,
    )


def _install_signal_handlers(worker: IngestionWorker) -> None:
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
    worker: IngestionWorker,
    _signum: int,
    _frame: FrameType | None,
) -> None:
    worker.request_stop()


async def run_worker(*, once: bool = False) -> None:
    """Create infrastructure, install signals, and run the worker process."""

    settings = get_settings()
    configure_logging(settings.log_level)
    container = AppContainer(
        settings=settings,
        database=Database(settings),
        embedding_provider=create_embedding_provider(settings),
    )
    try:
        token_counter = await create_chunk_token_counter(
            settings,
            container.embedding_provider,
        )
        worker = IngestionWorker(container, token_counter=token_counter)
        _install_signal_handlers(worker)
        await worker.run(once=once)
    finally:
        await container.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Attempt one job claim and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_worker(once=args.once))


if __name__ == "__main__":
    main()
