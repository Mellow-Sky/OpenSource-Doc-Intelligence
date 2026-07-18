"""Small, rate-limit-aware GitHub REST client used by source loaders."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, cast

import httpx
from pydantic import SecretStr

from app.ingestion.loaders.base import LoaderError

SleepCallable = Callable[[float], Awaitable[None]]
ClockCallable = Callable[[], float]

_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class GitHubAPIClient:
    """Authenticated GitHub REST client with pagination and bounded retries."""

    def __init__(
        self,
        *,
        repository: str,
        token: str | SecretStr | None = None,
        client: httpx.AsyncClient | None = None,
        api_base_url: str = "https://api.github.com",
        per_page: int = 100,
        max_pages: int = 1000,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        max_retry_delay_seconds: float = 60.0,
        sleep: SleepCallable = asyncio.sleep,
        clock: ClockCallable = time.time,
    ) -> None:
        if repository.count("/") != 1:
            msg = "GitHub repository must use owner/name form"
            raise ValueError(msg)
        if not 1 <= per_page <= 100:
            msg = "GitHub per_page must be between 1 and 100"
            raise ValueError(msg)
        if max_pages < 1 or max_retries < 0:
            msg = "GitHub max_pages must be positive and max_retries cannot be negative"
            raise ValueError(msg)

        self.repository = repository
        self.api_base_url = api_base_url.rstrip("/")
        self.per_page = per_page
        self.max_pages = max_pages
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self._sleep = sleep
        self._clock = clock
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        raw_token = token.get_secret_value() if isinstance(token, SecretStr) else token
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "opensource-doc-intelligence/0.1",
        }
        if raw_token:
            self._headers["Authorization"] = f"Bearer {raw_token}"

    def __repr__(self) -> str:
        """Return a diagnostic representation that never exposes credentials."""
        return f"GitHubAPIClient(repository={self.repository!r})"

    async def aclose(self) -> None:
        """Close an internally-created HTTP client."""
        if self._owns_client:
            await self._client.aclose()

    async def request(
        self,
        url_or_path: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> httpx.Response:
        """GET one endpoint, retrying transient and rate-limit responses."""
        url = self._absolute_url(url_or_path)
        last_transport_error: httpx.TransportError | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.get(url, params=params, headers=self._headers)
            except httpx.TransportError as exc:
                last_transport_error = exc
                if attempt >= self.max_retries:
                    break
                await self._sleep(self._exponential_delay(attempt))
                continue

            should_retry = response.status_code in _TRANSIENT_STATUS_CODES or (
                response.status_code == 403 and self._is_rate_limited(response)
            )
            if should_retry and attempt < self.max_retries:
                await self._sleep(self._retry_delay(response, attempt))
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                msg = (
                    f"GitHub API request failed with status {response.status_code} "
                    f"for {response.request.url.copy_with(query=None)}"
                )
                raise LoaderError(msg) from exc
            return response

        if last_transport_error is not None:
            msg = f"GitHub API request failed after retries for {url}"
            raise LoaderError(msg) from last_transport_error
        msg = f"GitHub API request exhausted retries for {url}"
        raise LoaderError(msg)

    async def get_paginated(
        self,
        url_or_path: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> list[dict[str, Any]]:
        """Collect a GitHub list endpoint while honoring Link pagination."""
        query = dict(params or {})
        query["per_page"] = self.per_page
        query.setdefault("page", 1)
        next_url = url_or_path
        rows: list[dict[str, Any]] = []

        for page_number in range(1, self.max_pages + 1):
            response = await self.request(next_url, params=query)
            payload = response.json()
            if not isinstance(payload, list):
                msg = "GitHub API list endpoint returned a non-list payload"
                raise LoaderError(msg)
            page_rows: list[dict[str, Any]] = []
            for item in payload:
                if not isinstance(item, dict):
                    msg = "GitHub API list endpoint returned a non-object item"
                    raise LoaderError(msg)
                page_rows.append(cast(dict[str, Any], item))
            rows.extend(page_rows)

            linked_next = _next_link(response.headers.get("Link"))
            if linked_next is not None:
                next_url = linked_next
                query = {}
                continue
            if len(page_rows) < self.per_page:
                return rows
            next_url = url_or_path
            query["page"] = page_number + 1

        msg = f"GitHub API pagination exceeded configured max_pages={self.max_pages}"
        raise LoaderError(msg)

    def _absolute_url(self, url_or_path: str) -> str:
        if url_or_path.startswith(("https://", "http://")):
            return url_or_path
        return f"{self.api_base_url}/{url_or_path.lstrip('/')}"

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            parsed = _parse_retry_after(retry_after, now=self._clock())
            if parsed is not None:
                return min(parsed, self.max_retry_delay_seconds)

        reset = response.headers.get("X-RateLimit-Reset")
        if reset and self._is_rate_limited(response):
            try:
                reset_delay = max(0.0, float(reset) - self._clock())
            except ValueError:
                reset_delay = 0.0
            if reset_delay > 0:
                return min(reset_delay, self.max_retry_delay_seconds)
        return self._exponential_delay(attempt)

    def _exponential_delay(self, attempt: int) -> float:
        return min(
            self.backoff_base_seconds * (2.0**attempt),
            self.max_retry_delay_seconds,
        )

    @staticmethod
    def _is_rate_limited(response: httpx.Response) -> bool:
        if response.headers.get("X-RateLimit-Remaining") == "0" or response.status_code == 429:
            return True
        if response.status_code != 403:
            return False
        if response.headers.get("Retry-After") is not None:
            return True
        # GitHub secondary/abuse limits commonly retain a positive primary quota.
        return "secondary rate limit" in response.text.casefold()


def _next_link(header: str | None) -> str | None:
    if not header:
        return None
    for value in header.split(","):
        segments = [segment.strip() for segment in value.split(";")]
        if len(segments) < 2 or 'rel="next"' not in segments[1:]:
            continue
        target = segments[0]
        if target.startswith("<") and target.endswith(">"):
            return target[1:-1]
    return None


def _parse_retry_after(value: str, *, now: float) -> float | None:
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0.0, parsed.timestamp() - now)


def parse_github_datetime(value: object) -> datetime | None:
    """Parse GitHub's UTC timestamps into timezone-aware datetimes."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
