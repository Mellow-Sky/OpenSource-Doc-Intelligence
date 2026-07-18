"""Small retrying JSON client shared by OpenAI-compatible providers."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import SecretStr

from app.core.exceptions import ProviderError, RateLimitError

_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class RetryingJSONClient:
    """Issue bounded concurrent requests without exposing credentials or bodies."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | None,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        client: httpx.AsyncClient | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._owns_client = client is None
        headers = {"Accept": "application/json"}
        if api_key is not None and api_key.get_secret_value():
            headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
        if request_headers is not None:
            headers.update(request_headers)
        self._request_headers = headers
        self._client = client or httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
        )

    def endpoint(self, resource: str) -> str:
        """Append a resource unless the configured URL already targets it."""
        if resource.startswith(("http://", "https://")):
            return resource
        normalized = resource.strip("/")
        if self._base_url.rsplit("/", maxsplit=1)[-1] == normalized:
            return self._base_url
        return f"{self._base_url}/{normalized}"

    async def request_json(
        self,
        method: str,
        resource: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Return a JSON object and retry temporary network or HTTP failures."""
        last_status: int | None = None
        async with self._semaphore:
            for attempt in range(self._max_retries + 1):
                try:
                    response = await self._client.request(
                        method,
                        self.endpoint(resource),
                        json=dict(payload) if payload is not None else None,
                        headers=self._request_headers,
                    )
                    last_status = response.status_code
                    if response.status_code in _RETRYABLE_STATUS_CODES:
                        if attempt < self._max_retries:
                            await asyncio.sleep(_retry_delay(response, attempt))
                            continue
                        if response.status_code == 429:
                            raise RateLimitError("Model provider rate limit was exhausted")
                        raise ProviderError(
                            "Model provider remained unavailable after bounded retries",
                            details={"status_code": response.status_code},
                        )
                    response.raise_for_status()
                    decoded = response.json()
                    if not isinstance(decoded, Mapping):
                        raise ProviderError("Model provider returned a non-object JSON response")
                    return decoded
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt >= self._max_retries:
                        raise ProviderError(
                            "Model provider request failed after bounded retries"
                        ) from exc
                    await asyncio.sleep(0.25 * (2**attempt))
                except httpx.HTTPStatusError as exc:
                    raise ProviderError(
                        "Model provider rejected the request",
                        details={"status_code": exc.response.status_code},
                    ) from exc
        raise ProviderError(
            "Model provider request failed",
            details={"status_code": last_status},
        )

    async def healthcheck(self, *, model: str | None = None) -> None:
        """Verify authentication and, when requested, catalog model availability."""
        decoded = await self.request_json("GET", "models")
        if model is not None:
            validate_model_catalog(decoded, model)

    async def close(self) -> None:
        """Close only clients created by this wrapper."""
        if self._owns_client:
            await self._client.aclose()


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), 10.0)
        except ValueError:
            pass
    return float(min(0.25 * (2**attempt), 5.0))


def validate_model_catalog(decoded: Mapping[str, Any], expected_model: str) -> None:
    """Require an OpenAI-compatible model-list response containing one exact ID."""
    raw_models = decoded.get("data")
    if not isinstance(raw_models, list):
        raise ProviderError("Model provider returned an invalid model catalog")
    model_ids = {
        item.get("id")
        for item in raw_models
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if expected_model not in model_ids:
        raise ProviderError(
            "Configured model is absent from the provider catalog",
            details={"model": expected_model},
        )
