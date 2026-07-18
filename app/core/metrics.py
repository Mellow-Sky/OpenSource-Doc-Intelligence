"""Low-cardinality Prometheus instrumentation for HTTP traffic."""

from __future__ import annotations

import time
from typing import Final

from prometheus_client import Counter, Histogram
from starlette.routing import BaseRoute
from starlette.types import ASGIApp, Message, Receive, Scope, Send

HTTP_REQUESTS: Final = Counter(
    "odi_http_requests_total",
    "Total HTTP requests handled by the API.",
    labelnames=("method", "route", "status_code"),
)
HTTP_REQUEST_DURATION: Final = Histogram(
    "odi_http_request_duration_seconds",
    "HTTP request duration from ASGI entry through response completion.",
    labelnames=("method", "route"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
HTTP_RESPONSE_SIZE: Final = Histogram(
    "odi_http_response_size_bytes",
    "HTTP response body size observed by the ASGI server.",
    labelnames=("method", "route"),
    buckets=(100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000),
)


class PrometheusMiddleware:
    """Observe completed HTTP requests using normalized route templates as labels."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        started = time.perf_counter()
        status_code = 500
        response_size = 0

        async def observe_send(message: Message) -> None:
            nonlocal response_size, status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            elif message["type"] == "http.response.body":
                response_size += len(message.get("body", b""))
            await send(message)

        try:
            await self._app(scope, receive, observe_send)
        finally:
            route = _route_label(scope)
            method = str(scope.get("method", "UNKNOWN"))
            HTTP_REQUESTS.labels(method=method, route=route, status_code=str(status_code)).inc()
            HTTP_REQUEST_DURATION.labels(method=method, route=route).observe(
                time.perf_counter() - started
            )
            HTTP_RESPONSE_SIZE.labels(method=method, route=route).observe(response_size)


def _route_label(scope: Scope) -> str:
    route = scope.get("route")
    if isinstance(route, BaseRoute):
        path = getattr(route, "path", None)
        if isinstance(path, str):
            return path
    return "__unmatched__"
