"""Measure the HTTP chat path without calling model providers directly.

The target service decides which providers are used. Tests inject an HTTPX mock
transport, so this client never requires a paid API in the test suite.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from evaluation.metrics.performance import summarize_performance

DEFAULT_QUERIES = (
    "How do I roll back a Kubernetes Deployment?",
    "What does spec.selector mean on a Deployment?",
    "How can I inspect a failed Pod rollout?",
    "What changed in the latest indexed Kubernetes release notes?",
    "How do readiness probes affect Service endpoints?",
    "What is the difference between ClusterIP and NodePort?",
)


@dataclass(frozen=True, slots=True)
class RequestMeasurement:
    """One transport-level request measurement with bounded error metadata."""

    status_code: int | None
    latency_ms: float
    answerable: bool | None
    error: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--input",
        type=Path,
        help="Text or JSONL queries; JSON objects may use question or query",
    )
    parser.add_argument("--requests", type=int, default=20, dest="request_count")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20250717)
    parser.add_argument("--top-k", type=int, choices=range(1, 101))
    parser.add_argument("--mode", choices=("keyword", "vector", "hybrid"))
    parser.add_argument(
        "--api-key-env",
        default="API_KEY",
        help="Environment variable containing X-API-Key (never printed)",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args(argv)


def load_queries(path: Path | None) -> list[str]:
    """Load plain lines or evaluation-style JSONL while rejecting invalid rows."""
    if path is None:
        return list(DEFAULT_QUERIES)
    queries: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Unable to read benchmark input: {path}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        query = line
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on benchmark input line {line_number}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"Benchmark input line {line_number} must be a JSON object")
            value = payload.get("question", payload.get("query"))
            if not isinstance(value, str):
                raise ValueError(
                    f"Benchmark input line {line_number} needs a question or query string"
                )
            query = value.strip()
        if not query:
            raise ValueError(f"Benchmark query on line {line_number} must not be blank")
        if len(query) > 50_000:
            raise ValueError(f"Benchmark query on line {line_number} exceeds 50000 characters")
        queries.append(query)
    if not queries:
        raise ValueError("Benchmark input contains no queries")
    return queries


async def run_benchmark(
    *,
    base_url: str,
    queries: Sequence[str],
    request_count: int,
    concurrency: int,
    timeout_seconds: float,
    seed: int,
    api_key: str | None = None,
    top_k: int | None = None,
    mode: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Run a bounded concurrent workload and return transport-level statistics."""
    if not queries or any(not query.strip() for query in queries):
        raise ValueError("At least one non-blank benchmark query is required")
    if request_count < 1:
        raise ValueError("request count must be positive")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout must be positive")

    randomizer = random.Random(seed)
    workload = [queries[index % len(queries)] for index in range(request_count)]
    randomizer.shuffle(workload)
    queue: asyncio.Queue[str] = asyncio.Queue()
    for query in workload:
        queue.put_nowait(query)

    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    owns_client = client is None
    runtime_client = client or httpx.AsyncClient(
        base_url=f"{base_url.rstrip('/')}/",
        headers=headers,
        timeout=httpx.Timeout(timeout_seconds),
    )
    measurements: list[RequestMeasurement] = []

    async def worker() -> None:
        while True:
            try:
                query = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                measurements.append(
                    await _measure_request(
                        runtime_client,
                        query=query,
                        top_k=top_k,
                        mode=mode,
                    )
                )
            finally:
                queue.task_done()

    started = time.perf_counter()
    try:
        worker_count = min(concurrency, request_count)
        await asyncio.gather(*(worker() for _ in range(worker_count)))
    finally:
        if owns_client:
            await runtime_client.aclose()
    elapsed_seconds = time.perf_counter() - started
    return summarize_measurements(
        measurements,
        elapsed_seconds=elapsed_seconds,
        concurrency=concurrency,
    )


async def _measure_request(
    client: httpx.AsyncClient,
    *,
    query: str,
    top_k: int | None,
    mode: str | None,
) -> RequestMeasurement:
    payload: dict[str, Any] = {"query": query, "debug": False}
    if top_k is not None:
        payload["top_k"] = top_k
    if mode is not None:
        payload["mode"] = mode
    started = time.perf_counter()
    status_code: int | None = None
    try:
        response = await client.post("api/v1/chat", json=payload)
        status_code = response.status_code
        response.raise_for_status()
        decoded = response.json()
        if not isinstance(decoded, Mapping):
            raise ValueError("chat response is not a JSON object")
        answerable = decoded.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError("chat response has no boolean answerable field")
        error = None
    except httpx.HTTPStatusError as exc:
        answerable = None
        error = f"HTTP {exc.response.status_code}"
    except (httpx.HTTPError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        answerable = None
        error = type(exc).__name__
    return RequestMeasurement(
        status_code=status_code,
        latency_ms=(time.perf_counter() - started) * 1000,
        answerable=answerable,
        error=error,
    )


def summarize_measurements(
    measurements: Sequence[RequestMeasurement],
    *,
    elapsed_seconds: float,
    concurrency: int,
) -> dict[str, Any]:
    """Aggregate successful latency, all-attempt throughput, status, and errors."""
    successful = [item for item in measurements if item.error is None]
    failed = [item for item in measurements if item.error is not None]
    performance = summarize_performance(
        [item.latency_ms for item in successful],
        elapsed_seconds=elapsed_seconds,
        error_count=len(failed),
    )
    status_counts = Counter(
        str(item.status_code) if item.status_code is not None else "transport_error"
        for item in measurements
    )
    error_counts = Counter(item.error for item in failed if item.error is not None)
    return {
        "attempted_requests": len(measurements),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "answerable_responses": sum(item.answerable is True for item in successful),
        "refusal_responses": sum(item.answerable is False for item in successful),
        "concurrency": concurrency,
        "elapsed_seconds": elapsed_seconds,
        "latency_ms": {
            key.removesuffix("_ms"): value
            for key, value in asdict(performance).items()
            if key.endswith("_ms")
        },
        "throughput_requests_per_second": performance.throughput_per_second,
        "error_rate": performance.error_rate,
        "status_counts": dict(sorted(status_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
    }


async def run(args: argparse.Namespace) -> int:
    queries = load_queries(args.input)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    summary = await run_benchmark(
        base_url=args.base_url,
        queries=queries,
        request_count=args.request_count,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout,
        seed=args.seed,
        api_key=api_key,
        top_k=args.top_k,
        mode=args.mode,
    )
    encoded = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return int(summary["failed_requests"] > 0)


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
