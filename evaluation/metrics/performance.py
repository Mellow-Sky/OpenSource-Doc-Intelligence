"""Latency, throughput, and error summaries for measured evaluation requests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_per_second: float
    error_rate: float


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile."""
    if not values:
        return 0.0
    if not 0 <= quantile <= 1:
        msg = "quantile must be between zero and one"
        raise ValueError(msg)
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_performance(
    latency_ms: Sequence[float],
    *,
    elapsed_seconds: float,
    error_count: int = 0,
) -> PerformanceSummary:
    """Summarize successful latencies and all-attempt throughput/error rate."""
    if error_count < 0:
        msg = "error_count cannot be negative"
        raise ValueError(msg)
    successful = len(latency_ms)
    total = successful + error_count
    return PerformanceSummary(
        mean_ms=fmean(latency_ms) if latency_ms else 0.0,
        p50_ms=percentile(latency_ms, 0.50),
        p90_ms=percentile(latency_ms, 0.90),
        p95_ms=percentile(latency_ms, 0.95),
        p99_ms=percentile(latency_ms, 0.99),
        min_ms=min(latency_ms, default=0.0),
        max_ms=max(latency_ms, default=0.0),
        throughput_per_second=total / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        error_rate=error_count / total if total else 0.0,
    )
