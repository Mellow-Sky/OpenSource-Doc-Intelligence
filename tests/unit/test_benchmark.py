from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from scripts.benchmark import (
    RequestMeasurement,
    load_queries,
    parse_args,
    run_benchmark,
    summarize_measurements,
)


def test_load_queries_accepts_evaluation_jsonl_and_plain_text(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"question": "How do Deployments roll back?"}),
                json.dumps({"query": "What is a StatefulSet?"}),
                "How do readiness probes work?",
            ]
        ),
        encoding="utf-8",
    )

    assert load_queries(path) == [
        "How do Deployments roll back?",
        "What is a StatefulSet?",
        "How do readiness probes work?",
    ]


@pytest.mark.asyncio
async def test_benchmark_uses_mock_transport_and_reports_errors_without_bodies() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/api/v1/chat"
        if payload["query"] == "fails":
            return httpx.Response(503, json={"secret": "must-not-be-reported"})
        return httpx.Response(200, json={"answerable": payload["query"] == "answerable"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://rag.test/",
    ) as client:
        summary = await run_benchmark(
            base_url="http://ignored.test",
            queries=["answerable", "refused", "fails"],
            request_count=3,
            concurrency=2,
            timeout_seconds=1,
            seed=7,
            client=client,
        )

    assert summary["attempted_requests"] == 3
    assert summary["successful_requests"] == 2
    assert summary["failed_requests"] == 1
    assert summary["answerable_responses"] == 1
    assert summary["refusal_responses"] == 1
    assert summary["status_counts"] == {"200": 2, "503": 1}
    assert summary["error_counts"] == {"HTTP 503": 1}
    assert "secret" not in json.dumps(summary)


def test_summary_and_cli_validation_are_deterministic() -> None:
    summary = summarize_measurements(
        [
            RequestMeasurement(200, 10.0, True),
            RequestMeasurement(200, 30.0, False),
            RequestMeasurement(None, 5.0, None, "ConnectError"),
        ],
        elapsed_seconds=1.0,
        concurrency=2,
    )

    assert summary["latency_ms"]["mean"] == 20.0
    assert summary["latency_ms"]["p50"] == 20.0
    assert summary["throughput_requests_per_second"] == 3.0
    assert summary["error_rate"] == pytest.approx(1 / 3)
    assert parse_args([]).request_count == 20


def test_load_queries_rejects_invalid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"id": "missing-question"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="question or query"):
        load_queries(path)
