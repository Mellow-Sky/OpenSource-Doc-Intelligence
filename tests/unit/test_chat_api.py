"""Chat HTTP and citation-safe SSE contracts with fully offline dependencies."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from app.api.exception_handlers import RequestIDMiddleware, install_exception_handlers
from app.api.routes.chat import get_chat_service, router
from app.core.security import require_api_key
from app.domain.chat import ChatLatency, ChatResult, ChatUsage
from app.domain.citations import Citation, CitationReport
from app.domain.retrieval import (
    NoAnswerDecision,
    QueryFilters,
    RetrievalCandidate,
    RetrievalMode,
    RetrievalOutcome,
    RetrievalQuery,
    RetrievalTimings,
)

REQUEST_ID = UUID("10000000-0000-0000-0000-000000000001")
CONVERSATION_ID = UUID("20000000-0000-0000-0000-000000000002")
MESSAGE_ID = UUID("30000000-0000-0000-0000-000000000003")
DOCUMENT_ID = UUID("40000000-0000-0000-0000-000000000004")
CHUNK_ID = UUID("50000000-0000-0000-0000-000000000005")


class FakeChatService:
    """Capture transport inputs and return an already validated domain result."""

    def __init__(self, result: ChatResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        query: str,
        *,
        request_id: UUID | None = None,
        conversation_id: UUID | None = None,
        filters: QueryFilters | None = None,
        mode: RetrievalMode | None = None,
        top_k: int | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "query": query,
                "request_id": request_id,
                "conversation_id": conversation_id,
                "filters": filters,
                "mode": mode,
                "top_k": top_k,
            }
        )
        return self.result.model_copy(update={"request_id": request_id or self.result.request_id})


def _chat_result() -> ChatResult:
    candidate = RetrievalCandidate(
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        document_title="Kubernetes Deployments",
        document_type="official_documentation",
        heading_path=["Deployments", "Rolling Back"],
        content="Use kubectl rollout undo to roll back a Deployment.",
        canonical_url="https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
        keyword_rank=2,
        vector_rank=1,
        fused_rank=1,
        rerank_rank=1,
        keyword_score=0.7,
        vector_score=0.9,
        fused_score=0.03,
        rerank_score=0.95,
        start_offset=120,
        end_offset=174,
    )
    retrieval = RetrievalOutcome(
        query=RetrievalQuery(
            original="它失败后怎么回滚?",
            rewritten="Kubernetes Deployment 滚动更新失败后如何回滚?",
            language="zh",
            filters=QueryFilters(document_types=["official_documentation"]),
            mode=RetrievalMode.HYBRID,
            top_k=8,
        ),
        candidates=[candidate],
        trace_candidates=[candidate],
        keyword_count=30,
        vector_count=30,
        reranked_count=8,
        timings=RetrievalTimings(keyword_ms=4, vector_ms=5, rerank_ms=6, total_ms=16),
    )
    no_answer = NoAnswerDecision(
        answerable=True,
        confidence=0.91,
        reason="retrieval_confident",
        retrieval_confidence=0.94,
        evidence_sufficiency_score=0.88,
        diagnostics={"top1_score": 0.95},
    )
    citation = Citation(
        number=1,
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        title="Kubernetes Deployments",
        section="Deployments > Rolling Back",
        url=candidate.canonical_url,
        quoted_text="kubectl rollout undo",
        document_type="official_documentation",
        score=0.95,
        start_offset=124,
        end_offset=145,
        valid=True,
        validation_score=0.97,
    )
    citation_report = CitationReport(
        citations=[citation],
        citation_marker_count=1,
        citation_precision=1,
        citation_recall=1,
        claim_coverage=1,
        citation_correctness=0.97,
        citation_completeness=1,
    )
    return ChatResult(
        request_id=REQUEST_ID,
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        original_query="它失败后怎么回滚?",
        rewritten_query="Kubernetes Deployment 滚动更新失败后如何回滚?",
        answer="可以运行 kubectl rollout undo。 [1]\n该引用来自官方文档。 [1]",
        answerable=True,
        confidence=0.91,
        citations=[citation],
        retrieval=retrieval,
        usage=ChatUsage(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            estimated_cost_usd=None,
        ),
        latency=ChatLatency(
            rewrite_ms=3,
            keyword_retrieval_ms=4,
            vector_retrieval_ms=5,
            rerank_ms=6,
            generation_ms=20,
            total_ms=40,
        ),
        no_answer=no_answer,
        citation_report=citation_report,
    )


def _test_app(service: FakeChatService) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    install_exception_handlers(app)
    app.include_router(router)

    async def override_service() -> FakeChatService:
        return service

    async def allow_api_key() -> None:
        return None

    app.dependency_overrides[get_chat_service] = override_service
    app.dependency_overrides[require_api_key] = allow_api_key
    return app


@pytest.mark.asyncio
async def test_chat_api_returns_grounded_contract_and_forwards_filters() -> None:
    service = FakeChatService(_chat_result())
    app = _test_app(service)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat",
            headers={"x-request-id": str(REQUEST_ID)},
            json={
                "query": "它失败后怎么回滚?",
                "conversation_id": str(CONVERSATION_ID),
                "filters": {"document_types": ["official_documentation"]},
                "mode": "hybrid",
                "top_k": 8,
                "debug": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == str(REQUEST_ID)
    assert response.headers["x-request-id"] == payload["request_id"]
    assert payload["rewritten_query"].startswith("Kubernetes Deployment")
    assert payload["answerable"] is True
    assert payload["citations"][0]["chunk_id"] == str(CHUNK_ID)
    assert payload["citations"][0]["url"].startswith("https://kubernetes.io/")
    assert payload["retrieval"] == {
        "keyword_count": 30,
        "vector_count": 30,
        "reranked_count": 8,
        "degraded_channels": [],
        "reranker_degraded": False,
    }
    assert payload["usage"]["estimated_cost_usd"] is None
    assert payload["debug"]["citation_metrics"]["claim_coverage"] == 1
    assert response.headers["x-request-id"]
    assert len(service.calls) == 1
    assert service.calls[0]["mode"] is RetrievalMode.HYBRID
    assert service.calls[0]["request_id"] == REQUEST_ID
    assert service.calls[0]["top_k"] == 8
    assert service.calls[0]["filters"].document_types == ["official_documentation"]


@pytest.mark.asyncio
async def test_chat_sse_emits_metadata_deltas_then_complete_validated_result() -> None:
    service = FakeChatService(_chat_result())
    app = _test_app(service)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat/stream",
            headers={"x-request-id": str(REQUEST_ID)},
            json={"query": "它失败后怎么回滚?"},
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == str(REQUEST_ID)
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    assert events[0]["event"] == "metadata"
    assert events[-1]["event"] == "done"
    assert events[1:-1]
    assert all(event["event"] == "delta" for event in events[1:-1])
    metadata = json.loads(events[0]["data"])
    assert metadata == {
        "request_id": str(REQUEST_ID),
        "conversation_id": str(CONVERSATION_ID),
        "message_id": str(MESSAGE_ID),
        "answerable": True,
    }
    streamed_answer = "".join(
        json.loads(event["data"])["text"] for event in events if event["event"] == "delta"
    )
    assert streamed_answer == service.result.answer
    done = json.loads(events[-1]["data"])
    assert done["answer"] == service.result.answer
    assert done["citations"][0]["chunk_id"] == str(CHUNK_ID)
    assert len(service.calls) == 1


@pytest.mark.asyncio
async def test_chat_rejects_whitespace_query_without_calling_service() -> None:
    service = FakeChatService(_chat_result())
    app = _test_app(service)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/chat", json={"query": "   "})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_FAILED"
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]
    assert service.calls == []


@pytest.mark.asyncio
async def test_chat_rejects_oversized_or_nested_filters_before_service_call() -> None:
    service = FakeChatService(_chat_result())
    transport = httpx.ASGITransport(app=_test_app(service))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        too_many = await client.post(
            "/api/v1/chat",
            json={"query": "Deployment", "filters": {"kinds": ["Pod"] * 51}},
        )
        nested = await client.post(
            "/api/v1/chat",
            json={"query": "Deployment", "filters": {"metadata": {"tenant": {"id": 1}}}},
        )

    assert too_many.status_code == 422
    assert nested.status_code == 422
    assert service.calls == []


@pytest.mark.asyncio
async def test_chat_normalizes_non_uuid_correlation_id_across_header_body_and_service() -> None:
    service = FakeChatService(_chat_result())
    transport = httpx.ASGITransport(app=_test_app(service))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat",
            headers={"x-request-id": "not-a-uuid"},
            json={"query": "Deployment rollback"},
        )

    normalized = UUID(response.headers["x-request-id"])
    assert response.json()["request_id"] == str(normalized)
    assert service.calls[0]["request_id"] == normalized


def _parse_sse(body: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    normalized = body.replace("\r\n", "\n")
    for block in normalized.strip().split("\n\n"):
        event: dict[str, str] = {}
        for line in block.splitlines():
            if ": " not in line:
                continue
            field, value = line.split(": ", 1)
            event[field] = value
        events.append(event)
    return events
