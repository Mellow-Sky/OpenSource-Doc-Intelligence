"""Grounded chat, citation-safe SSE, and conversation history endpoints."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Annotated
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends
from fastapi import Request as FastAPIRequest
from sse_starlette.sse import EventSourceResponse

from app.api.dependencies import get_container
from app.container import AppContainer
from app.core.exceptions import ValidationError
from app.core.security import require_api_key
from app.repositories.conversation_repository import ConversationRepository
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationMessageResponse,
    ConversationResponse,
)
from app.services.chat_service import ChatService

router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
)
_STREAM_CHUNK = re.compile(
    r".+?(?:\n|[.!?。\uFF01\uFF1F](?:\s+|$)|$)",
    re.DOTALL,
)


def get_chat_service(
    container: Annotated[AppContainer, Depends(get_container)],
) -> ChatService:
    """Build a lightweight request orchestrator around process-scoped providers."""
    return ChatService(
        session_factory=container.database.session_factory,
        settings=container.settings,
        embedding_provider=container.embedding_provider,
        reranker_provider=container.reranker_provider,
        llm_provider=container.llm_provider,
        context_token_counter=container.context_token_counter,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: FastAPIRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    """Return one fully validated answer and its exact source citations."""
    _validate_query(request.query)
    request_id = _chat_request_id(http_request)
    result = await service.complete(
        request.query,
        request_id=request_id,
        conversation_id=request.conversation_id,
        filters=request.filters,
        mode=request.mode,
        top_k=request.top_k,
    )
    return ChatResponse.from_result(result, debug=request.debug)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: FastAPIRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> EventSourceResponse:
    """Stream a citation-validated result as SSE events.

    Generation is intentionally buffered until citation/no-answer validation completes,
    preventing an unsupported token stream from reaching clients before it can be rejected.
    """
    _validate_query(request.query)
    request_id = _chat_request_id(http_request)

    async def events() -> AsyncIterator[dict[str, str]]:
        result = await service.complete(
            request.query,
            request_id=request_id,
            conversation_id=request.conversation_id,
            filters=request.filters,
            mode=request.mode,
            top_k=request.top_k,
        )
        response = ChatResponse.from_result(result, debug=request.debug)
        yield {
            "event": "metadata",
            "data": json.dumps(
                {
                    "request_id": str(result.request_id),
                    "conversation_id": str(result.conversation_id),
                    "message_id": str(result.message_id),
                    "answerable": result.answerable,
                },
                ensure_ascii=False,
            ),
        }
        for match in _STREAM_CHUNK.finditer(result.answer):
            if match.group(0):
                yield {
                    "event": "delta",
                    "data": json.dumps({"text": match.group(0)}, ensure_ascii=False),
                }
        yield {
            "event": "done",
            "data": response.model_dump_json(exclude_none=True),
        }

    return EventSourceResponse(events())


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def conversation(
    conversation_id: UUID,
    container: Annotated[AppContainer, Depends(get_container)],
) -> ConversationResponse:
    """Return a complete bounded transcript without lazy-load N+1 queries."""
    async with container.database.session_factory() as session:
        record = await ConversationRepository(session).get(
            conversation_id,
            with_messages=True,
        )
    if record is None:
        raise ValidationError("Conversation does not exist")
    return ConversationResponse(
        id=record.id,
        user_id=record.user_id,
        title=record.title,
        created_at=record.created_at,
        updated_at=record.updated_at,
        messages=[
            ConversationMessageResponse(
                id=item.id,
                role=item.role,
                original_query=item.original_query,
                rewritten_query=item.rewritten_query,
                content=item.content,
                token_usage=item.token_usage,
                cost=_decimal_float(item.cost),
                latency_ms=item.latency_ms,
                created_at=item.created_at,
            )
            for item in record.messages
        ],
    )


def _validate_query(query: str) -> None:
    # ChatService's preprocessor enforces the configured dynamic maximum. This
    # early check keeps transport errors predictable without reading private config.
    if not query.strip():
        raise ValidationError("Query must not be blank")


def _chat_request_id(request: FastAPIRequest) -> UUID:
    """Use one UUID for HTTP headers, logs, response bodies, and usage rows."""

    supplied = str(getattr(request.state, "request_id", ""))
    try:
        request_id = UUID(supplied)
    except ValueError:
        request_id = uuid4()
    normalized = str(request_id)
    request.state.request_id = normalized
    structlog.contextvars.bind_contextvars(request_id=normalized)
    return request_id


def _decimal_float(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None
