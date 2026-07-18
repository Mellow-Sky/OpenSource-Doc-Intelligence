"""Transaction-scoped persistence for conversations and generated messages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import Select, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.conversation import Conversation, Message

MAX_MESSAGE_PAGE_SIZE = 500
MAX_CONVERSATION_PAGE_SIZE = 200


@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationCreate:
    """Fields required to start a conversation."""

    user_id: str | None = None
    title: str | None = None
    id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageCreate:
    """Complete persisted message payload, including usage and request timings."""

    conversation_id: UUID
    role: str
    content: str
    original_query: str | None = None
    rewritten_query: str | None = None
    token_usage: dict[str, Any] = field(default_factory=dict)
    cost: Decimal | None = None
    latency_ms: int | None = None
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("role must not be blank")
        if not self.content:
            raise ValueError("content must not be empty")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if self.cost is not None and self.cost < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True, slots=True)
class ConversationPage:
    """A bounded conversation page and the matching total count."""

    items: list[Conversation]
    total: int
    limit: int
    offset: int


class ConversationRepository:
    """Persist a chat transcript without owning commit or rollback boundaries."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        record: ConversationCreate,
        *,
        created_at: datetime | None = None,
    ) -> Conversation:
        """Create a conversation and return its ORM row without committing."""
        timestamp = _utc(created_at)
        statement = (
            insert(Conversation)
            .values(
                id=record.id,
                user_id=record.user_id,
                title=record.title,
                created_at=timestamp,
                updated_at=timestamp,
            )
            .returning(Conversation)
        )
        return (await self._session.scalars(statement)).one()

    async def get(
        self,
        conversation_id: UUID,
        *,
        with_messages: bool = False,
    ) -> Conversation | None:
        """Return one conversation, optionally preloading its messages in fixed queries."""
        statement: Select[tuple[Conversation]] = select(Conversation).where(
            Conversation.id == conversation_id
        )
        if with_messages:
            statement = statement.options(selectinload(Conversation.messages))
        conversation = cast(Conversation | None, await self._session.scalar(statement))
        if conversation is not None and with_messages:
            conversation.messages.sort(key=lambda item: (item.created_at, item.id))
        return conversation

    async def list_page(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ConversationPage:
        """List conversations newest-first with parameterized user filtering."""
        _validate_page(limit, offset, maximum=MAX_CONVERSATION_PAGE_SIZE)
        predicates = [] if user_id is None else [Conversation.user_id == user_id]
        total = int(
            await self._session.scalar(select(func.count(Conversation.id)).where(*predicates)) or 0
        )
        rows = await self._session.scalars(
            select(Conversation)
            .where(*predicates)
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return ConversationPage(list(rows), total, limit, offset)

    async def add_message(
        self,
        record: MessageCreate,
        *,
        created_at: datetime | None = None,
    ) -> Message:
        """Persist one message and advance the parent conversation timestamp."""
        messages = await self.add_messages([record], created_at=created_at)
        return messages[0]

    async def add_messages(
        self,
        records: Sequence[MessageCreate],
        *,
        created_at: datetime | None = None,
    ) -> list[Message]:
        """Insert a message batch and touch all parents without per-message writes."""
        if not records:
            return []
        timestamp = _utc(created_at)
        statement = (
            insert(Message)
            .values(
                [
                    {
                        "id": record.id,
                        "conversation_id": record.conversation_id,
                        "role": record.role,
                        "original_query": record.original_query,
                        "rewritten_query": record.rewritten_query,
                        "content": record.content,
                        "token_usage": dict(record.token_usage),
                        "cost": record.cost,
                        "latency_ms": record.latency_ms,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    }
                    for record in records
                ]
            )
            .returning(Message)
        )
        result = list(await self._session.scalars(statement))
        conversation_ids = {record.conversation_id for record in records}
        await self._session.execute(
            update(Conversation)
            .where(Conversation.id.in_(conversation_ids))
            .values(updated_at=timestamp)
        )
        return result

    async def list_messages(
        self,
        conversation_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Message]:
        """Read one transcript page in deterministic chronological order."""
        _validate_page(limit, offset, maximum=MAX_MESSAGE_PAGE_SIZE)
        rows = await self._session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at, Message.id)
            .limit(limit)
            .offset(offset)
        )
        return list(rows)

    async def list_recent_messages(
        self,
        conversation_id: UUID,
        *,
        limit: int = 20,
    ) -> list[Message]:
        """Read the most recent bounded history and return it chronologically."""
        _validate_page(limit, 0, maximum=MAX_MESSAGE_PAGE_SIZE)
        rows = list(
            await self._session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(limit)
            )
        )
        rows.reverse()
        return rows


def _utc(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return timestamp.astimezone(UTC)


def _validate_page(limit: int, offset: int, *, maximum: int) -> None:
    if not 1 <= limit <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    if offset < 0:
        raise ValueError("offset must be non-negative")
