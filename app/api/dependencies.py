"""Typed FastAPI dependencies backed by application state."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.container import AppContainer


def get_container(request: Request) -> AppContainer:
    """Return the process container stored during application lifespan."""
    return cast(AppContainer, request.app.state.container)


async def get_db_session(
    container: Annotated[AppContainer, Depends(get_container)],
) -> AsyncIterator[AsyncSession]:
    """Yield a rollback-safe request-scoped database session."""
    async for session in container.database.session():
        yield session
        return
    raise RuntimeError("Database session generator ended unexpectedly")
