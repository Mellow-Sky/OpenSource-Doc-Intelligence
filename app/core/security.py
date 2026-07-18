"""API-key authentication helpers for user and administrative routes."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError


def _matches(candidate: str | None, configured: SecretStr | None) -> bool:
    expected = configured.get_secret_value() if configured is not None else None
    return expected is None or (
        candidate is not None and secrets.compare_digest(candidate, expected)
    )


async def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Require the configured public API key when one is set."""
    if not _matches(x_api_key, settings.api_key):
        raise AuthenticationError("Invalid API key")


async def require_admin_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_admin_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Require an admin API key for ingestion and evaluation mutation routes."""
    configured = settings.admin_api_key or settings.api_key
    if not _matches(x_admin_api_key, configured):
        raise AuthenticationError("Invalid admin API key")
