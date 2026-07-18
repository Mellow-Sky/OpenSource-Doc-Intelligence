"""Incremental loader for GitHub issues and optional issue comments."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import SecretStr

from app.domain.documents import RawDocument
from app.ingestion.loaders._github_api import GitHubAPIClient, parse_github_datetime
from app.ingestion.loaders.base import DocumentLoader


class GitHubIssuesLoader(DocumentLoader):
    """Load GitHub issues with server-side updated-at filtering and pagination."""

    def __init__(
        self,
        *,
        repository: str,
        token: str | SecretStr | None = None,
        include_pull_requests: bool = False,
        include_comments: bool = False,
        updated_since: datetime | None = None,
        state: str = "all",
        client: httpx.AsyncClient | None = None,
        api_base_url: str = "https://api.github.com",
        per_page: int = 100,
        max_pages: int = 1000,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        max_retry_delay_seconds: float = 60.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if state not in {"open", "closed", "all"}:
            msg = "Issue state must be one of: open, closed, all"
            raise ValueError(msg)
        if updated_since is not None and updated_since.tzinfo is None:
            updated_since = updated_since.replace(tzinfo=UTC)

        self.repository = repository
        self.include_pull_requests = include_pull_requests
        self.include_comments = include_comments
        self.updated_since = updated_since.astimezone(UTC) if updated_since else None
        self.state = state
        protected_token = SecretStr(token) if isinstance(token, str) else token
        self._api_options: dict[str, Any] = {
            "repository": repository,
            "token": protected_token,
            "client": client,
            "api_base_url": api_base_url,
            "per_page": per_page,
            "max_pages": max_pages,
            "max_retries": max_retries,
            "backoff_base_seconds": backoff_base_seconds,
            "max_retry_delay_seconds": max_retry_delay_seconds,
            "sleep": sleep,
            "clock": clock,
        }

    def __repr__(self) -> str:
        """Return non-sensitive loader configuration for diagnostics."""
        return (
            f"GitHubIssuesLoader(repository={self.repository!r}, "
            f"include_pull_requests={self.include_pull_requests!r}, "
            f"include_comments={self.include_comments!r})"
        )

    async def load(self) -> list[RawDocument]:
        """Load all issue pages and, when enabled, their comment pages."""
        api = GitHubAPIClient(**self._api_options)
        try:
            params: dict[str, str | int] = {
                "state": self.state,
                "sort": "updated",
                "direction": "asc",
            }
            if self.updated_since is not None:
                params["since"] = self.updated_since.isoformat().replace("+00:00", "Z")
            issues = await api.get_paginated(
                f"/repos/{self.repository}/issues",
                params=params,
            )
            documents: list[RawDocument] = []
            for issue in issues:
                if "pull_request" in issue and not self.include_pull_requests:
                    continue
                issue_updated_at = parse_github_datetime(issue.get("updated_at"))
                if (
                    self.updated_since is not None
                    and issue_updated_at is not None
                    and issue_updated_at < self.updated_since
                ):
                    continue
                comments: list[dict[str, Any]] = []
                if self.include_comments and _positive_int(issue.get("comments")) > 0:
                    number = _positive_int(issue.get("number"))
                    comments = await api.get_paginated(
                        f"/repos/{self.repository}/issues/{number}/comments"
                    )
                documents.append(self._to_raw_document(issue, comments))
            return documents
        finally:
            await api.aclose()

    def _to_raw_document(
        self,
        issue: dict[str, Any],
        comments: list[dict[str, Any]],
    ) -> RawDocument:
        number = _positive_int(issue.get("number"))
        title = _non_empty_string(issue.get("title")) or f"Issue #{number}"
        raw_body = issue.get("body")
        body = raw_body if isinstance(raw_body, str) else ""
        content_parts = [f"# {title}", "", body.strip()]
        if comments:
            content_parts.extend(["", "## Comments"])
            for comment in comments:
                user = comment.get("user")
                login = user.get("login") if isinstance(user, dict) else None
                author = login if isinstance(login, str) and login else "unknown"
                comment_created_at = _non_empty_string(comment.get("created_at")) or "unknown time"
                raw_comment_body = comment.get("body")
                comment_body = raw_comment_body if isinstance(raw_comment_body, str) else ""
                content_parts.extend(
                    ["", f"### @{author} — {comment_created_at}", "", comment_body.strip()]
                )

        updated_at = parse_github_datetime(issue.get("updated_at"))
        created_at = parse_github_datetime(issue.get("created_at"))
        closed_at = parse_github_datetime(issue.get("closed_at"))
        labels = _label_names(issue.get("labels"))
        user = issue.get("user")
        issue_author = user.get("login") if isinstance(user, dict) else None
        is_pull_request = "pull_request" in issue
        canonical_url = _non_empty_string(issue.get("html_url"))
        return RawDocument(
            source_type="github_issue",
            external_id=f"github-issue:{self.repository}:{number}",
            title=title,
            content="\n".join(content_parts).strip(),
            canonical_url=canonical_url,
            source_version=updated_at.isoformat() if updated_at else None,
            updated_at=updated_at,
            metadata={
                "repository": self.repository,
                "issue_number": number,
                "state": _non_empty_string(issue.get("state")) or "unknown",
                "labels": labels,
                "author": issue_author if isinstance(issue_author, str) else None,
                "locked": bool(issue.get("locked", False)),
                "is_pull_request": is_pull_request,
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
                "closed_at": closed_at.isoformat() if closed_at else None,
                "comments_count": len(comments)
                if self.include_comments
                else _positive_int(issue.get("comments")),
                "comments_included": self.include_comments,
            },
        )


def _positive_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    return 0


def _non_empty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _label_names(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    labels: list[str] = []
    for label in value:
        if isinstance(label, str) and label:
            labels.append(label)
        elif isinstance(label, dict):
            name = label.get("name")
            if isinstance(name, str) and name:
                labels.append(name)
    return labels
