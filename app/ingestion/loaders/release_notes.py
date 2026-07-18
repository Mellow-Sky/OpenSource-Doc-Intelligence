"""Loader for GitHub Releases and release-note Markdown tracked in Git."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from app.domain.documents import RawDocument
from app.ingestion.loaders._github_api import GitHubAPIClient, parse_github_datetime
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.github_repo import (
    DEFAULT_GIT_TIMEOUT_SECONDS,
    GitHubRepositoryLoader,
)

_DEFAULT_RELEASE_GLOBS = (
    "CHANGELOG.md",
    "CHANGELOG/**/*.md",
    "changelog.md",
    "changelogs/**/*.md",
    "releases/**/*.md",
    "docs/releases/**/*.md",
    "docs/release-notes/**/*.md",
)


class ReleaseNotesLoader(DocumentLoader):
    """Combine GitHub Releases API records with repository release documents."""

    def __init__(
        self,
        *,
        repository: str,
        branch: str = "master",
        token: str | SecretStr | None = None,
        include_github_releases: bool = True,
        include_repository_files: bool = False,
        include_drafts: bool = False,
        include_prereleases: bool = True,
        checkout_path: Path | None = None,
        clone_url: str | None = None,
        repository_include_globs: Sequence[str] = _DEFAULT_RELEASE_GLOBS,
        repository_exclude_globs: Sequence[str] = (),
        git_timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
        api_base_url: str = "https://api.github.com",
        per_page: int = 100,
        max_pages: int = 100,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        max_retry_delay_seconds: float = 60.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not include_github_releases and not include_repository_files:
            msg = "ReleaseNotesLoader requires at least one enabled source"
            raise ValueError(msg)
        if include_repository_files and checkout_path is None:
            msg = "checkout_path is required when repository release files are enabled"
            raise ValueError(msg)
        if git_timeout_seconds <= 0:
            msg = "Git command timeout must be positive"
            raise ValueError(msg)

        self.repository = repository
        self.branch = branch
        self.include_github_releases = include_github_releases
        self.include_repository_files = include_repository_files
        self.include_drafts = include_drafts
        self.include_prereleases = include_prereleases
        self.checkout_path = checkout_path
        self.clone_url = clone_url
        self.repository_include_globs = tuple(repository_include_globs)
        self.repository_exclude_globs = tuple(repository_exclude_globs)
        self.git_timeout_seconds = git_timeout_seconds
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
        """Return non-sensitive release source configuration."""
        return (
            f"ReleaseNotesLoader(repository={self.repository!r}, "
            f"include_github_releases={self.include_github_releases!r}, "
            f"include_repository_files={self.include_repository_files!r})"
        )

    async def load(self) -> list[RawDocument]:
        """Load enabled release sources and return one canonical collection."""
        documents: list[RawDocument] = []
        if self.include_github_releases:
            documents.extend(await self._load_github_releases())
        if self.include_repository_files:
            documents.extend(await self._load_repository_files())
        return documents

    async def _load_github_releases(self) -> list[RawDocument]:
        api = GitHubAPIClient(**self._api_options)
        try:
            releases = await api.get_paginated(f"/repos/{self.repository}/releases")
        finally:
            await api.aclose()

        documents: list[RawDocument] = []
        for release in releases:
            if bool(release.get("draft")) and not self.include_drafts:
                continue
            if bool(release.get("prerelease")) and not self.include_prereleases:
                continue
            documents.append(self._release_to_document(release))
        return documents

    async def _load_repository_files(self) -> list[RawDocument]:
        if self.checkout_path is None:
            return []
        loader = GitHubRepositoryLoader(
            repository=self.repository,
            branch=self.branch,
            checkout_path=self.checkout_path,
            clone_url=self.clone_url,
            include_globs=self.repository_include_globs,
            exclude_globs=self.repository_exclude_globs,
            git_timeout_seconds=self.git_timeout_seconds,
        )
        repository_documents = await loader.load()
        converted: list[RawDocument] = []
        for document in repository_documents:
            repository_path = document.metadata.get("repository_path")
            converted.append(
                document.model_copy(
                    update={
                        "source_type": "release_note",
                        "external_id": (f"github-release-file:{self.repository}:{repository_path}"),
                        "metadata": {
                            **document.metadata,
                            "origin": "repository_file",
                        },
                    }
                )
            )
        return converted

    def _release_to_document(self, release: dict[str, Any]) -> RawDocument:
        tag = _string(release.get("tag_name"))
        release_id = release.get("id")
        stable_id = str(release_id) if isinstance(release_id, int) else tag or "unknown"
        title = _string(release.get("name")) or tag or f"Release {stable_id}"
        body = _string(release.get("body")) or ""
        published_at = parse_github_datetime(release.get("published_at"))
        created_at = parse_github_datetime(release.get("created_at"))
        updated_at = parse_github_datetime(release.get("updated_at"))
        effective_time: datetime | None = published_at or updated_at or created_at
        author_data = release.get("author")
        author = author_data.get("login") if isinstance(author_data, dict) else None
        return RawDocument(
            source_type="release_note",
            external_id=f"github-release:{self.repository}:{stable_id}",
            title=title,
            content=f"# {title}\n\n{body.strip()}".strip(),
            canonical_url=_string(release.get("html_url")),
            source_version=tag,
            updated_at=effective_time,
            metadata={
                "origin": "github_releases_api",
                "repository": self.repository,
                "release_id": release_id if isinstance(release_id, int) else None,
                "tag_name": tag,
                "target_commitish": _string(release.get("target_commitish")),
                "draft": bool(release.get("draft")),
                "prerelease": bool(release.get("prerelease")),
                "author": author if isinstance(author, str) else None,
                "created_at": created_at.isoformat() if created_at else None,
                "published_at": published_at.isoformat() if published_at else None,
            },
        )


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
