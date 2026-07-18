"""Build source loaders from persisted, strictly bounded source configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from app.core.exceptions import ConfigurationError
from app.db.models.source_document import Source
from app.ingestion.loaders import (
    DEFAULT_GIT_TIMEOUT_SECONDS,
    DocumentLoader,
    GitHubIssuesLoader,
    GitHubRepositoryLoader,
    KubernetesAPIReferenceLoader,
    ReleaseNotesLoader,
)

_DEFAULT_REPOSITORY_INCLUDE = (
    "**/*.md",
    "**/*.markdown",
    "**/*.rst",
    "**/*.html",
    "**/*.htm",
    "**/*.yaml",
    "**/*.yml",
    "**/*.json",
)
_DEFAULT_RELEASE_INCLUDE = (
    "CHANGELOG.md",
    "CHANGELOG/**/*.md",
    "changelog.md",
    "changelogs/**/*.md",
    "releases/**/*.md",
    "docs/releases/**/*.md",
    "docs/release-notes/**/*.md",
)


@dataclass(frozen=True, slots=True)
class LoaderSpec:
    """A loader plus the snapshot semantics needed by incremental persistence."""

    loader: DocumentLoader
    complete_snapshot: bool
    cursor_type: str
    previous_cursor: str | None = None


def create_loader(
    source: Source,
    *,
    cache_root: Path,
    github_token: SecretStr | str | None = None,
    cursors: Mapping[str, str] | None = None,
) -> DocumentLoader:
    """Create the concrete loader configured by one persisted ``Source`` row."""

    return create_loader_spec(
        source,
        cache_root=cache_root,
        github_token=github_token,
        cursors=cursors,
    ).loader


def create_loader_spec(
    source: Source,
    *,
    cache_root: Path,
    github_token: SecretStr | str | None = None,
    cursors: Mapping[str, str] | None = None,
) -> LoaderSpec:
    """Create a loader without permitting repository checkouts outside ``cache_root``."""

    source_type = source.source_type.casefold().strip()
    config = dict(source.config or {})
    checkpoints = dict(cursors or {})
    repository = _required_repository(source)
    branch = source.branch or "master"
    loader: DocumentLoader

    if source_type in {"github_repository", "github_repo", "repository"}:
        loader = GitHubRepositoryLoader(
            repository=repository,
            branch=branch,
            checkout_path=_checkout_path(cache_root, source, config),
            clone_url=_optional_string(config, "clone_url"),
            include_globs=_string_sequence(
                config,
                "include",
                default=_DEFAULT_REPOSITORY_INCLUDE,
            ),
            exclude_globs=_string_sequence(config, "exclude", default=()),
            shallow=_boolean(config, "shallow", default=True),
            git_timeout_seconds=_positive_float(
                config,
                "git_timeout_seconds",
                default=DEFAULT_GIT_TIMEOUT_SECONDS,
            ),
        )
        return LoaderSpec(
            loader=loader,
            complete_snapshot=True,
            cursor_type="repository_commit_sha",
            previous_cursor=checkpoints.get("repository_commit_sha"),
        )

    if source_type in {"github_issues", "github_issue", "issues"}:
        cursor_type = "issues_updated_at"
        previous_cursor = checkpoints.get(cursor_type)
        updated_since = _parse_cursor_datetime(previous_cursor, cursor_type)
        loader = GitHubIssuesLoader(
            repository=repository,
            token=github_token,
            include_pull_requests=_boolean(config, "include_pull_requests", default=False),
            include_comments=_boolean(config, "include_comments", default=False),
            updated_since=updated_since,
            state=_optional_string(config, "state") or "all",
            api_base_url=_optional_string(config, "api_base_url") or "https://api.github.com",
            per_page=_integer(config, "per_page", default=100, minimum=1),
            max_pages=_integer(config, "max_pages", default=1000, minimum=1),
            max_retries=_integer(config, "max_retries", default=3, minimum=0),
        )
        return LoaderSpec(
            loader=loader,
            # A cursor-filtered issue scan is a delta and can never authorize deletes.
            complete_snapshot=previous_cursor is None,
            cursor_type=cursor_type,
            previous_cursor=previous_cursor,
        )

    if source_type in {"release_notes", "release_note", "releases"}:
        include_patterns = _release_patterns(config)
        include_repository_files = _boolean(
            config,
            "include_repository_files",
            default=bool(include_patterns),
        )
        if include_repository_files and not include_patterns:
            include_patterns = _DEFAULT_RELEASE_INCLUDE
        loader = ReleaseNotesLoader(
            repository=repository,
            branch=branch,
            token=github_token,
            include_github_releases=_boolean(config, "include_github_releases", default=True),
            include_repository_files=include_repository_files,
            include_drafts=_boolean(config, "include_drafts", default=False),
            include_prereleases=_boolean(config, "include_prereleases", default=True),
            checkout_path=(
                _checkout_path(cache_root, source, config) if include_repository_files else None
            ),
            clone_url=_optional_string(config, "clone_url"),
            repository_include_globs=include_patterns,
            repository_exclude_globs=_string_sequence(config, "exclude", default=()),
            api_base_url=_optional_string(config, "api_base_url") or "https://api.github.com",
            per_page=_integer(config, "per_page", default=100, minimum=1),
            max_pages=_integer(config, "max_pages", default=100, minimum=1),
            max_retries=_integer(config, "max_retries", default=3, minimum=0),
            git_timeout_seconds=_positive_float(
                config,
                "git_timeout_seconds",
                default=DEFAULT_GIT_TIMEOUT_SECONDS,
            ),
        )
        cursor_type = "releases_updated_at"
        return LoaderSpec(
            loader=loader,
            complete_snapshot=True,
            cursor_type=cursor_type,
            previous_cursor=checkpoints.get(cursor_type),
        )

    if source_type in {
        "kubernetes_api_reference",
        "kubernetes_api",
        "api_reference",
    }:
        urls = _string_sequence(config, "html_urls", default=())
        if not urls and source.base_url:
            urls = (str(source.base_url),)
        structured_data = config.get("structured_data")
        if not urls and structured_data is None:
            raise ConfigurationError(
                f"Source {source.name!r} requires base_url, html_urls, or structured_data"
            )
        loader = KubernetesAPIReferenceLoader(
            html_urls=urls,
            structured_data=structured_data,
            source_url=_optional_string(config, "source_url")
            or (str(source.base_url) if source.base_url else None),
            max_retries=_integer(config, "max_retries", default=3, minimum=0),
        )
        cursor_type = "api_snapshot_at"
        return LoaderSpec(
            loader=loader,
            complete_snapshot=True,
            cursor_type=cursor_type,
            previous_cursor=checkpoints.get(cursor_type),
        )

    raise ConfigurationError(f"Unsupported ingestion source type: {source.source_type}")


def _required_repository(source: Source) -> str:
    if source.repository:
        return source.repository
    if source.source_type.casefold() in {
        "kubernetes_api_reference",
        "kubernetes_api",
        "api_reference",
    }:
        # API-reference HTTP loading does not consume the repository value.
        return "kubernetes/website"
    raise ConfigurationError(f"Source {source.name!r} requires a repository")


def _checkout_path(cache_root: Path, source: Source, config: Mapping[str, Any]) -> Path:
    """Resolve a source checkout and reject absolute/traversing/symlink escapes."""

    root = cache_root.expanduser().resolve()
    raw_subdirectory = config.get("checkout_subdir", str(source.id))
    if not isinstance(raw_subdirectory, str) or not raw_subdirectory.strip():
        raise ConfigurationError("checkout_subdir must be a non-empty relative path")
    relative = Path(raw_subdirectory)
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigurationError("checkout_subdir must remain inside the ingestion cache")
    candidate = (root / relative).resolve()
    if candidate == root or root not in candidate.parents:
        raise ConfigurationError("repository checkout escaped the ingestion cache")
    return candidate


def _release_patterns(config: Mapping[str, Any]) -> tuple[str, ...]:
    if "repository_include_globs" in config:
        return _string_sequence(config, "repository_include_globs", default=())
    raw = config.get("changelog_glob")
    if raw is None:
        return ()
    if isinstance(raw, str) and raw.strip():
        return (raw,)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = tuple(str(item) for item in raw if isinstance(item, str) and item)
        if values:
            return values
    raise ConfigurationError("changelog_glob must be a string or non-empty string list")


def _string_sequence(
    config: Mapping[str, Any], key: str, *, default: tuple[str, ...]
) -> tuple[str, ...]:
    raw = config.get(key)
    if raw is None:
        return default
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ConfigurationError(f"{key} must be a list of strings")
    values = tuple(item for item in raw if isinstance(item, str) and item.strip())
    if len(values) != len(raw):
        raise ConfigurationError(f"{key} must contain only non-empty strings")
    return values


def _optional_string(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty string")
    return value


def _boolean(config: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{key} must be a boolean")
    return value


def _integer(config: Mapping[str, Any], key: str, *, default: int, minimum: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{key} must be an integer >= {minimum}")
    return int(value)


def _positive_float(
    config: Mapping[str, Any],
    key: str,
    *,
    default: float,
) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"{key} must be a positive number")
    return float(value)


def _parse_cursor_datetime(value: str | None, cursor_type: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigurationError(f"Invalid {cursor_type} cursor timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = ["LoaderSpec", "create_loader", "create_loader_spec"]
