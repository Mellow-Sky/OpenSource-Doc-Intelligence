"""Loader for documentation tracked in a GitHub-compatible Git repository."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from app.domain.documents import RawDocument
from app.ingestion.loaders.base import DocumentLoader, LoaderError

_DEFAULT_INCLUDE_GLOBS = (
    "**/*.md",
    "**/*.markdown",
    "**/*.rst",
    "**/*.html",
    "**/*.htm",
    "**/*.yaml",
    "**/*.yml",
    "**/*.json",
)
_SUPPORTED_SUFFIXES = frozenset(
    {".md", ".markdown", ".rst", ".html", ".htm", ".yaml", ".yml", ".json"}
)
_CREDENTIALS_PATTERN = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)
DEFAULT_GIT_TIMEOUT_SECONDS = 300.0


class GitHubRepositoryLoader(DocumentLoader):
    """Clone or refresh a repository and load selected documentation files."""

    def __init__(
        self,
        *,
        repository: str,
        branch: str,
        checkout_path: Path,
        clone_url: str | None = None,
        include_globs: Sequence[str] = _DEFAULT_INCLUDE_GLOBS,
        exclude_globs: Sequence[str] = (),
        shallow: bool = True,
        git_timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
        github_web_url: str = "https://github.com",
        github_raw_url: str = "https://raw.githubusercontent.com",
    ) -> None:
        if repository.count("/") != 1:
            msg = "GitHub repository must use owner/name form"
            raise ValueError(msg)
        if not branch.strip():
            msg = "Repository branch cannot be empty"
            raise ValueError(msg)
        if not include_globs:
            msg = "At least one repository include glob is required"
            raise ValueError(msg)
        if git_timeout_seconds <= 0:
            msg = "Git command timeout must be positive"
            raise ValueError(msg)

        self.repository = repository
        self.branch = branch
        self.checkout_path = checkout_path
        self.clone_url = clone_url or f"https://github.com/{repository}.git"
        self.include_globs = tuple(include_globs)
        self.exclude_globs = tuple(exclude_globs)
        self.shallow = shallow
        self.git_timeout_seconds = git_timeout_seconds
        self.github_web_url = github_web_url.rstrip("/")
        self.github_raw_url = github_raw_url.rstrip("/")

    async def load(self) -> list[RawDocument]:
        """Refresh the checkout without blocking the event loop and load tracked docs."""
        await asyncio.to_thread(self._sync_repository)
        return await asyncio.to_thread(self._read_documents)

    def _sync_repository(self) -> None:
        checkout = self.checkout_path
        if (checkout / ".git").is_dir():
            fetch_args = ["fetch", "origin", self.branch]
            pull_args = ["pull", "--ff-only", "origin", self.branch]
            self._run_git(fetch_args, cwd=checkout)
            self._run_git(["checkout", self.branch], cwd=checkout)
            self._run_git(pull_args, cwd=checkout)
            return

        if checkout.exists() and any(checkout.iterdir()):
            msg = f"Repository checkout path is non-empty and is not a Git repository: {checkout}"
            raise LoaderError(msg)
        checkout.parent.mkdir(parents=True, exist_ok=True)
        clone_args = ["clone", "--branch", self.branch]
        if self.shallow:
            clone_args.extend(["--depth", "1"])
        clone_args.extend([self.clone_url, str(checkout)])
        self._run_git(clone_args)

    def _read_documents(self) -> list[RawDocument]:
        commit_sha = self._run_git(["rev-parse", "HEAD"], cwd=self.checkout_path).strip()
        tracked_output = self._run_git(["ls-files", "-z"], cwd=self.checkout_path)
        paths = sorted(path for path in tracked_output.split("\0") if path)
        included_paths = [path for path in paths if self._should_include(path)]
        modified_by_path, fallback_modified_at = self._last_commit_times(
            included_paths,
            commit_sha,
        )
        documents: list[RawDocument] = []

        for relative_path in included_paths:
            file_path = self.checkout_path / relative_path
            if not file_path.is_file() or file_path.is_symlink():
                continue
            content = file_path.read_text(encoding="utf-8", errors="replace")
            modified_at = modified_by_path.get(relative_path, fallback_modified_at)
            escaped_path = quote(relative_path, safe="/")
            escaped_ref = quote(commit_sha, safe="")
            raw_url = f"{self.github_raw_url}/{self.repository}/{escaped_ref}/{escaped_path}"
            canonical_url = (
                f"{self.github_web_url}/{self.repository}/blob/{escaped_ref}/{escaped_path}"
            )
            documents.append(
                RawDocument(
                    source_type="github_repository",
                    external_id=f"github:{self.repository}:{relative_path}",
                    title=_extract_title(relative_path, content),
                    content=content,
                    canonical_url=canonical_url,
                    source_version=commit_sha,
                    updated_at=modified_at,
                    metadata={
                        "repository": self.repository,
                        "repository_path": relative_path,
                        "branch": self.branch,
                        "commit_sha": commit_sha,
                        "last_commit_at": modified_at.isoformat(),
                        "last_commit_at_fallback": relative_path not in modified_by_path,
                        "raw_url": raw_url,
                        "format": file_path.suffix.lower().lstrip("."),
                    },
                )
            )
        return documents

    def _last_commit_times(
        self,
        tracked_paths: Sequence[str],
        commit_sha: str,
    ) -> tuple[dict[str, datetime], datetime]:
        """Build a path-to-commit-time map with one history traversal.

        A depth-one checkout cannot expose commits older than its shallow boundary.
        Those paths use the immutable checkout commit time and are explicitly marked
        as fallback metadata rather than using machine-local filesystem mtimes.
        """

        head_time = _parse_git_time(
            self._run_git(["show", "-s", "--format=%cI", commit_sha], cwd=self.checkout_path)
        )
        if not tracked_paths:
            return {}, head_time
        wanted = set(tracked_paths)
        is_shallow = (
            self._run_git(
                ["rev-parse", "--is-shallow-repository"],
                cwd=self.checkout_path,
            ).strip()
            == "true"
        )
        shallow_boundaries = (
            set(
                self._run_git(
                    ["rev-list", "--max-parents=0", commit_sha],
                    cwd=self.checkout_path,
                ).splitlines()
            )
            if is_shallow
            else set()
        )
        history = self._run_git(
            [
                "log",
                "--format=%x1e%H%x1f%cI%x00",
                "--name-only",
                "-z",
                "--no-renames",
                commit_sha,
            ],
            cwd=self.checkout_path,
        )
        current_time: datetime | None = None
        current_is_boundary = False
        resolved: dict[str, datetime] = {}
        for raw_token in history.split("\0"):
            token = raw_token.strip("\n")
            if not token:
                continue
            if token.startswith("\x1e"):
                try:
                    commit, timestamp = token[1:].split("\x1f", maxsplit=1)
                except ValueError as exc:
                    raise LoaderError("Git returned invalid history metadata") from exc
                current_time = _parse_git_time(timestamp)
                current_is_boundary = commit in shallow_boundaries
                continue
            if (
                current_time is not None
                and not current_is_boundary
                and token in wanted
                and token not in resolved
            ):
                resolved[token] = current_time
        return resolved, head_time

    def _should_include(self, relative_path: str) -> bool:
        if Path(relative_path).suffix.lower() not in _SUPPORTED_SUFFIXES:
            return False
        if not _matches_any_glob(relative_path, self.include_globs):
            return False
        return not _matches_any_glob(relative_path, self.exclude_globs)

    def _run_git(self, arguments: Sequence[str], *, cwd: Path | None = None) -> str:
        operation = arguments[0] if arguments else "command"
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=cwd,
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.git_timeout_seconds,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
        except subprocess.TimeoutExpired as exc:
            msg = f"Git {operation} timed out after {self.git_timeout_seconds:g} seconds"
            raise LoaderError(msg) from exc
        except OSError as exc:
            msg = "Unable to execute Git while loading repository"
            raise LoaderError(msg) from exc
        if completed.returncode != 0:
            detail = _CREDENTIALS_PATTERN.sub(r"\1***@", completed.stderr.strip())
            msg = f"Git {operation} failed"
            if detail:
                msg = f"{msg}: {detail}"
            raise LoaderError(msg)
        return completed.stdout


def _matches_any_glob(relative_path: str, patterns: Sequence[str]) -> bool:
    path = PurePosixPath(relative_path)
    for pattern in patterns:
        if path.match(pattern):
            return True
        if pattern.startswith("**/") and path.match(pattern[3:]):
            return True
    return False


def _extract_title(relative_path: str, content: str) -> str:
    suffix = Path(relative_path).suffix.lower()
    lines = content.splitlines()
    if suffix in {".md", ".markdown"}:
        for line in lines:
            match = re.match(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$", line)
            if match:
                return match.group(1).strip()
    elif suffix == ".rst":
        for index in range(len(lines) - 1):
            heading = lines[index].strip()
            underline = lines[index + 1].strip()
            if (
                heading
                and len(underline) >= len(heading)
                and len(set(underline)) == 1
                and underline[0] in "=-~^*+#"
            ):
                return heading
    elif suffix in {".html", ".htm"}:
        match = re.search(r"<title[^>]*>(.*?)</title>", content, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    elif suffix in {".yaml", ".yml"}:
        match = re.search(r"(?mi)^\s*title\s*:\s*['\"]?(.+?)['\"]?\s*$", content)
        if match:
            return match.group(1).strip()
    return Path(relative_path).stem.replace("-", " ").replace("_", " ").strip() or relative_path


def _parse_git_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise LoaderError("Git returned an invalid commit timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LoaderError("Git returned a timezone-naive commit timestamp")
    return parsed.astimezone(UTC)
