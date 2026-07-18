"""Data-source loaders exposed by the ingestion package."""

from app.ingestion.loaders.base import DocumentLoader, LoaderError
from app.ingestion.loaders.github_issues import GitHubIssuesLoader
from app.ingestion.loaders.github_repo import (
    DEFAULT_GIT_TIMEOUT_SECONDS,
    GitHubRepositoryLoader,
)
from app.ingestion.loaders.kubernetes_api import KubernetesAPIReferenceLoader
from app.ingestion.loaders.release_notes import ReleaseNotesLoader

__all__ = [
    "DEFAULT_GIT_TIMEOUT_SECONDS",
    "DocumentLoader",
    "GitHubIssuesLoader",
    "GitHubRepositoryLoader",
    "KubernetesAPIReferenceLoader",
    "LoaderError",
    "ReleaseNotesLoader",
]
