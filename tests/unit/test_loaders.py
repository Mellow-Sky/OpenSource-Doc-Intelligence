"""Unit tests for source loaders without live network dependencies."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from app.ingestion.loaders import (
    DocumentLoader,
    GitHubIssuesLoader,
    GitHubRepositoryLoader,
    KubernetesAPIReferenceLoader,
    ReleaseNotesLoader,
)
from app.ingestion.loaders.base import LoaderError


@pytest.fixture
def inline_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute offloaded work inline so unit tests remain deterministic in restricted CI."""

    async def run_inline(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _create_repository(path: Path) -> Path:
    path.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch", "main", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(path, "config", "user.email", "loader-tests@example.invalid")
    _git(path, "config", "user.name", "Loader Tests")
    (path / "docs").mkdir()
    (path / "README.md").write_text("# Initial title\n\nRepository body.\n", encoding="utf-8")
    (path / "docs" / "guide.rst").write_text("Guide\n=====\n\nRST body.\n", encoding="utf-8")
    (path / "docs" / "config.yaml").write_text("title: API Settings\nenabled: true\n")
    (path / "docs" / "private.md").write_text("# Private\n")
    (path / "ignored.py").write_text("print('not documentation')\n")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "initial docs")
    return path


def test_document_loader_is_an_abstract_async_contract() -> None:
    with pytest.raises(TypeError):
        DocumentLoader()


@pytest.mark.asyncio
async def test_repository_loader_clones_filters_and_pulls_updates(
    tmp_path: Path,
    inline_to_thread: None,
) -> None:
    source = _create_repository(tmp_path / "source")
    checkout = tmp_path / "checkout"
    loader = GitHubRepositoryLoader(
        repository="example/docs",
        branch="main",
        checkout_path=checkout,
        clone_url=str(source),
        exclude_globs=("docs/private.md",),
    )

    first_snapshot = await loader.load()

    assert {document.metadata["repository_path"] for document in first_snapshot} == {
        "README.md",
        "docs/config.yaml",
        "docs/guide.rst",
    }
    by_path = {document.metadata["repository_path"]: document for document in first_snapshot}
    assert by_path["README.md"].title == "Initial title"
    assert by_path["docs/guide.rst"].title == "Guide"
    assert by_path["docs/config.yaml"].title == "API Settings"
    first_commit = _git(source, "rev-parse", "HEAD")
    assert all(document.source_version == first_commit for document in first_snapshot)
    assert by_path["README.md"].metadata["commit_sha"] == first_commit
    assert by_path["README.md"].metadata["raw_url"].endswith(f"/{first_commit}/README.md")
    readme_commit_time = _git(source, "log", "-1", "--format=%cI", "--", "README.md")
    assert by_path["README.md"].updated_at == datetime.fromisoformat(readme_commit_time)
    assert datetime.fromisoformat(
        by_path["README.md"].metadata["last_commit_at"]
    ) == datetime.fromisoformat(readme_commit_time)
    assert by_path["README.md"].metadata["last_commit_at"].endswith("+00:00")
    assert by_path["README.md"].metadata["last_commit_at_fallback"] is False

    (source / "README.md").write_text("# Updated title\n\nSecond revision.\n", encoding="utf-8")
    _git(source, "add", "README.md")
    _git(source, "commit", "-m", "update readme")

    second_snapshot = await loader.load()

    updated = next(
        document
        for document in second_snapshot
        if document.metadata["repository_path"] == "README.md"
    )
    assert updated.title == "Updated title"
    assert updated.source_version == _git(source, "rev-parse", "HEAD")
    assert updated.source_version != first_commit
    updated_commit_time = _git(source, "log", "-1", "--format=%cI", "--", "README.md")
    guide_commit_time = _git(source, "log", "-1", "--format=%cI", "--", "docs/guide.rst")
    by_second_path = {
        document.metadata["repository_path"]: document for document in second_snapshot
    }
    assert datetime.fromisoformat(updated.metadata["last_commit_at"]) == datetime.fromisoformat(
        updated_commit_time
    )
    assert datetime.fromisoformat(
        by_second_path["docs/guide.rst"].metadata["last_commit_at"]
    ) == datetime.fromisoformat(guide_commit_time)


@pytest.mark.asyncio
async def test_repository_loader_uses_noninteractive_bounded_git_commands(
    tmp_path: Path,
    inline_to_thread: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def timed_out(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **kwargs})
        raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timed_out)
    loader = GitHubRepositoryLoader(
        repository="example/docs",
        branch="main",
        checkout_path=tmp_path / "checkout",
        git_timeout_seconds=0.25,
    )

    with pytest.raises(LoaderError, match=r"Git clone timed out after 0\.25 seconds"):
        await loader.load()

    assert calls[0]["timeout"] == 0.25
    assert calls[0]["stdin"] is subprocess.DEVNULL
    assert calls[0]["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_repository_loader_rejects_non_positive_git_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        GitHubRepositoryLoader(
            repository="example/docs",
            branch="main",
            checkout_path=tmp_path / "checkout",
            git_timeout_seconds=0,
        )


@pytest.mark.asyncio
@respx.mock
async def test_issue_loader_paginates_filters_prs_and_includes_comments() -> None:
    issues_url = "https://api.github.com/repos/kubernetes/kubernetes/issues"
    comments_url = "https://api.github.com/repos/kubernetes/kubernetes/issues/10/comments"
    issues_route = respx.get(issues_url).mock(
        side_effect=[
            httpx.Response(
                200,
                json=[
                    {
                        "number": 10,
                        "title": "Kubelet restart behavior",
                        "body": "The kubelet restarts the pod.",
                        "html_url": "https://github.com/kubernetes/kubernetes/issues/10",
                        "state": "open",
                        "labels": [{"name": "kind/bug"}],
                        "user": {"login": "alice"},
                        "comments": 1,
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-03T00:00:00Z",
                    },
                    {
                        "number": 11,
                        "title": "A pull request",
                        "body": "Not an issue.",
                        "pull_request": {"url": "https://api.github.com/pulls/11"},
                        "comments": 0,
                        "updated_at": "2026-01-04T00:00:00Z",
                    },
                ],
            ),
            httpx.Response(
                200,
                json=[
                    {
                        "number": 12,
                        "title": "Unchanged issue",
                        "body": "Equal to the cursor.",
                        "comments": 0,
                        "updated_at": "2026-01-02T00:00:00Z",
                    }
                ],
            ),
        ]
    )
    comments_route = respx.get(comments_url).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "body": "Confirmed on v1.34.",
                    "created_at": "2026-01-03T01:00:00Z",
                    "user": {"login": "bob"},
                }
            ],
        )
    )
    loader = GitHubIssuesLoader(
        repository="kubernetes/kubernetes",
        token="top-secret-token",
        include_comments=True,
        updated_since=datetime(2026, 1, 2, tzinfo=UTC),
        per_page=2,
    )

    documents = await loader.load()

    # The exact cursor boundary is deliberately replayed; idempotent persistence
    # prevents duplicates and avoids losing issues sharing one-second timestamps.
    assert len(documents) == 2
    document = documents[0]
    assert document.external_id == "github-issue:kubernetes/kubernetes:10"
    assert "Confirmed on v1.34." in document.content
    assert document.metadata["labels"] == ["kind/bug"]
    assert document.metadata["comments_included"] is True
    assert len(issues_route.calls) == 2
    assert issues_route.calls[0].request.url.params["since"] == "2026-01-02T00:00:00Z"
    assert issues_route.calls[0].request.headers["Authorization"] == "Bearer top-secret-token"
    assert len(comments_route.calls) == 1
    assert "top-secret-token" not in repr(loader)
    assert documents[1].external_id == "github-issue:kubernetes/kubernetes:12"


@pytest.mark.asyncio
@respx.mock
async def test_issue_loader_honors_retry_after_without_network_sleep() -> None:
    endpoint = "https://api.github.com/repos/kubernetes/kubernetes/issues"
    route = respx.get(endpoint).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(200, json=[]),
        ]
    )
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    loader = GitHubIssuesLoader(
        repository="kubernetes/kubernetes",
        max_retries=1,
        sleep=record_sleep,
    )

    assert await loader.load() == []
    assert delays == [2.0]
    assert len(route.calls) == 2


@pytest.mark.asyncio
@respx.mock
async def test_issue_loader_retries_secondary_rate_limit() -> None:
    endpoint = "https://api.github.com/repos/kubernetes/kubernetes/issues"
    route = respx.get(endpoint).mock(
        side_effect=[
            httpx.Response(403, json={"message": "secondary rate limit"}),
            httpx.Response(200, json=[]),
        ]
    )
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    loader = GitHubIssuesLoader(
        repository="kubernetes/kubernetes",
        max_retries=1,
        backoff_base_seconds=0.1,
        sleep=record_sleep,
    )

    assert await loader.load() == []
    assert delays == [0.1]
    assert len(route.calls) == 2


@pytest.mark.asyncio
@respx.mock
async def test_api_html_without_data_attributes_splits_kinds_stably(
    inline_to_thread: None,
) -> None:
    url = "https://kubernetes.io/docs/reference/kubernetes-api/"
    html = """
    <main>
      <h2 id="deployment-v1-apps">Deployment v1 apps</h2>
      <table><tr><th>Field</th><th>Description</th></tr>
      <tr><td>spec.replicas</td><td>Desired Pods.</td></tr></table>
      <h2 id="service-v1-core">Service v1 core</h2>
      <table><tr><th>Field</th><th>Description</th></tr>
      <tr><td>spec.clusterIP</td><td>Cluster IP address.</td></tr></table>
    </main>
    """
    respx.get(url).mock(return_value=httpx.Response(200, text=html))

    documents = await KubernetesAPIReferenceLoader(html_urls=[url]).load()

    assert len(documents) == 2
    by_kind = {document.metadata["kind"]: document for document in documents}
    assert by_kind["Deployment"].metadata["field_paths"] == ["spec.replicas"]
    assert by_kind["Service"].metadata["field_paths"] == ["spec.clusterIP"]
    assert "clusterIP" not in by_kind["Deployment"].content
    assert by_kind["Deployment"].canonical_url == f"{url}#deployment-v1-apps"


@pytest.mark.asyncio
@respx.mock
async def test_release_loader_combines_api_and_repository_markdown(
    tmp_path: Path,
    inline_to_thread: None,
) -> None:
    source = _create_repository(tmp_path / "release-source")
    (source / "CHANGELOG.md").write_text("# Version 1.2.3\n\nStable release.\n", encoding="utf-8")
    _git(source, "add", "CHANGELOG.md")
    _git(source, "commit", "-m", "add changelog")
    releases_url = "https://api.github.com/repos/example/docs/releases"
    respx.get(releases_url).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": 123,
                    "tag_name": "v1.2.3",
                    "name": "Kubernetes-style 1.2.3",
                    "body": "Release API body.",
                    "html_url": "https://github.com/example/docs/releases/tag/v1.2.3",
                    "target_commitish": "main",
                    "draft": False,
                    "prerelease": False,
                    "author": {"login": "release-bot"},
                    "created_at": "2026-02-01T00:00:00Z",
                    "published_at": "2026-02-02T00:00:00Z",
                }
            ],
        )
    )
    loader = ReleaseNotesLoader(
        repository="example/docs",
        branch="main",
        include_repository_files=True,
        checkout_path=tmp_path / "release-checkout",
        clone_url=str(source),
    )

    documents = await loader.load()

    assert len(documents) == 2
    by_origin = {document.metadata["origin"]: document for document in documents}
    api_release = by_origin["github_releases_api"]
    repository_release = by_origin["repository_file"]
    assert api_release.source_version == "v1.2.3"
    assert api_release.updated_at == datetime(2026, 2, 2, tzinfo=UTC)
    assert repository_release.metadata["repository_path"] == "CHANGELOG.md"
    assert repository_release.source_type == "release_note"


@pytest.mark.asyncio
async def test_kubernetes_openapi_loader_keeps_gvk_and_nested_field_paths(
    inline_to_thread: None,
) -> None:
    openapi = {
        "definitions": {
            "io.k8s.api.apps.v1.DeploymentSpec": {
                "type": "object",
                "properties": {
                    "replicas": {
                        "type": "integer",
                        "description": "Desired number of replicas.",
                    }
                },
            },
            "io.k8s.api.apps.v1.Deployment": {
                "description": "Deployment enables declarative updates.",
                "x-kubernetes-group-version-kind": [
                    {"group": "apps", "version": "v1", "kind": "Deployment"}
                ],
                "properties": {
                    "apiVersion": {"type": "string"},
                    "kind": {"type": "string"},
                    "spec": {"$ref": "#/definitions/io.k8s.api.apps.v1.DeploymentSpec"},
                },
            },
        }
    }
    loader = KubernetesAPIReferenceLoader(
        structured_data=openapi,
        source_url="https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.34/",
    )

    documents = await loader.load()

    deployment = next(
        document for document in documents if document.metadata["kind"] == "Deployment"
    )
    assert deployment.metadata["api_group"] == "apps"
    assert deployment.metadata["version"] == "v1"
    assert "spec.replicas" in deployment.metadata["field_paths"]
    assert "Desired number of replicas." in deployment.content


@pytest.mark.asyncio
async def test_kubernetes_openapi_v3_components_are_supported(
    inline_to_thread: None,
) -> None:
    payload = {
        "openapi": "3.0.0",
        "components": {
            "schemas": {
                "io.k8s.api.core.v1.Service": {
                    "x-kubernetes-group-version-kind": [
                        {"group": "", "version": "v1", "kind": "Service"}
                    ],
                    "properties": {
                        "spec": {
                            "type": "object",
                            "properties": {
                                "clusterIP": {
                                    "type": "string",
                                    "description": "Cluster IP address.",
                                }
                            },
                        }
                    },
                }
            }
        },
    }
    loader = KubernetesAPIReferenceLoader(
        structured_data=payload,
        source_url="https://kubernetes.io/openapi/v3",
    )

    documents = await loader.load()

    assert len(documents) == 1
    assert documents[0].metadata["kind"] == "Service"
    assert "spec.clusterIP" in documents[0].metadata["field_paths"]


@pytest.mark.asyncio
@respx.mock
async def test_kubernetes_html_loader_preserves_field_locations_and_source_anchor(
    inline_to_thread: None,
) -> None:
    url = "https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.34/"
    html = """
    <html><head><title>Kubernetes API Reference</title></head><body>
      <article id="job-v1" data-api-group="batch" data-api-version="v1" data-kind="Job">
        <h2>Job v1 batch</h2>
        <p>Job represents a finite task.</p>
        <table>
          <tr><th>Field</th><th>Type</th><th>Description</th></tr>
          <tr data-field-path="spec.parallelism">
            <td><code>parallelism</code></td><td>integer</td><td>Maximum desired pods.</td>
          </tr>
        </table>
      </article>
    </body></html>
    """
    respx.get(url).mock(
        return_value=httpx.Response(
            200,
            text=html,
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "Last-Modified": "Wed, 01 Jul 2026 10:00:00 GMT",
            },
        )
    )
    loader = KubernetesAPIReferenceLoader(html_urls=[url])

    documents = await loader.load()

    assert len(documents) == 1
    document = documents[0]
    assert document.metadata["api_group"] == "batch"
    assert document.metadata["version"] == "v1"
    assert document.metadata["kind"] == "Job"
    assert document.metadata["field_paths"] == ["spec.parallelism"]
    assert document.metadata["fields"][0]["type"] == "integer"
    assert document.canonical_url == f"{url}#job-v1"
    assert document.updated_at == datetime(2026, 7, 1, 10, tzinfo=UTC)
