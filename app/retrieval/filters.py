"""Pure query-filter extraction for Kubernetes-oriented retrieval.

The extractor is intentionally conservative: a filter is emitted only when the
query contains an explicit qualifier or a well-known Kubernetes API object.  It
does not remove filter phrases from the query, so identifiers remain available
to both full-text and vector retrieval.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.domain.retrieval import QueryFilters

_KUBERNETES_VERSION = re.compile(
    r"(?<![\w./-])(?:kubernetes|k8s)\s*(?:version|版本)?\s*[:=]?\s*"
    r"v?(?P<version>\d+\.\d+(?:\.\d+)?)(?![\w.-])",
    re.IGNORECASE,
)
_BARE_KUBERNETES_VERSION = re.compile(
    r"(?<![\w./-])v(?P<version>\d+\.\d+(?:\.\d+)?)(?![\w.-])",
    re.IGNORECASE,
)
_QUALIFIED_KUBERNETES_VERSION = re.compile(
    r"(?:\bversion\b|版本)\s*[:=]?\s*v?(?P<version>\d+\.\d+(?:\.\d+)?)"
    r"|(?<![\w./-])v?(?P<suffix_version>\d+\.\d+(?:\.\d+)?)\s*版本",
    re.IGNORECASE,
)
_RELEASE_VERSION = re.compile(
    r"(?:release(?:\s+(?:notes?|version))?|changelog|发行(?:版本|说明)?|发布(?:版本|说明)?)"
    r"\s*[:=#]?\s*v?(?P<version>\d+\.\d+(?:\.\d+)?)(?![\w.-])",
    re.IGNORECASE,
)
_RELEASE_CONTEXT = re.compile(
    r"(?:release(?:\s+(?:notes?|version))?|changelog|发行(?:版本|说明)?|发布(?:版本|说明)?)"
    r"\s*[:=#]?\s*$",
    re.IGNORECASE,
)
_EXPLICIT_API_GROUP = re.compile(
    r"(?:api\s*group|apiGroup|api\s*组)\s*[:=]?\s*"
    r"(?P<group>[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)",
    re.IGNORECASE,
)
_GROUP_VERSION = re.compile(
    r"(?<![\w./-])(?P<group>[A-Za-z][A-Za-z0-9.-]*)/"
    r"(?P<version>v\d+(?:(?:alpha|beta)\d+)?)(?![\w.-])",
    re.IGNORECASE,
)
_EXPLICIT_KIND = re.compile(
    r"(?:kind|资源类型|对象类型)\s*[:=]?\s*(?P<kind>[A-Za-z][A-Za-z0-9]*)",
    re.IGNORECASE,
)

_KNOWN_API_GROUPS = frozenset(
    {
        "admissionregistration.k8s.io",
        "apiextensions.k8s.io",
        "apiregistration.k8s.io",
        "apps",
        "authentication.k8s.io",
        "authorization.k8s.io",
        "autoscaling",
        "batch",
        "certificates.k8s.io",
        "coordination.k8s.io",
        "core",
        "discovery.k8s.io",
        "events.k8s.io",
        "flowcontrol.apiserver.k8s.io",
        "networking.k8s.io",
        "node.k8s.io",
        "policy",
        "rbac.authorization.k8s.io",
        "resource.k8s.io",
        "scheduling.k8s.io",
        "storage.k8s.io",
    }
)

_KNOWN_KINDS = {
    name.casefold(): name
    for name in (
        "APIService",
        "CertificateSigningRequest",
        "ClusterRole",
        "ClusterRoleBinding",
        "ConfigMap",
        "ControllerRevision",
        "CronJob",
        "CustomResourceDefinition",
        "DaemonSet",
        "Deployment",
        "EndpointSlice",
        "Endpoints",
        "Event",
        "HorizontalPodAutoscaler",
        "Ingress",
        "IngressClass",
        "Job",
        "Lease",
        "LimitRange",
        "Namespace",
        "NetworkPolicy",
        "Node",
        "PersistentVolume",
        "PersistentVolumeClaim",
        "Pod",
        "PodDisruptionBudget",
        "PodTemplate",
        "PriorityClass",
        "ReplicaSet",
        "ReplicationController",
        "ResourceClaim",
        "ResourceQuota",
        "Role",
        "RoleBinding",
        "RuntimeClass",
        "Secret",
        "Service",
        "ServiceAccount",
        "StatefulSet",
        "StorageClass",
        "ValidatingAdmissionPolicy",
        "ValidatingWebhookConfiguration",
        "VolumeAttachment",
    )
}
_KNOWN_KIND_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?P<kind>"
    + "|".join(sorted((re.escape(kind) for kind in _KNOWN_KINDS.values()), key=len, reverse=True))
    + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_DOCUMENT_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "official_documentation",
        re.compile(r"\b(?:official\s+docs?|official\s+documentation)\b|官方文档", re.IGNORECASE),
    ),
    (
        "github_issue",
        re.compile(
            r"\b(?:github\s+issues?|issues?\s*#?\d+|open\s+issues?|closed\s+issues?)\b"
            r"|GitHub\s*议题",
            re.IGNORECASE,
        ),
    ),
    (
        "release_note",
        re.compile(
            r"\b(?:release\s+(?:notes?|versions?)|changelog)\b|发行说明|发布说明",
            re.IGNORECASE,
        ),
    ),
    (
        "api_reference",
        re.compile(
            r"\bapi\s+(?:reference|docs?|documentation)\b|API\s*(?:参考|文档)", re.IGNORECASE
        ),
    ),
    (
        "repository_document",
        re.compile(
            r"\b(?:repository|repo)\s+(?:docs?|markdown)\b|仓库(?:文档|Markdown)",
            re.IGNORECASE,
        ),
    ),
    ("kep", re.compile(r"(?<![A-Za-z0-9_])KEP(?:-\d+)?(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("blog", re.compile(r"\b(?:official\s+)?blogs?\b|官方博客|博客", re.IGNORECASE)),
)

_OPEN_ISSUE = re.compile(
    r"\b(?:open|opened)\s+(?:github\s+)?issues?\b|"
    r"\bissues?\s+(?:is\s+)?(?:open|opened)\b|未关闭(?:的)?\s*(?:GitHub\s*)?Issue",
    re.IGNORECASE,
)
_CLOSED_ISSUE = re.compile(
    r"\b(?:closed|resolved)\s+(?:github\s+)?issues?\b|"
    r"\bissues?\s+(?:is\s+)?(?:closed|resolved)\b|已关闭(?:的)?\s*(?:GitHub\s*)?Issue",
    re.IGNORECASE,
)
_EXPLICIT_ISSUE_STATE = re.compile(
    r"(?:github\s+)?issues?\s*(?:state|status|状态)\s*[:=]?\s*"
    r"(?P<state>open(?:ed)?|closed|resolved|未关闭|已关闭)",
    re.IGNORECASE,
)


def _unique[T](values: Iterable[T], *, case_insensitive: bool = False) -> list[T]:
    """Return values in first-seen order without duplicates."""
    result: list[T] = []
    seen: set[object] = set()
    for value in values:
        key: object = value.casefold() if case_insensitive and isinstance(value, str) else value
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _extract_versions(query: str) -> list[str]:
    matches = [
        (match.start(), match.group("version")) for match in _KUBERNETES_VERSION.finditer(query)
    ]
    for match in _BARE_KUBERNETES_VERSION.finditer(query):
        if not _RELEASE_CONTEXT.search(query[max(0, match.start() - 32) : match.start()]):
            matches.append((match.start(), match.group("version")))
    for match in _QUALIFIED_KUBERNETES_VERSION.finditer(query):
        if not _RELEASE_CONTEXT.search(query[max(0, match.start() - 32) : match.start()]):
            matches.append((match.start(), match.group("version") or match.group("suffix_version")))
    return _unique((value for _, value in sorted(matches)), case_insensitive=True)


def _extract_release_versions(query: str) -> list[str]:
    return _unique(
        (f"v{match.group('version')}" for match in _RELEASE_VERSION.finditer(query)),
        case_insensitive=True,
    )


def _extract_document_types(query: str) -> list[str]:
    return [
        document_type for document_type, pattern in _DOCUMENT_TYPE_PATTERNS if pattern.search(query)
    ]


def _extract_api_groups(query: str) -> tuple[list[str], list[str]]:
    groups: list[str] = []
    api_versions: list[str] = []
    groups.extend(match.group("group") for match in _EXPLICIT_API_GROUP.finditer(query))
    for match in _GROUP_VERSION.finditer(query):
        group = match.group("group")
        if group.casefold() in _KNOWN_API_GROUPS or "." in group:
            groups.append(group)
            api_versions.append(match.group("version"))
    return (
        _unique(groups, case_insensitive=True),
        _unique(api_versions, case_insensitive=True),
    )


def _extract_kinds(query: str) -> list[str]:
    kinds: list[str] = []
    for match in _EXPLICIT_KIND.finditer(query):
        value = match.group("kind")
        kinds.append(_KNOWN_KINDS.get(value.casefold(), value))
    for match in _KNOWN_KIND_PATTERN.finditer(query):
        kinds.append(_KNOWN_KINDS[match.group("kind").casefold()])
    return _unique(kinds, case_insensitive=True)


def _extract_issue_states(query: str) -> list[str]:
    states: list[str] = []
    if _OPEN_ISSUE.search(query):
        states.append("open")
    if _CLOSED_ISSUE.search(query):
        states.append("closed")
    for match in _EXPLICIT_ISSUE_STATE.finditer(query):
        state = match.group("state").casefold()
        states.append("open" if state in {"open", "opened", "未关闭"} else "closed")
    return _unique(states)


def extract_query_filters(query: str) -> QueryFilters:
    """Extract conservative structured filters while leaving the query intact."""
    api_groups, api_versions = _extract_api_groups(query)
    issue_states = _extract_issue_states(query)
    document_types = _extract_document_types(query)
    if issue_states and "github_issue" not in document_types:
        document_types.append("github_issue")
    document_type_order = {
        document_type: index for index, (document_type, _) in enumerate(_DOCUMENT_TYPE_PATTERNS)
    }
    document_types.sort(key=document_type_order.__getitem__)
    return QueryFilters(
        document_types=document_types,
        versions=_extract_versions(query),
        api_groups=api_groups,
        api_versions=api_versions,
        kinds=_extract_kinds(query),
        issue_states=issue_states,
        release_versions=_extract_release_versions(query),
    )


def merge_query_filters(extracted: QueryFilters, supplied: QueryFilters | None) -> QueryFilters:
    """Merge filters without weakening explicit caller constraints.

    For each list field, caller-supplied values replace extracted values when
    present.  Extracted metadata is retained, while caller metadata wins on key
    collisions.  This prevents query heuristics from broadening API-level
    source or tenancy filters.
    """
    if supplied is None:
        return extracted.model_copy(deep=True)

    metadata = {**extracted.metadata, **supplied.metadata}
    return QueryFilters(
        source_ids=_unique(supplied.source_ids or extracted.source_ids),
        document_types=_unique(
            supplied.document_types or extracted.document_types,
            case_insensitive=True,
        ),
        versions=_unique(supplied.versions or extracted.versions, case_insensitive=True),
        api_groups=_unique(supplied.api_groups or extracted.api_groups, case_insensitive=True),
        api_versions=_unique(
            supplied.api_versions or extracted.api_versions,
            case_insensitive=True,
        ),
        kinds=_unique(supplied.kinds or extracted.kinds, case_insensitive=True),
        issue_states=_unique(
            supplied.issue_states or extracted.issue_states,
            case_insensitive=True,
        ),
        release_versions=_unique(
            supplied.release_versions or extracted.release_versions,
            case_insensitive=True,
        ),
        metadata=metadata,
    )


class QueryFilterExtractor:
    """Small injectable facade for filter extraction and merging."""

    def extract(self, query: str) -> QueryFilters:
        """Extract filters from one normalized query."""
        return extract_query_filters(query)

    def extract_and_merge(self, query: str, supplied: QueryFilters | None = None) -> QueryFilters:
        """Extract filters and apply explicit caller constraints."""
        return merge_query_filters(self.extract(query), supplied)
