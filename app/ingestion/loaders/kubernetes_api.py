"""Loader for Kubernetes HTML and OpenAPI reference material."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from bs4.element import Tag

from app.domain.documents import RawDocument
from app.ingestion.loaders.base import DocumentLoader, LoaderError

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_GVK_TITLE_PATTERN = re.compile(
    r"(?P<kind>[A-Z][A-Za-z0-9]+)\s+(?P<version>v[0-9][A-Za-z0-9]*)\s+(?P<group>[A-Za-z0-9.-]+)"
)


class KubernetesAPIReferenceLoader(DocumentLoader):
    """Load Kubernetes API kinds and field paths from HTML or structured data."""

    def __init__(
        self,
        *,
        html_urls: Sequence[str] = (),
        structured_data: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        source_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        max_retry_delay_seconds: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not html_urls and structured_data is None:
            msg = "Kubernetes API loader requires at least one URL or structured payload"
            raise ValueError(msg)
        for url in html_urls:
            _validate_http_url(url)
        if source_url is not None:
            _validate_http_url(source_url)
        if max_retries < 0:
            msg = "max_retries cannot be negative"
            raise ValueError(msg)

        self.html_urls = tuple(html_urls)
        self.structured_data = structured_data
        self.source_url = source_url
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    async def load(self) -> list[RawDocument]:
        """Load configured reference inputs without blocking on HTML parsing."""
        documents: list[RawDocument] = []
        try:
            if self.structured_data is not None:
                documents.extend(
                    await asyncio.to_thread(
                        _parse_structured_reference,
                        self.structured_data,
                        self.source_url,
                    )
                )
            for url in self.html_urls:
                response = await self._get_with_retry(url)
                content_type = response.headers.get("Content-Type", "").lower()
                if "json" in content_type or url.lower().endswith(".json"):
                    payload = response.json()
                    documents.extend(
                        await asyncio.to_thread(_parse_structured_reference, payload, url)
                    )
                    continue
                updated_at = _parse_http_datetime(response.headers.get("Last-Modified"))
                documents.extend(
                    await asyncio.to_thread(_parse_html_reference, response.text, url, updated_at)
                )
            return documents
        finally:
            if self._owns_client:
                await self._client.aclose()

    async def _get_with_retry(self, url: str) -> httpx.Response:
        last_error: httpx.TransportError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.get(
                    url,
                    headers={"User-Agent": "opensource-doc-intelligence/0.1"},
                )
            except httpx.TransportError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                await self._sleep(self._exponential_delay(attempt))
                continue
            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                await self._sleep(self._response_delay(response, attempt))
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                msg = f"Kubernetes API reference request failed with status {response.status_code}"
                raise LoaderError(msg) from exc
            return response
        msg = f"Kubernetes API reference request failed after retries for {url}"
        raise LoaderError(msg) from last_error

    def _response_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(max(0.0, float(retry_after)), self.max_retry_delay_seconds)
            except ValueError:
                pass
        return self._exponential_delay(attempt)

    def _exponential_delay(self, attempt: int) -> float:
        return min(
            self.backoff_base_seconds * (2.0**attempt),
            self.max_retry_delay_seconds,
        )


def _parse_html_reference(
    html: str,
    source_url: str,
    updated_at: datetime | None,
) -> list[RawDocument]:
    soup = BeautifulSoup(html, "html.parser")
    containers = [tag for tag in soup.select("[data-kind]") if isinstance(tag, Tag)]
    if not containers:
        main = soup.find("main") or soup.find("article") or soup.body
        if isinstance(main, Tag):
            containers = _split_gvk_sections(main)
            if not containers:
                containers = [main]
    documents: list[RawDocument] = []
    page_title = _tag_text(soup.title) or "Kubernetes API Reference"

    for container in containers:
        heading_tag = container.find(re.compile(r"^h[1-6]$"))
        heading = _tag_text(heading_tag) or page_title
        kind = _attribute(container, "data-kind")
        group = _attribute(container, "data-api-group") or _attribute(container, "data-group")
        version = _attribute(container, "data-api-version") or _attribute(container, "data-version")
        title_match = _GVK_TITLE_PATTERN.search(heading)
        if title_match:
            kind = kind or title_match.group("kind")
            version = version or title_match.group("version")
            group = group or title_match.group("group")
        kind = kind or _meta_content(soup, "api-kind") or heading.split()[0]
        group = group or _meta_content(soup, "api-group") or "core"
        version = version or _meta_content(soup, "api-version") or "unknown"
        fields = _extract_html_fields(container)
        fallback_text = container.get_text("\n", strip=True)
        content = _render_reference_content(
            heading,
            group,
            version,
            kind,
            fields,
            fallback_text=fallback_text,
        )
        anchor = _attribute(container, "id")
        if not anchor and isinstance(heading_tag, Tag):
            anchor = _attribute(heading_tag, "id")
        canonical_url = f"{source_url}#{anchor}" if anchor else source_url
        identifier_material = f"{canonical_url}|{group}|{version}|{kind}"
        stable_hash = hashlib.sha256(identifier_material.encode()).hexdigest()[:20]
        documents.append(
            RawDocument(
                source_type="kubernetes_api_reference",
                external_id=f"k8s-api-html:{stable_hash}",
                title=heading,
                content=content,
                canonical_url=canonical_url,
                source_version=version,
                updated_at=updated_at,
                metadata={
                    "origin": "html",
                    "api_group": group,
                    "version": version,
                    "kind": kind,
                    "fields": fields,
                    "field_paths": [field["path"] for field in fields],
                    "page_title": page_title,
                    "anchor": anchor,
                },
            )
        )
    return documents


def _split_gvk_sections(main: Tag) -> list[Tag]:
    """Split a plain reference page into stable per-GVK sections when possible."""

    headings = [
        heading
        for heading in main.find_all(re.compile(r"^h[1-6]$"))
        if isinstance(heading, Tag) and _GVK_TITLE_PATTERN.search(_tag_text(heading) or "")
    ]
    sections: list[Tag] = []
    for heading in headings:
        level = int(heading.name[1]) if heading.name else 6
        fragments = [str(heading)]
        for sibling in heading.next_siblings:
            if isinstance(sibling, Tag) and re.fullmatch(r"h[1-6]", sibling.name or ""):
                sibling_level = int(sibling.name[1])
                sibling_is_gvk = _GVK_TITLE_PATTERN.search(_tag_text(sibling) or "")
                if sibling_level <= level and sibling_is_gvk:
                    break
            fragments.append(str(sibling))
        fragment = BeautifulSoup(f"<section>{''.join(fragments)}</section>", "html.parser")
        section = fragment.find("section")
        if isinstance(section, Tag):
            sections.append(section)
    return sections


def _extract_html_fields(container: Tag) -> list[dict[str, str]]:
    fields: list[dict[str, str]] = []
    for table in container.find_all("table"):
        if not isinstance(table, Tag):
            continue
        rows = [row for row in table.find_all("tr") if isinstance(row, Tag)]
        if not rows:
            continue
        header_cells = rows[0].find_all(["th", "td"])
        headers = [cell.get_text(" ", strip=True).lower() for cell in header_cells]
        path_index = _find_column(headers, ("field", "name", "field path", "path"), default=0)
        description_index = _find_column(
            headers,
            ("description", "details"),
            default=min(1, len(headers) - 1),
        )
        type_index = _find_column(headers, ("type",), default=-1)
        for row in rows[1:]:
            cells = row.find_all(["th", "td"])
            if not cells or path_index >= len(cells):
                continue
            explicit_path = _attribute(row, "data-field-path")
            path = explicit_path or cells[path_index].get_text(" ", strip=True).strip("`")
            if not path:
                continue
            description = (
                cells[description_index].get_text(" ", strip=True)
                if 0 <= description_index < len(cells)
                else ""
            )
            field_type = (
                cells[type_index].get_text(" ", strip=True) if 0 <= type_index < len(cells) else ""
            )
            fields.append({"path": path, "description": description, "type": field_type})
    return fields


def _parse_structured_reference(
    payload: object,
    source_url: str | None,
) -> list[RawDocument]:
    if isinstance(payload, Mapping):
        definitions = payload.get("definitions")
        if isinstance(definitions, Mapping):
            return _parse_openapi_definitions(definitions, source_url)
        components = payload.get("components")
        if isinstance(components, Mapping):
            schemas = components.get("schemas")
            if isinstance(schemas, Mapping):
                return _parse_openapi_definitions(schemas, source_url)
        items = payload.get("items")
        if isinstance(items, list):
            return _parse_field_records(items, source_url)
        return _parse_field_records([payload], source_url)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return _parse_field_records(payload, source_url)
    msg = "Kubernetes structured API reference must be an object or list"
    raise LoaderError(msg)


def _parse_openapi_definitions(
    raw_definitions: Mapping[object, object],
    source_url: str | None,
) -> list[RawDocument]:
    definitions = {
        key: value
        for key, value in raw_definitions.items()
        if isinstance(key, str) and isinstance(value, Mapping)
    }
    documents: list[RawDocument] = []
    for definition_name, raw_schema in sorted(definitions.items()):
        schema = cast(Mapping[object, object], raw_schema)
        gvks = _definition_gvks(definition_name, schema)
        if not gvks:
            continue
        properties = schema.get("properties")
        fields = _flatten_properties(
            properties if isinstance(properties, Mapping) else {},
            definitions,
            prefix="",
            visited_references=frozenset({definition_name}),
        )
        description = schema.get("description")
        fallback = description if isinstance(description, str) else ""
        external_docs = schema.get("externalDocs")
        external_url = external_docs.get("url") if isinstance(external_docs, Mapping) else None
        canonical_url = external_url if isinstance(external_url, str) else source_url
        for group, version, kind in gvks:
            title = f"{kind} {version} {group}"
            documents.append(
                RawDocument(
                    source_type="kubernetes_api_reference",
                    external_id=f"k8s-openapi:{group}:{version}:{kind}:{definition_name}",
                    title=title,
                    content=_render_reference_content(
                        title,
                        group,
                        version,
                        kind,
                        fields,
                        fallback_text=fallback,
                    ),
                    canonical_url=canonical_url,
                    source_version=version,
                    metadata={
                        "origin": "openapi",
                        "api_group": group,
                        "version": version,
                        "kind": kind,
                        "definition": definition_name,
                        "fields": fields,
                        "field_paths": [field["path"] for field in fields],
                    },
                )
            )
    return documents


def _definition_gvks(
    definition_name: str,
    schema: Mapping[object, object],
) -> list[tuple[str, str, str]]:
    extension = schema.get("x-kubernetes-group-version-kind")
    gvks: list[tuple[str, str, str]] = []
    if isinstance(extension, list):
        for item in extension:
            if not isinstance(item, Mapping):
                continue
            group = item.get("group")
            version = item.get("version")
            kind = item.get("kind")
            if isinstance(version, str) and isinstance(kind, str):
                gvks.append((group if isinstance(group, str) and group else "core", version, kind))
    if gvks:
        return gvks

    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not {"apiVersion", "kind"}.issubset(properties):
        return []
    marker = ".api."
    if marker not in definition_name:
        return []
    parts = definition_name.split(marker, maxsplit=1)[1].split(".")
    if len(parts) < 3:
        return []
    return [(parts[-3], parts[-2], parts[-1])]


def _flatten_properties(
    raw_properties: Mapping[object, object],
    definitions: Mapping[str, Mapping[object, object]],
    *,
    prefix: str,
    visited_references: frozenset[str],
    depth: int = 0,
) -> list[dict[str, str]]:
    if depth >= 8:
        return []
    fields: list[dict[str, str]] = []
    for name, raw_schema in raw_properties.items():
        if not isinstance(name, str) or not isinstance(raw_schema, Mapping):
            continue
        path = f"{prefix}.{name}" if prefix else name
        description = raw_schema.get("description")
        field_type = _schema_type(raw_schema)
        fields.append(
            {
                "path": path,
                "description": description if isinstance(description, str) else "",
                "type": field_type,
            }
        )
        nested_schema: Mapping[object, object] = raw_schema
        reference = raw_schema.get("$ref")
        reference_name: str | None = None
        if isinstance(reference, str) and reference.startswith("#/definitions/"):
            reference_name = reference.removeprefix("#/definitions/")
            resolved = definitions.get(reference_name)
            if resolved is not None and reference_name not in visited_references:
                nested_schema = resolved
        elif raw_schema.get("type") == "array":
            items = raw_schema.get("items")
            if isinstance(items, Mapping):
                nested_schema = items
                item_reference = items.get("$ref")
                if isinstance(item_reference, str) and item_reference.startswith("#/definitions/"):
                    reference_name = item_reference.removeprefix("#/definitions/")
                    resolved = definitions.get(reference_name)
                    if resolved is not None and reference_name not in visited_references:
                        nested_schema = resolved
        nested_properties = nested_schema.get("properties")
        if not isinstance(nested_properties, Mapping):
            continue
        next_visited = visited_references
        if reference_name is not None:
            next_visited = visited_references | {reference_name}
        fields.extend(
            _flatten_properties(
                nested_properties,
                definitions,
                prefix=path,
                visited_references=next_visited,
                depth=depth + 1,
            )
        )
    return fields


def _parse_field_records(
    records: Sequence[object],
    source_url: str | None,
) -> list[RawDocument]:
    grouped: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    titles: dict[tuple[str, str, str], str] = {}
    urls: dict[tuple[str, str, str], str | None] = {}
    descriptions: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    updated_times: dict[tuple[str, str, str], datetime | None] = {}
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            continue
        group = _mapping_string(raw_record, "api_group", "group") or "core"
        version = _mapping_string(raw_record, "version", "api_version") or "unknown"
        kind = _mapping_string(raw_record, "kind") or "UnknownKind"
        key = (group, version, kind)
        titles[key] = _mapping_string(raw_record, "title") or f"{kind} {version} {group}"
        urls[key] = _mapping_string(raw_record, "url", "canonical_url") or source_url
        description = _mapping_string(raw_record, "description")
        field_path = _mapping_string(raw_record, "field_path", "path")
        if field_path:
            grouped[key].append(
                {
                    "path": field_path,
                    "description": description or "",
                    "type": _mapping_string(raw_record, "type", "field_type") or "",
                }
            )
        elif description:
            descriptions[key].append(description)
        raw_fields = raw_record.get("fields")
        if isinstance(raw_fields, list):
            for raw_field in raw_fields:
                if not isinstance(raw_field, Mapping):
                    continue
                path = _mapping_string(raw_field, "field_path", "path", "name")
                if path:
                    grouped[key].append(
                        {
                            "path": path,
                            "description": _mapping_string(raw_field, "description") or "",
                            "type": _mapping_string(raw_field, "type", "field_type") or "",
                        }
                    )
        updated_times[key] = _parse_iso_datetime(_mapping_string(raw_record, "updated_at"))

    documents: list[RawDocument] = []
    all_keys = set(grouped) | set(titles)
    for group, version, kind in sorted(all_keys):
        key = (group, version, kind)
        fields = grouped[key]
        title = titles[key]
        documents.append(
            RawDocument(
                source_type="kubernetes_api_reference",
                external_id=f"k8s-api-record:{group}:{version}:{kind}",
                title=title,
                content=_render_reference_content(
                    title,
                    group,
                    version,
                    kind,
                    fields,
                    fallback_text="\n\n".join(descriptions[key]),
                ),
                canonical_url=urls[key],
                source_version=version,
                updated_at=updated_times.get(key),
                metadata={
                    "origin": "structured_records",
                    "api_group": group,
                    "version": version,
                    "kind": kind,
                    "fields": fields,
                    "field_paths": [field["path"] for field in fields],
                },
            )
        )
    return documents


def _render_reference_content(
    title: str,
    group: str,
    version: str,
    kind: str,
    fields: Sequence[Mapping[str, str]],
    *,
    fallback_text: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- API Group: `{group}`",
        f"- Version: `{version}`",
        f"- Kind: `{kind}`",
    ]
    if fallback_text:
        lines.extend(["", fallback_text.strip()])
    if fields:
        lines.extend(["", "## Fields", ""])
        for field in fields:
            path = field.get("path", "")
            field_type = field.get("type", "")
            description = field.get("description", "")
            type_suffix = f" ({field_type})" if field_type else ""
            lines.append(f"### `{path}`{type_suffix}")
            if description:
                lines.extend(["", description])
    return "\n".join(lines).strip()


def _schema_type(schema: Mapping[object, object]) -> str:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        if schema_type == "array":
            items = schema.get("items")
            if isinstance(items, Mapping):
                return f"array[{_schema_type(items)}]"
        return schema_type
    reference = schema.get("$ref")
    if isinstance(reference, str):
        return reference.rsplit("/", maxsplit=1)[-1]
    return "object"


def _mapping_string(mapping: Mapping[object, object], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _find_column(headers: Sequence[str], candidates: Sequence[str], *, default: int) -> int:
    for candidate in candidates:
        if candidate in headers:
            return headers.index(candidate)
    return default


def _tag_text(value: object) -> str | None:
    if isinstance(value, Tag):
        text = value.get_text(" ", strip=True)
        return text or None
    return None


def _attribute(tag: Tag, name: str) -> str | None:
    value = tag.attrs.get(name)
    if isinstance(value, str):
        return value.strip() or None
    return None


def _meta_content(soup: BeautifulSoup, name: str) -> str | None:
    tag = soup.find("meta", attrs={"name": name})
    return _attribute(tag, "content") if isinstance(tag, Tag) else None


def _parse_http_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _validate_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        msg = f"Kubernetes API source URL must be HTTP(S): {url}"
        raise ValueError(msg)
