"""Shared helpers for source parsers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.domain.documents import DocumentType, RawDocument, SourceMapEntry

_SOURCE_TYPE_TO_DOCUMENT_TYPE: dict[str, DocumentType] = {
    "api_reference": DocumentType.API_REFERENCE,
    "github_issue": DocumentType.GITHUB_ISSUE,
    "github_issues": DocumentType.GITHUB_ISSUE,
    "github_repo": DocumentType.REPOSITORY_DOCUMENT,
    "github_repository": DocumentType.REPOSITORY_DOCUMENT,
    "html": DocumentType.OFFICIAL_DOCUMENTATION,
    "kep": DocumentType.KEP,
    "kubernetes_api": DocumentType.API_REFERENCE,
    "kubernetes_api_reference": DocumentType.API_REFERENCE,
    "official_documentation": DocumentType.OFFICIAL_DOCUMENTATION,
    "release": DocumentType.RELEASE_NOTE,
    "release_note": DocumentType.RELEASE_NOTE,
    "release_notes": DocumentType.RELEASE_NOTE,
    "repository": DocumentType.REPOSITORY_DOCUMENT,
}


def infer_document_type(document: RawDocument) -> DocumentType:
    """Resolve a document category without coupling parsers to individual loaders."""

    configured = document.metadata.get("document_type")
    if configured is not None:
        try:
            return DocumentType(str(configured))
        except ValueError:
            pass
    return _SOURCE_TYPE_TO_DOCUMENT_TYPE.get(
        document.source_type.casefold(), DocumentType.OFFICIAL_DOCUMENTATION
    )


def normalize_newlines_with_map(content: str) -> tuple[str, list[SourceMapEntry]]:
    """Normalize CRLF/CR newlines and retain an exact line-oriented source map."""

    normalized_parts: list[str] = []
    entries: list[SourceMapEntry] = []
    source_cursor = 0
    normalized_cursor = 0

    for source_line_number, source_line in enumerate(content.splitlines(keepends=True), start=1):
        body = source_line
        if source_line.endswith("\r\n"):
            body = f"{source_line[:-2]}\n"
        elif source_line.endswith("\r"):
            body = f"{source_line[:-1]}\n"
        normalized_parts.append(body)
        entries.append(
            SourceMapEntry(
                normalized_start=normalized_cursor,
                normalized_end=normalized_cursor + len(body),
                source_start=source_cursor,
                source_end=source_cursor + len(source_line),
                source_start_line=source_line_number,
                source_end_line=source_line_number,
            )
        )
        normalized_cursor += len(body)
        source_cursor += len(source_line)

    # splitlines() returns nothing for an empty string and omits the final empty line.
    if not normalized_parts and content == "":
        return "", []
    if source_cursor < len(content):
        tail = content[source_cursor:]
        normalized_parts.append(tail)
        entries.append(
            SourceMapEntry(
                normalized_start=normalized_cursor,
                normalized_end=normalized_cursor + len(tail),
                source_start=source_cursor,
                source_end=len(content),
                source_start_line=max(1, len(entries) + 1),
                source_end_line=max(1, len(entries) + 1),
            )
        )
    return "".join(normalized_parts), entries


def line_starts(content: str) -> list[int]:
    """Return the character offset of every line start, plus an EOF sentinel."""

    starts = [0]
    starts.extend(index + 1 for index, character in enumerate(content) if character == "\n")
    if starts[-1] != len(content):
        starts.append(len(content))
    return starts


def line_range_to_offsets(
    starts: list[int], line_range: Iterable[int] | Sequence[int] | None, content_length: int
) -> tuple[int, int]:
    """Convert markdown-it's half-open zero-based line map into offsets."""

    if line_range is None:
        return 0, 0
    values = list(line_range)
    if len(values) != 2:
        return 0, 0
    start_line, end_line = values
    start = starts[start_line] if start_line < len(starts) else content_length
    end = starts[end_line] if end_line < len(starts) else content_length
    return start, end
