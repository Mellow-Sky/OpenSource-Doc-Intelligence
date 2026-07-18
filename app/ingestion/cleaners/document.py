"""Safe, structure-aware document cleaning."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.domain.documents import ParsedDocument, RawDocument, SourceMapEntry
from app.ingestion.parsers.markdown import MarkdownDocumentParser


class DocumentReparser(Protocol):
    """Parser used to rebuild structure after cleanup."""

    def parse(self, document: RawDocument) -> ParsedDocument:
        """Parse one normalized raw document."""


_CONTROL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MARKDOWN_ANCHOR_RE = re.compile(r"\s*\{#[A-Za-z][\w:.-]*\}\s*$")
_HTML_ANCHOR_RE = re.compile(r"<a\s+(?:name|id)=[\"'][^\"']+[\"']\s*>\s*</a>", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TOC_TITLE_RE = re.compile(r"^(?:table\s+of\s+contents|contents|toc|目录|本文目录)$", re.IGNORECASE)
_TOC_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)?\[[^\]]+\]\(#[^)]+\)\s*$", re.IGNORECASE)
_NAVIGATION_LINE_RE = re.compile(
    r"^\s*(?:previous|next|back to top|edit this page|view page source|"
    r"上一页|下一页|返回顶部|编辑此页)\s*(?:[\u203a\u00bb\u2192|].*)?$",
    re.IGNORECASE,
)
_ISSUE_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:_no response_|no response|not applicable|please describe here|"
    r"replace this text|在此填写|请在此描述)\s*$",
    re.IGNORECASE,
)
_DISCLAIMER_MARKERS = (
    "automatically generated",
    "do not edit",
    "legal disclaimer",
    "this disclaimer",
    "自动生成",
    "免责声明",
    "请勿编辑",
)


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    """Conservative cleaning controls."""

    minimum_content_characters: int = 40
    duplicate_paragraph_characters: int = 160

    def __post_init__(self) -> None:
        if self.minimum_content_characters < 0:
            raise ValueError("minimum_content_characters must be non-negative")
        if self.duplicate_paragraph_characters < 1:
            raise ValueError("duplicate_paragraph_characters must be positive")


class DocumentCleaner:
    """Remove source chrome and templates while treating fenced code as opaque."""

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config = config or CleaningConfig()
        self._parser = MarkdownDocumentParser()

    def clean(
        self,
        document: ParsedDocument,
        *,
        parser: DocumentReparser | None = None,
    ) -> ParsedDocument:
        """Clean a parsed document and rebuild all normalized structural offsets."""

        original = document.content
        seen_disclaimers: set[str] = set()
        cleaned = self._transform_outside_intervals(
            original,
            [(block.start_offset, block.end_offset) for block in document.code_blocks]
            + [(table.start_offset, table.end_offset) for table in document.tables],
            lambda segment: self._clean_unprotected(segment, seen_disclaimers),
        )
        cleaned = cleaned.strip("\r\n")
        if cleaned:
            cleaned += "\n"

        metadata = dict(document.metadata)
        source_parser = str(metadata.get("parser", "unknown"))
        significant_length = len(re.sub(r"\s+", "", cleaned))
        metadata.update(
            {
                "cleaner": "safe-structural-v1",
                "source_parser": source_parser,
                "cleaning_removed_characters": max(0, len(original) - len(cleaned)),
                "quality_status": (
                    "ready"
                    if significant_length >= self.config.minimum_content_characters
                    else "too_short"
                ),
                "document_type": document.document_type.value,
            }
        )
        raw = RawDocument(
            source_type=document.source_type,
            external_id=document.external_id,
            title=document.title,
            content=cleaned,
            canonical_url=str(document.canonical_url) if document.canonical_url else None,
            source_version=document.source_version,
            updated_at=document.updated_at,
            metadata=metadata,
        )
        reparsed = (parser or self._parser).parse(raw)
        reparsed_metadata = dict(reparsed.metadata)
        reparsed_metadata.update(metadata)
        return reparsed.model_copy(
            update={
                "source_map": self._compose_source_map(cleaned, original, document.source_map),
                "metadata": reparsed_metadata,
            }
        )

    def _clean_unprotected(self, segment: str, seen_disclaimers: set[str]) -> str:
        segment = _CONTROL_CHARACTERS_RE.sub("", segment)
        segment = _HTML_COMMENT_RE.sub("", segment)
        segment = _HTML_ANCHOR_RE.sub("", segment)
        segment = self._remove_toc(segment)

        retained_lines: list[str] = []
        for line in segment.splitlines(keepends=True):
            body = line.rstrip("\r\n")
            newline = "\n" if line.endswith(("\n", "\r")) else ""
            body = _MARKDOWN_ANCHOR_RE.sub("", body).rstrip()
            if _NAVIGATION_LINE_RE.fullmatch(body) or _ISSUE_PLACEHOLDER_RE.fullmatch(body):
                continue
            retained_lines.append(f"{body}{newline}")
        segment = "".join(retained_lines)
        segment = self._remove_duplicate_disclaimers(segment, seen_disclaimers)
        return re.sub(r"\n{3,}", "\n\n", segment)

    def _remove_duplicate_disclaimers(self, segment: str, seen: set[str]) -> str:
        paragraphs = re.split(r"(\n\s*\n)", segment)
        retained: list[str] = []
        for paragraph in paragraphs:
            normalized = re.sub(r"\s+", " ", paragraph).strip().casefold()
            is_separator = not normalized
            looks_repeated = len(normalized) >= self.config.duplicate_paragraph_characters
            looks_like_disclaimer = any(marker in normalized for marker in _DISCLAIMER_MARKERS)
            if not is_separator and (looks_repeated or looks_like_disclaimer):
                if normalized in seen:
                    continue
                seen.add(normalized)
            retained.append(paragraph)
        return "".join(retained)

    @staticmethod
    def _remove_toc(segment: str) -> str:
        lines = segment.splitlines(keepends=True)
        retained: list[str] = []
        index = 0
        while index < len(lines):
            body = lines[index].rstrip("\r\n")
            heading_match = _HEADING_RE.match(body)
            if not heading_match or not _TOC_TITLE_RE.fullmatch(heading_match.group(2).strip()):
                retained.append(lines[index])
                index += 1
                continue

            probe = index + 1
            found_toc_item = False
            while probe < len(lines):
                candidate = lines[probe].rstrip("\r\n")
                if not candidate.strip() or _TOC_ITEM_RE.fullmatch(candidate):
                    found_toc_item = found_toc_item or bool(_TOC_ITEM_RE.fullmatch(candidate))
                    probe += 1
                    continue
                break
            if found_toc_item:
                index = probe
            else:
                retained.append(lines[index])
                index += 1
        return "".join(retained)

    @staticmethod
    def _transform_outside_intervals(
        content: str,
        intervals: Sequence[tuple[int, int]],
        transform: Callable[[str], str],
    ) -> str:
        """Transform prose while copying parsed code and table slices byte-for-byte."""

        merged: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            start = max(0, min(start, len(content)))
            end = max(start, min(end, len(content)))
            if start == end:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        output: list[str] = []
        cursor = 0
        for start, end in merged:
            output.append(transform(content[cursor:start]))
            output.append(content[start:end])
            cursor = end
        output.append(transform(content[cursor:]))
        return "".join(output)

    @classmethod
    def _compose_source_map(
        cls,
        cleaned: str,
        original: str,
        original_map: list[SourceMapEntry],
    ) -> list[SourceMapEntry]:
        if not cleaned:
            return []
        entries: list[SourceMapEntry] = []
        search_cursor = 0
        normalized_cursor = 0
        for line in cleaned.splitlines(keepends=True):
            needle = line.rstrip("\r\n")
            found = original.find(needle, search_cursor) if needle else search_cursor
            if found < 0 and needle.strip():
                stripped = needle.strip()
                found = original.find(stripped, search_cursor)
                needle = stripped
            if found < 0:
                found = min(search_cursor, len(original))
            original_start = cls._original_offset(found, original_map)
            original_end = cls._original_offset(found + len(needle), original_map)
            source_start_line = cls._original_line(found, original_map)
            source_end_line = cls._original_line(max(found, found + len(needle) - 1), original_map)
            entries.append(
                SourceMapEntry(
                    normalized_start=normalized_cursor,
                    normalized_end=normalized_cursor + len(line),
                    source_start=original_start,
                    source_end=max(original_start, original_end),
                    source_start_line=source_start_line,
                    source_end_line=source_end_line,
                )
            )
            search_cursor = found + len(needle)
            normalized_cursor += len(line)
        return entries

    @staticmethod
    def _original_offset(offset: int, source_map: list[SourceMapEntry]) -> int:
        if not source_map:
            return offset
        for entry in source_map:
            if entry.normalized_start <= offset < entry.normalized_end:
                normalized_span = max(1, entry.normalized_end - entry.normalized_start)
                source_span = max(0, entry.source_end - entry.source_start)
                relative = min(normalized_span, max(0, offset - entry.normalized_start))
                return entry.source_start + round(relative / normalized_span * source_span)
        return source_map[-1].source_end

    @staticmethod
    def _original_line(offset: int, source_map: list[SourceMapEntry]) -> int | None:
        if not source_map:
            return None
        for entry in source_map:
            if entry.normalized_start <= offset < entry.normalized_end:
                return entry.source_start_line
        return source_map[-1].source_end_line


SafeDocumentCleaner = DocumentCleaner
