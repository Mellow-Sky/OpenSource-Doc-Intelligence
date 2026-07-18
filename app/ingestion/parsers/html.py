"""HTML parser that emits a structure-preserving Markdown representation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from app.domain.documents import ParsedDocument, RawDocument, SourceMapEntry
from app.ingestion.parsers.markdown import MarkdownDocumentParser

_EXCLUDED_TAGS = {"aside", "footer", "form", "header", "nav", "noscript", "script", "style", "svg"}
_CONTAINER_TAGS = {"article", "body", "div", "main", "section"}


@dataclass(slots=True)
class _RenderedPart:
    text: str
    source_start: int
    source_end: int


class HTMLDocumentParser:
    """Extract semantic HTML blocks without flattening code, tables, or links."""

    def __init__(self) -> None:
        self._markdown_parser = MarkdownDocumentParser()

    def parse(self, document: RawDocument) -> ParsedDocument:
        """Convert HTML to canonical Markdown and retain approximate source spans."""

        soup = BeautifulSoup(document.content, "html.parser")
        for excluded in soup.find_all(_EXCLUDED_TAGS):
            excluded.decompose()

        root = soup.find("main") or soup.find("article") or soup.body or soup
        parts: list[_RenderedPart] = []
        source_cursor = 0
        for child in root.children:
            source_cursor = self._render_block(child, document.content, source_cursor, parts)

        normalized = self._join_parts(parts)
        metadata = dict(document.metadata)
        metadata.update({"parser": "html", "normalized_format": "markdown"})
        synthetic = RawDocument(
            source_type=document.source_type,
            external_id=document.external_id,
            title=document.title,
            content=normalized,
            canonical_url=document.canonical_url,
            source_version=document.source_version,
            updated_at=document.updated_at,
            metadata=metadata,
        )
        parsed = self._markdown_parser.parse(synthetic)
        parsed_metadata = dict(parsed.metadata)
        parsed_metadata["parser"] = "html"
        return parsed.model_copy(
            update={
                "source_map": self._source_map(parts),
                "metadata": parsed_metadata,
            }
        )

    def _render_block(
        self,
        node: object,
        source: str,
        source_cursor: int,
        parts: list[_RenderedPart],
    ) -> int:
        if isinstance(node, NavigableString):
            text = self._normalize_inline(str(node))
            if text:
                parts.append(
                    _RenderedPart(
                        text=text,
                        source_start=source_cursor,
                        source_end=source_cursor,
                    )
                )
            return source_cursor
        if not isinstance(node, Tag):
            return source_cursor

        tag_name = node.name.casefold()
        if tag_name in _EXCLUDED_TAGS:
            return source_cursor
        start, end = self._locate_tag(node, source, source_cursor)

        if re.fullmatch(r"h[1-6]", tag_name):
            level = int(tag_name[1])
            text = self._render_inline(node).strip()
            if text:
                parts.append(_RenderedPart(f"{'#' * level} {text}", start, end))
        elif tag_name == "pre":
            code_tag = node.find("code")
            language = self._code_language(code_tag)
            code = node.get_text("", strip=False).strip("\n")
            fence = self._safe_fence(code)
            info = language or "text"
            parts.append(_RenderedPart(f"{fence}{info}\n{code}\n{fence}", start, end))
        elif tag_name == "table":
            table = self._render_table(node)
            if table:
                parts.append(_RenderedPart(table, start, end))
        elif tag_name in {"p", "blockquote"}:
            text = self._render_inline(node).strip()
            if text:
                if tag_name == "blockquote":
                    text = "\n".join(f"> {line}" for line in text.splitlines())
                parts.append(_RenderedPart(text, start, end))
        elif tag_name in {"ul", "ol"}:
            ordered = tag_name == "ol"
            lines: list[str] = []
            for item_index, item in enumerate(node.find_all("li", recursive=False), start=1):
                marker = f"{item_index}." if ordered else "-"
                lines.append(f"{marker} {self._render_inline(item).strip()}")
            if lines:
                parts.append(_RenderedPart("\n".join(lines), start, end))
        elif tag_name in _CONTAINER_TAGS or tag_name in {"dl", "figure"}:
            cursor = start
            for child in node.children:
                cursor = self._render_block(child, source, cursor, parts)
            return max(source_cursor, end, cursor)
        else:
            text = self._render_inline(node).strip()
            if text:
                parts.append(_RenderedPart(text, start, end))
        return max(source_cursor, end)

    def _render_inline(self, tag: Tag) -> str:
        rendered: list[str] = []
        for child in tag.children:
            if isinstance(child, NavigableString):
                rendered.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name.casefold()
            if name == "a":
                text = self._normalize_inline(child.get_text(" ", strip=True))
                target = cast(str | None, child.get("href"))
                rendered.append(f"[{text}]({target})" if text and target else text)
            elif name == "code":
                code = child.get_text("", strip=False)
                delimiter = "``" if "`" in code else "`"
                rendered.append(f"{delimiter}{code}{delimiter}")
            elif name == "br":
                rendered.append("\n")
            elif name in {"ul", "ol"}:
                rendered.append(" " + child.get_text(" ", strip=True))
            else:
                rendered.append(self._render_inline(child))
        return self._normalize_inline("".join(rendered), preserve_newlines=True)

    @staticmethod
    def _render_table(table: Tag) -> str:
        rows: list[list[str]] = []
        header_row_index: int | None = None
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"], recursive=False)
            if not cells:
                continue
            if header_row_index is None and any(cell.name == "th" for cell in cells):
                header_row_index = len(rows)
            rows.append(
                [
                    cell.get_text(" ", strip=True).replace("|", "\\|").replace("\n", " ")
                    for cell in cells
                ]
            )
        if not rows:
            return ""
        width = max(len(row) for row in rows)
        padded = [row + [""] * (width - len(row)) for row in rows]
        if header_row_index is None:
            header_row_index = 0
        header = padded.pop(header_row_index)
        lines = [
            f"| {' | '.join(header)} |",
            f"| {' | '.join('---' for _ in range(width))} |",
        ]
        lines.extend(f"| {' | '.join(row)} |" for row in padded)
        return "\n".join(lines)

    @staticmethod
    def _code_language(code_tag: Tag | None) -> str | None:
        if code_tag is None:
            return None
        raw_classes = code_tag.get("class")
        if raw_classes is None:
            return None
        classes = raw_classes.split() if isinstance(raw_classes, str) else raw_classes
        for value in classes:
            match = re.match(r"(?:language-|lang-)([\w.+-]+)", str(value))
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _safe_fence(code: str) -> str:
        longest = max((len(match.group(0)) for match in re.finditer(r"`+", code)), default=0)
        return "`" * max(3, longest + 1)

    @staticmethod
    def _normalize_inline(text: str, *, preserve_newlines: bool = False) -> str:
        if preserve_newlines:
            return "\n".join(re.sub(r"[\t \f\v]+", " ", line) for line in text.splitlines())
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _locate_tag(tag: Tag, source: str, cursor: int) -> tuple[int, int]:
        lower_source = source.casefold()
        open_start = lower_source.find(f"<{tag.name.casefold()}", cursor)
        if open_start < 0:
            open_start = cursor
        close_marker = f"</{tag.name.casefold()}>"
        close_start = lower_source.find(close_marker, open_start)
        end = close_start + len(close_marker) if close_start >= 0 else open_start
        return open_start, max(open_start, end)

    @staticmethod
    def _join_parts(parts: list[_RenderedPart]) -> str:
        return "\n\n".join(part.text.strip("\n") for part in parts if part.text.strip())

    @staticmethod
    def _source_map(parts: list[_RenderedPart]) -> list[SourceMapEntry]:
        entries: list[SourceMapEntry] = []
        normalized_cursor = 0
        nonempty_parts = [part for part in parts if part.text.strip()]
        for index, part in enumerate(nonempty_parts):
            text = part.text.strip("\n")
            entries.append(
                SourceMapEntry(
                    normalized_start=normalized_cursor,
                    normalized_end=normalized_cursor + len(text),
                    source_start=part.source_start,
                    source_end=part.source_end,
                )
            )
            normalized_cursor += len(text)
            if index < len(nonempty_parts) - 1:
                normalized_cursor += 2
        return entries


HTMLParser = HTMLDocumentParser
