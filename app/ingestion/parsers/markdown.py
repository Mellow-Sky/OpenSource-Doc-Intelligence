"""Structure-preserving Markdown parser."""

from __future__ import annotations

import re
from hashlib import sha256

from markdown_it import MarkdownIt

from app.domain.documents import (
    CodeBlock,
    DocumentLink,
    Heading,
    ParsedDocument,
    RawDocument,
    TableBlock,
)
from app.ingestion.parsers.common import (
    infer_document_type,
    line_range_to_offsets,
    line_starts,
    normalize_newlines_with_map,
)

_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\(\s*([^\s)]+)(?:\s+[\"'][^\n)]*[\"'])?\s*\)")
_AUTOLINK_RE = re.compile(r"<((?:https?://|mailto:)[^>\s]+)>", re.IGNORECASE)


class MarkdownDocumentParser:
    """Parse Markdown while keeping the original normalized text and offsets."""

    def __init__(self) -> None:
        self._parser = MarkdownIt("commonmark", {"html": True}).enable("table")

    def parse(self, document: RawDocument) -> ParsedDocument:
        """Convert a raw Markdown document into the canonical parsed model."""

        content, source_map = normalize_newlines_with_map(document.content)
        starts = line_starts(content)
        tokens = self._parser.parse(content)

        headings: list[Heading] = []
        code_blocks: list[CodeBlock] = []
        tables: list[TableBlock] = []

        for index, token in enumerate(tokens):
            if token.type == "heading_open" and token.map is not None:
                start, mapped_end = line_range_to_offsets(starts, token.map, len(content))
                # A heading location covers its source line, excluding the newline itself.
                end = mapped_end
                while end > start and content[end - 1] in "\r\n":
                    end -= 1
                inline_text = tokens[index + 1].content if index + 1 < len(tokens) else ""
                headings.append(
                    Heading(
                        level=int(token.tag[1]),
                        text=inline_text.strip(),
                        start_offset=start,
                        end_offset=end,
                    )
                )
            elif token.type in {"fence", "code_block"} and token.map is not None:
                start, end = line_range_to_offsets(starts, token.map, len(content))
                info_parts = token.info.strip().split(maxsplit=1)
                code_blocks.append(
                    CodeBlock(
                        language=info_parts[0] if info_parts else None,
                        content=content[start:end],
                        start_offset=start,
                        end_offset=end,
                    )
                )
            elif token.type == "table_open" and token.map is not None:
                start, end = line_range_to_offsets(starts, token.map, len(content))
                tables.append(
                    TableBlock(
                        content=content[start:end],
                        start_offset=start,
                        end_offset=end,
                    )
                )

        metadata = dict(document.metadata)
        metadata.update(
            {
                "parser": "markdown",
                "raw_content_sha256": sha256(document.content.encode("utf-8")).hexdigest(),
            }
        )
        return ParsedDocument(
            source_type=document.source_type,
            external_id=document.external_id,
            document_type=infer_document_type(document),
            title=document.title,
            content=content,
            canonical_url=document.canonical_url,
            source_version=document.source_version,
            updated_at=document.updated_at,
            headings=headings,
            code_blocks=code_blocks,
            tables=tables,
            links=self._extract_links(content),
            source_map=source_map,
            metadata=metadata,
        )

    @staticmethod
    def _extract_links(content: str) -> list[DocumentLink]:
        links = [
            DocumentLink(text=match.group(1), target=match.group(2), start_offset=match.start())
            for match in _MARKDOWN_LINK_RE.finditer(content)
        ]
        links.extend(
            DocumentLink(text=match.group(1), target=match.group(1), start_offset=match.start())
            for match in _AUTOLINK_RE.finditer(content)
        )
        return sorted(links, key=lambda link: link.start_offset)


# Concise alias used by ingestion service wiring.
MarkdownParser = MarkdownDocumentParser
