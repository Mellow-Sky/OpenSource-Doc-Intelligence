"""Structure-preserving parser for reStructuredText documentation."""

from __future__ import annotations

import re
from hashlib import sha256

from app.domain.documents import (
    CodeBlock,
    DocumentLink,
    Heading,
    ParsedDocument,
    RawDocument,
    TableBlock,
)
from app.ingestion.parsers.common import infer_document_type, normalize_newlines_with_map

_UNDERLINE_RE = re.compile(r"^\s*([=\-~^\"'`:+*#<>_])\1{2,}\s*$")
_DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+(?:code|code-block)::\s*([\w.+-]+)?\s*$")
_RST_LINK_RE = re.compile(r"`([^`<>]+?)\s*<((?:https?://|mailto:)[^>]+)>`_", re.IGNORECASE)
_GRID_TABLE_RE = re.compile(r"^\s*\+(?:[-=]+\+)+\s*$")


class RSTDocumentParser:
    """Extract RST headings, code directives, tables, links, and source offsets."""

    def parse(self, document: RawDocument) -> ParsedDocument:
        content, source_map = normalize_newlines_with_map(document.content)
        lines = content.splitlines(keepends=True)
        starts = _line_starts(lines)
        headings: list[Heading] = []
        code_blocks: list[CodeBlock] = []
        tables: list[TableBlock] = []
        adornment_levels: dict[str, int] = {}

        for index in range(len(lines) - 1):
            title = lines[index].rstrip("\r\n")
            underline = lines[index + 1].rstrip("\r\n")
            match = _UNDERLINE_RE.fullmatch(underline)
            if not title.strip() or match is None:
                continue
            marker = match.group(1)
            level = adornment_levels.setdefault(marker, min(6, len(adornment_levels) + 1))
            headings.append(
                Heading(
                    level=level,
                    text=title.strip(),
                    start_offset=starts[index],
                    end_offset=starts[index + 1] + len(underline),
                )
            )

        index = 0
        while index < len(lines):
            directive = _DIRECTIVE_RE.match(lines[index].rstrip("\r\n"))
            literal = lines[index].rstrip().endswith("::") and directive is None
            if directive is None and not literal:
                index += 1
                continue
            block_start = index
            probe = index + 1
            while probe < len(lines) and not lines[probe].strip():
                probe += 1
            content_start = probe
            while probe < len(lines) and (
                not lines[probe].strip() or lines[probe].startswith((" ", "\t"))
            ):
                probe += 1
            if content_start < probe:
                end = starts[probe] if probe < len(starts) else len(content)
                code_blocks.append(
                    CodeBlock(
                        language=directive.group(1) if directive is not None else None,
                        content=content[starts[block_start] : end],
                        start_offset=starts[block_start],
                        end_offset=end,
                    )
                )
            index = max(index + 1, probe)

        index = 0
        while index < len(lines):
            if not _GRID_TABLE_RE.match(lines[index].rstrip("\r\n")):
                index += 1
                continue
            start = index
            index += 1
            while index < len(lines) and lines[index].lstrip().startswith(("+", "|")):
                index += 1
            end = starts[index] if index < len(starts) else len(content)
            tables.append(
                TableBlock(
                    content=content[starts[start] : end],
                    start_offset=starts[start],
                    end_offset=end,
                )
            )

        metadata = dict(document.metadata)
        metadata.update(
            {
                "parser": "rst",
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
            links=[
                DocumentLink(text=match.group(1), target=match.group(2), start_offset=match.start())
                for match in _RST_LINK_RE.finditer(content)
            ],
            source_map=source_map,
            metadata=metadata,
        )


def _line_starts(lines: list[str]) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)
    starts.append(cursor)
    return starts


RSTParser = RSTDocumentParser
