"""YAML and JSON parser that treats indentation-sensitive nodes as atomic blocks."""

from __future__ import annotations

from hashlib import sha256

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode

from app.domain.documents import CodeBlock, Heading, ParsedDocument, RawDocument
from app.ingestion.parsers.common import infer_document_type, normalize_newlines_with_map


class StructuredTextParser:
    """Parse top-level YAML/JSON keys without changing source whitespace."""

    def parse(self, document: RawDocument) -> ParsedDocument:
        content, source_map = normalize_newlines_with_map(document.content)
        source_format = str(document.metadata.get("format", "yaml")).casefold()
        language = "json" if source_format == "json" else "yaml"
        headings: list[Heading] = []
        code_blocks: list[CodeBlock] = []
        try:
            roots = list(yaml.compose_all(content))
        except yaml.YAMLError:
            roots = []

        for root in roots:
            if isinstance(root, MappingNode):
                self._mapping_blocks(root, content, language, headings, code_blocks)
        if not code_blocks and content:
            code_blocks.append(
                CodeBlock(
                    language=language,
                    content=content,
                    start_offset=0,
                    end_offset=len(content),
                )
            )

        metadata = dict(document.metadata)
        metadata.update(
            {
                "parser": language,
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
            source_map=source_map,
            metadata=metadata,
        )

    @staticmethod
    def _mapping_blocks(
        root: MappingNode,
        content: str,
        language: str,
        headings: list[Heading],
        code_blocks: list[CodeBlock],
    ) -> None:
        pairs: list[tuple[ScalarNode, Node]] = [
            (key, value)
            for key, value in root.value
            if isinstance(key, ScalarNode) and isinstance(key.value, str)
        ]
        for index, (key, _value) in enumerate(pairs):
            start = key.start_mark.index
            next_start = pairs[index + 1][0].start_mark.index if index + 1 < len(pairs) else None
            end = next_start if next_start is not None else len(content)
            headings.append(
                Heading(
                    level=1,
                    text=key.value,
                    start_offset=start,
                    end_offset=key.end_mark.index,
                )
            )
            code_blocks.append(
                CodeBlock(
                    language=language,
                    content=content[start:end],
                    start_offset=start,
                    end_offset=end,
                )
            )


YAMLParser = StructuredTextParser
