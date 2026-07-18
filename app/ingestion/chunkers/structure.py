"""Heading-first, structure-aware token chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from app.domain.chunks import ChunkDraft, SourcePosition
from app.domain.documents import ParsedDocument, SourceMapEntry
from app.ingestion.deduplication.content import normalized_content_hash

_TOKEN_RE = re.compile(r"[\w]+(?:[./:@-][\w]+)*|[^\w\s]", re.UNICODE)
_PARAGRAPH_RE = re.compile(r"\S[\s\S]*?(?=\n[ \t]*\n|\Z)")


class TokenCounter(Protocol):
    """Minimal adapter implemented by model-specific tokenizer wrappers."""

    def count(self, text: str) -> int:
        """Return the model token count for text."""


class RegexTokenCounter:
    """Deterministic fallback that preserves technical identifiers as units."""

    def count(self, text: str) -> int:
        return len(_TOKEN_RE.findall(text))


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Token limits for child chunks; atomic code and tables may exceed max_tokens."""

    target_tokens: int = 500
    max_tokens: int = 800
    overlap_tokens: int = 80
    min_tokens: int = 80

    def __post_init__(self) -> None:
        if self.target_tokens < 1:
            raise ValueError("target_tokens must be positive")
        if self.max_tokens < self.target_tokens:
            raise ValueError("max_tokens must be at least target_tokens")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens must be non-negative and less than max_tokens")
        if not 1 <= self.min_tokens <= self.target_tokens:
            raise ValueError("min_tokens must be between 1 and target_tokens")


@dataclass(frozen=True, slots=True)
class _Interval:
    start: int
    end: int
    atomic: bool = False


@dataclass(frozen=True, slots=True)
class _Section:
    start: int
    end: int
    heading_path: tuple[str, ...]
    heading_level: int | None


class StructureAwareChunker:
    """Split by headings and paragraphs while never cutting code blocks or tables."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.token_counter = token_counter or RegexTokenCounter()

    def chunk(self, document: ParsedDocument) -> list[ChunkDraft]:
        """Create citation-ready chunks with hierarchy, overlap, and exact offsets."""

        if not document.content.strip():
            return []
        atomic = self._atomic_intervals(document)
        drafts: list[ChunkDraft] = []
        ancestor_first_chunk: dict[int, int] = {}

        for section in self._sections(document):
            section_content = document.content[section.start : section.end]
            if section.start >= section.end or not section_content.strip():
                continue
            parent_index = self._ancestor_index(section.heading_level, ancestor_first_chunk)
            prefix = self._context_prefix(document.title, section.heading_path)
            units = self._units(document.content, section, atomic, prefix)
            core_spans = self._pack(document.content, units, prefix)
            spans = self._add_overlap(document.content, core_spans, section, atomic, prefix)
            first_section_index = len(drafts)

            for section_chunk_index, span in enumerate(spans):
                content = document.content[span.start : span.end].strip("\n")
                if not content.strip():
                    continue
                contextualized = f"{prefix}{content}"
                index = len(drafts)
                resolved_parent = parent_index if section_chunk_index == 0 else first_section_index
                source_start = self._map_source_offset(span.start, document.source_map)
                source_end = self._map_source_offset(span.end, document.source_map)
                source_start_line = self._map_source_line(span.start, document.source_map)
                source_end_line = self._map_source_line(
                    max(span.start, span.end - 1), document.source_map
                )
                is_oversized_atomic = self.token_counter.count(
                    contextualized
                ) > self.config.max_tokens and self._covered_by_one_atomic(span, atomic)
                previous_end = (
                    spans[section_chunk_index - 1].end if section_chunk_index else span.start
                )
                overlap_count = (
                    self.token_counter.count(
                        document.content[span.start : min(previous_end, span.end)]
                    )
                    if section_chunk_index
                    else 0
                )
                drafts.append(
                    ChunkDraft(
                        chunk_index=index,
                        parent_index=resolved_parent,
                        heading_path=list(section.heading_path),
                        content=content,
                        contextualized_content=contextualized,
                        token_count=max(1, self.token_counter.count(contextualized)),
                        # The embedding payload includes title/heading context, so the
                        # incremental fingerprint must cover that context as well.
                        content_hash=normalized_content_hash(contextualized),
                        position=SourcePosition(
                            start_offset=source_start,
                            end_offset=source_end,
                            start_line=source_start_line
                            or self._line_number(document.content, span.start),
                            end_line=source_end_line
                            or self._line_number(document.content, max(span.start, span.end - 1)),
                        ),
                        metadata={
                            "document_external_id": document.external_id,
                            "document_type": document.document_type.value,
                            "source_type": document.source_type,
                            "source_start_offset": source_start,
                            "source_end_offset": source_end,
                            "normalized_start_offset": span.start,
                            "normalized_end_offset": span.end,
                            "heading_level": section.heading_level,
                            "chunk_role": (
                                "section_root" if section_chunk_index == 0 else "continuation"
                            ),
                            "overlap_tokens": overlap_count,
                            "oversized_atomic": is_oversized_atomic,
                        },
                    )
                )

            if section.heading_level is not None and len(drafts) > first_section_index:
                for level in tuple(ancestor_first_chunk):
                    if level >= section.heading_level:
                        del ancestor_first_chunk[level]
                ancestor_first_chunk[section.heading_level] = first_section_index
        return drafts

    @staticmethod
    def _context_prefix(title: str, heading_path: tuple[str, ...]) -> str:
        path = list(heading_path)
        if path and path[0].strip().casefold() == title.strip().casefold():
            path = path[1:]
        prefix = f"Document: {title}\n"
        if path:
            prefix += f"Section: {' > '.join(path)}\n"
        return f"{prefix}\n"

    @staticmethod
    def _sections(document: ParsedDocument) -> list[_Section]:
        headings = sorted(document.headings, key=lambda heading: heading.start_offset)
        if not headings:
            return [_Section(0, len(document.content), (), None)]

        sections: list[_Section] = []
        if headings[0].start_offset > 0 and document.content[: headings[0].start_offset].strip():
            sections.append(_Section(0, headings[0].start_offset, (), None))

        stack: list[tuple[int, str]] = []
        for index, heading in enumerate(headings):
            while stack and stack[-1][0] >= heading.level:
                stack.pop()
            stack.append((heading.level, heading.text))
            end = (
                headings[index + 1].start_offset
                if index + 1 < len(headings)
                else len(document.content)
            )
            sections.append(
                _Section(
                    heading.start_offset,
                    end,
                    tuple(text for _, text in stack),
                    heading.level,
                )
            )
        return sections

    @staticmethod
    def _atomic_intervals(document: ParsedDocument) -> list[_Interval]:
        intervals: list[_Interval] = [
            _Interval(block.start_offset, block.end_offset, atomic=True)
            for block in document.code_blocks
            if block.end_offset > block.start_offset
        ]
        intervals.extend(
            _Interval(table.start_offset, table.end_offset, atomic=True)
            for table in document.tables
            if table.end_offset > table.start_offset
        )
        intervals.sort(key=lambda interval: (interval.start, interval.end))
        merged: list[_Interval] = []
        for interval in intervals:
            if merged and interval.start < merged[-1].end:
                previous = merged[-1]
                merged[-1] = _Interval(previous.start, max(previous.end, interval.end), atomic=True)
            else:
                merged.append(interval)
        return merged

    def _units(
        self,
        content: str,
        section: _Section,
        atomic: list[_Interval],
        prefix: str,
    ) -> list[_Interval]:
        # Reserve room for the configured overlap on all continuation chunks.
        content_target = min(
            self.config.target_tokens,
            self.config.max_tokens - self.config.overlap_tokens,
        )
        budget = max(1, content_target - self.token_counter.count(prefix))
        units: list[_Interval] = []
        cursor = section.start
        for protected in atomic:
            if protected.end <= section.start or protected.start >= section.end:
                continue
            protected_start = max(section.start, protected.start)
            protected_end = min(section.end, protected.end)
            if cursor < protected_start:
                units.extend(self._normal_units(content, cursor, protected_start, budget))
            units.append(_Interval(protected_start, protected_end, atomic=True))
            cursor = protected_end
        if cursor < section.end:
            units.extend(self._normal_units(content, cursor, section.end, budget))
        return sorted(units, key=lambda unit: (unit.start, unit.end))

    def _normal_units(self, content: str, start: int, end: int, budget: int) -> list[_Interval]:
        segment = content[start:end]
        units: list[_Interval] = []
        for match in _PARAGRAPH_RE.finditer(segment):
            unit_start = start + match.start()
            unit_end = start + match.end()
            if self.token_counter.count(content[unit_start:unit_end]) <= budget:
                units.append(_Interval(unit_start, unit_end))
            else:
                units.extend(self._split_token_span(content, unit_start, unit_end, budget))
        return units

    def _split_token_span(self, content: str, start: int, end: int, budget: int) -> list[_Interval]:
        matches = list(_TOKEN_RE.finditer(content, start, end))
        if not matches:
            return []
        intervals: list[_Interval] = []
        token_index = 0
        while token_index < len(matches):
            next_index = min(len(matches), token_index + budget)
            interval_start = matches[token_index].start()
            interval_end = matches[next_index - 1].end()
            # Custom token counters can disagree with the fallback boundary estimator.
            while (
                next_index > token_index + 1
                and self.token_counter.count(content[interval_start:interval_end]) > budget
            ):
                next_index -= 1
                interval_end = matches[next_index - 1].end()
            intervals.append(_Interval(interval_start, interval_end))
            token_index = next_index
        return intervals

    def _pack(self, content: str, units: list[_Interval], prefix: str) -> list[_Interval]:
        if not units:
            return []
        spans: list[_Interval] = []
        current = units[0]
        for unit in units[1:]:
            current_text = f"{prefix}{content[current.start : current.end]}"
            candidate_text = f"{prefix}{content[current.start : unit.end]}"
            current_tokens = self.token_counter.count(current_text)
            candidate_tokens = self.token_counter.count(candidate_text)
            if current_tokens >= self.config.target_tokens or (
                candidate_tokens > self.config.max_tokens and current.end > current.start
            ):
                spans.append(current)
                current = unit
            else:
                current = _Interval(current.start, unit.end, current.atomic and unit.atomic)
        spans.append(current)

        if len(spans) >= 2:
            tail_tokens = self.token_counter.count(
                f"{prefix}{content[spans[-1].start : spans[-1].end]}"
            )
            merged_tokens = self.token_counter.count(
                f"{prefix}{content[spans[-2].start : spans[-1].end]}"
            )
            if tail_tokens < self.config.min_tokens and merged_tokens <= self.config.max_tokens:
                spans[-2:] = [_Interval(spans[-2].start, spans[-1].end)]
        return spans

    def _add_overlap(
        self,
        content: str,
        core_spans: list[_Interval],
        section: _Section,
        atomic: list[_Interval],
        prefix: str,
    ) -> list[_Interval]:
        if self.config.overlap_tokens == 0 or len(core_spans) < 2:
            return core_spans
        spans = [core_spans[0]]
        for core in core_spans[1:]:
            previous = spans[-1]
            desired = self._tail_start(content, section.start, previous.end)
            desired = min(desired, core.start)
            for protected in atomic:
                if protected.start < desired < protected.end:
                    desired = protected.start
                    break
            start = self._earliest_fitting_start(
                content, desired, core.start, core.end, prefix, atomic
            )
            spans.append(_Interval(start, core.end, core.atomic))
        return spans

    def _tail_start(self, content: str, lower_bound: int, end: int) -> int:
        tokens = list(_TOKEN_RE.finditer(content, lower_bound, end))
        if not tokens:
            return end
        index = max(0, len(tokens) - self.config.overlap_tokens)
        return tokens[index].start()

    def _earliest_fitting_start(
        self,
        content: str,
        desired: int,
        core_start: int,
        end: int,
        prefix: str,
        atomic: list[_Interval],
    ) -> int:
        candidates = [desired]
        candidates.extend(
            match.start() for match in _TOKEN_RE.finditer(content, desired, core_start)
        )
        candidates.append(core_start)
        for candidate in sorted(set(candidates)):
            adjusted = candidate
            for protected in atomic:
                if protected.start < adjusted < protected.end:
                    adjusted = protected.end
                    break
            if adjusted > core_start:
                continue
            candidate_text = f"{prefix}{content[adjusted:end]}"
            if self.token_counter.count(candidate_text) <= self.config.max_tokens:
                return adjusted
        return core_start

    @staticmethod
    def _ancestor_index(level: int | None, ancestors: dict[int, int]) -> int | None:
        if level is None:
            return None
        eligible = [candidate for candidate in ancestors if candidate < level]
        return ancestors[max(eligible)] if eligible else None

    @staticmethod
    def _covered_by_one_atomic(span: _Interval, atomic: list[_Interval]) -> bool:
        return any(
            protected.start <= span.start and protected.end >= span.end for protected in atomic
        )

    @staticmethod
    def _line_number(content: str, offset: int) -> int:
        return content.count("\n", 0, offset) + 1

    @staticmethod
    def _map_source_offset(offset: int, source_map: list[SourceMapEntry]) -> int:
        if not source_map:
            return offset
        for entry in source_map:
            if entry.normalized_start <= offset < entry.normalized_end:
                normalized_span = max(1, entry.normalized_end - entry.normalized_start)
                source_span = max(0, entry.source_end - entry.source_start)
                relative = max(0, offset - entry.normalized_start)
                return entry.source_start + round(relative / normalized_span * source_span)
        return source_map[-1].source_end

    @staticmethod
    def _map_source_line(offset: int, source_map: list[SourceMapEntry]) -> int | None:
        for entry in source_map:
            if entry.normalized_start <= offset < entry.normalized_end:
                return entry.source_start_line
        return source_map[-1].source_end_line if source_map else None


StructuredChunker = StructureAwareChunker
