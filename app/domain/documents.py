"""Canonical documents exchanged between ingestion pipeline stages."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class DocumentType(StrEnum):
    """Supported logical document categories."""

    OFFICIAL_DOCUMENTATION = "official_documentation"
    GITHUB_ISSUE = "github_issue"
    RELEASE_NOTE = "release_note"
    API_REFERENCE = "api_reference"
    REPOSITORY_DOCUMENT = "repository_document"
    KEP = "kep"
    BLOG = "blog"


class RawDocument(BaseModel):
    """Provider-neutral representation returned by every source loader."""

    model_config = ConfigDict(extra="forbid")

    source_type: str = Field(min_length=1, max_length=64)
    external_id: str = Field(min_length=1, max_length=1024)
    title: str = Field(min_length=1, max_length=2000)
    content: str
    canonical_url: str | None = None
    source_version: str | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Heading(BaseModel):
    """Heading captured from a parsed source document."""

    level: int = Field(ge=1, le=6)
    text: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)


class CodeBlock(BaseModel):
    """A fenced or structural code block with offsets into normalized text."""

    language: str | None = None
    content: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)


class TableBlock(BaseModel):
    """A table retained as one structural element."""

    content: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)


class DocumentLink(BaseModel):
    """A link discovered while parsing the source."""

    text: str
    target: str
    start_offset: int = Field(ge=0)


class SourceMapEntry(BaseModel):
    """Mapping from normalized offsets back to source line and character positions."""

    normalized_start: int = Field(ge=0)
    normalized_end: int = Field(ge=0)
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=0)
    source_start_line: int | None = Field(default=None, ge=1)
    source_end_line: int | None = Field(default=None, ge=1)


class ParsedDocument(BaseModel):
    """Normalized document that preserves structural and source-location metadata."""

    model_config = ConfigDict(extra="forbid")

    source_type: str
    external_id: str
    document_type: DocumentType
    title: str
    content: str
    canonical_url: HttpUrl | None = None
    source_version: str | None = None
    updated_at: datetime | None = None
    headings: list[Heading] = Field(default_factory=list)
    code_blocks: list[CodeBlock] = Field(default_factory=list)
    tables: list[TableBlock] = Field(default_factory=list)
    links: list[DocumentLink] = Field(default_factory=list)
    source_map: list[SourceMapEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
