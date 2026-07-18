"""Reproducible evaluation-dataset generation from indexed source chunks.

The builder is intentionally provider-free: it creates evidence-grounded candidate
questions from curated chunk seeds when present and conservative extractive candidates
otherwise.  Generated rows are always marked as awaiting human review.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select

from app.core.config import Settings
from app.db.models import Chunk, Document
from app.db.session import Database
from app.domain.evaluation import ConversationTurn, Difficulty, EvaluationCase

REQUIRED_CATEGORIES = frozenset(
    {
        "factual",
        "how_to",
        "api_field",
        "version_difference",
        "release_change",
        "issue_status",
        "multi_hop",
        "multi_turn_reference",
        "ambiguous",
        "unanswerable",
        "out_of_scope",
        "hallucination_trap",
        "similar_api_distinction",
    }
)


class DatasetBuildError(ValueError):
    """Raised when source chunks cannot produce a valid evaluation dataset."""


class QuestionSeed(BaseModel):
    """Optional human-authored seed attached to a source chunk."""

    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    category: str = Field(min_length=1)
    difficulty: Difficulty = Difficulty.MEDIUM
    answerable: bool = True
    source_chunk_ids: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceChunk(BaseModel):
    """Portable chunk record accepted from JSONL or the PostgreSQL index."""

    chunk_id: str = Field(min_length=1)
    document_id: str | None = None
    title: str = Field(min_length=1)
    section: str = ""
    content: str = Field(min_length=1)
    url: str | None = None
    document_type: str = "official_documentation"
    source_type: str = "official_documentation"
    metadata: dict[str, Any] = Field(default_factory=dict)
    qa_candidates: list[QuestionSeed] = Field(default_factory=list)


class _Candidate(BaseModel):
    question: str
    reference_answer: str
    category: str
    difficulty: Difficulty
    answerable: bool
    source_chunk_ids: list[str]
    expected_citations: list[str]
    conversation_history: list[ConversationTurn]
    source_type: str | None
    metadata: dict[str, Any]


def load_source_chunks(path: Path) -> list[SourceChunk]:
    """Load portable source chunks and reject malformed or duplicate identifiers."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetBuildError(f"Unable to read source chunks: {path}") from exc

    chunks: list[SourceChunk] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            chunk = SourceChunk.model_validate_json(line)
        except ValidationError as exc:
            raise DatasetBuildError(f"Invalid source chunk at line {line_number}: {exc}") from exc
        if chunk.chunk_id in seen:
            raise DatasetBuildError(f"Duplicate source chunk id: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
        chunks.append(chunk)
    if not chunks:
        raise DatasetBuildError("No source chunks were supplied")
    return chunks


async def load_source_chunks_from_database(
    database_url: str,
    *,
    seed: int,
    limit: int,
) -> list[SourceChunk]:
    """Sample active chunks from PostgreSQL in a reproducible client-side order."""
    if limit < 1:
        raise DatasetBuildError("Database chunk limit must be positive")
    settings = Settings(database_url=database_url)
    database = Database(settings)
    try:
        async with database.session_factory() as session:
            statement = (
                select(Chunk, Document)
                .join(Document, Chunk.document_id == Document.id)
                .where(Chunk.deleted_at.is_(None), Document.deleted_at.is_(None))
                .order_by(Chunk.id)
                .limit(max(limit * 5, limit))
            )
            rows = list((await session.execute(statement)).all())
    finally:
        await database.close()

    rng = random.Random(seed)
    rng.shuffle(rows)
    sampled = rows[:limit]
    return [
        SourceChunk(
            chunk_id=str(chunk.id),
            document_id=str(document.id),
            title=document.title,
            section=" > ".join(chunk.heading_path),
            content=chunk.content,
            url=document.canonical_url,
            document_type=document.document_type,
            source_type=str(document.metadata_.get("source_type", document.document_type)),
            metadata={
                **chunk.metadata_,
                "source_version": document.source_version,
                "repository_path": document.repository_path,
            },
        )
        for chunk, document in sampled
    ]


def load_database_chunks(database_url: str, *, seed: int, limit: int) -> list[SourceChunk]:
    """Synchronous CLI bridge for :func:`load_source_chunks_from_database`."""
    return asyncio.run(load_source_chunks_from_database(database_url, seed=seed, limit=limit))


def _normalize_question(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    return normalized.rstrip("?.!\u3002\uff1f\uff01")


def _first_evidence_sentence(content: str, *, maximum: int = 500) -> str:
    compact = re.sub(r"\s+", " ", content).strip()
    if not compact:
        return "The indexed source does not contain a usable answer."
    matches = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+", compact)
    answer = " ".join(matches[:2]).strip()
    if len(answer) <= maximum:
        return answer
    return answer[: maximum - 1].rstrip() + "…"


def _chunk_snapshot(chunk: SourceChunk) -> dict[str, Any]:
    """Embed enough immutable evidence to audit how a candidate was produced."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "section": chunk.section,
        "url": chunk.url,
        "document_type": chunk.document_type,
        "content": chunk.content,
        "content_sha256": hashlib.sha256(chunk.content.encode("utf-8")).hexdigest(),
    }


def _seed_candidates(chunks: Sequence[SourceChunk]) -> list[_Candidate]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    candidates: list[_Candidate] = []
    for primary in chunks:
        for seed in primary.qa_candidates:
            source_ids = seed.source_chunk_ids or ([primary.chunk_id] if seed.answerable else [])
            missing = sorted(set(source_ids).difference(by_id))
            if missing:
                raise DatasetBuildError(
                    f"Question seed for {primary.chunk_id} refers to unknown chunks: {missing}"
                )
            relevant_ids = list(dict.fromkeys(source_ids)) if seed.answerable else []
            expected = seed.expected_citations or relevant_ids
            snapshots = [_chunk_snapshot(by_id[item]) for item in source_ids]
            metadata = {
                **seed.metadata,
                "source_chunk": _chunk_snapshot(primary),
                "source_chunks": snapshots,
                "generation_method": "source_seed",
                "review_status": "pending",
            }
            if not seed.answerable and source_ids:
                metadata["hard_negative_source_chunks"] = snapshots
            candidates.append(
                _Candidate(
                    question=seed.question.strip(),
                    reference_answer=seed.reference_answer.strip(),
                    category=seed.category,
                    difficulty=seed.difficulty,
                    answerable=seed.answerable,
                    source_chunk_ids=relevant_ids,
                    expected_citations=(list(dict.fromkeys(expected)) if seed.answerable else []),
                    conversation_history=seed.conversation_history,
                    source_type=primary.source_type if seed.answerable else None,
                    metadata=metadata,
                )
            )
    return candidates


def _extractive_candidates(chunks: Sequence[SourceChunk]) -> list[_Candidate]:
    """Create conservative candidates when no curated seed is available."""
    candidates: list[_Candidate] = []
    for chunk in chunks:
        reference = _first_evidence_sentence(chunk.content)
        section = chunk.section or "the indexed section"
        snapshot = _chunk_snapshot(chunk)
        templates = (
            (
                f"What does {chunk.title} state in {section}?",
                "factual",
                Difficulty.EASY,
            ),
            (
                f"How should a user apply the guidance in {chunk.title}, {section}?",
                "how_to",
                Difficulty.MEDIUM,
            ),
            (
                f"Which documented detail in {chunk.title} is important for this task?",
                "factual",
                Difficulty.MEDIUM,
            ),
        )
        for question, category, difficulty in templates:
            candidates.append(
                _Candidate(
                    question=question,
                    reference_answer=reference,
                    category=category,
                    difficulty=difficulty,
                    answerable=True,
                    source_chunk_ids=[chunk.chunk_id],
                    expected_citations=[chunk.chunk_id],
                    conversation_history=[],
                    source_type=chunk.source_type,
                    metadata={
                        "source_chunk": snapshot,
                        "source_chunks": [snapshot],
                        "generation_method": "extractive_template",
                        "review_status": "pending",
                    },
                )
            )
        candidates.append(
            _Candidate(
                question=f"For {chunk.title}, what exact result will occur at the next incident?",
                reference_answer=(
                    "The indexed evidence does not predict a future incident or its exact result."
                ),
                category="unanswerable",
                difficulty=Difficulty.HARD,
                answerable=False,
                source_chunk_ids=[],
                expected_citations=[],
                conversation_history=[],
                source_type=None,
                metadata={
                    "hard_negative_source_chunks": [snapshot],
                    "negative_reason": "asks for an unknowable future outcome",
                    "generation_method": "hard_negative_template",
                    "review_status": "pending",
                },
            )
        )
        pronoun_question = f"For it, what does the {section} section say?"
        candidates.append(
            _Candidate(
                question=pronoun_question,
                reference_answer=reference,
                category="multi_turn_reference",
                difficulty=Difficulty.MEDIUM,
                answerable=True,
                source_chunk_ids=[chunk.chunk_id],
                expected_citations=[chunk.chunk_id],
                conversation_history=[
                    ConversationTurn(role="user", content=f"Tell me about {chunk.title}."),
                    ConversationTurn(
                        role="assistant",
                        content=f"We are discussing {chunk.title}.",
                    ),
                ],
                source_type=chunk.source_type,
                metadata={
                    "source_chunk": snapshot,
                    "source_chunks": [snapshot],
                    "generation_method": "multi_turn_template",
                    "review_status": "pending",
                },
            )
        )

    if len(chunks) >= 2:
        for left, right in pairwise(chunks):
            left_snapshot = _chunk_snapshot(left)
            right_snapshot = _chunk_snapshot(right)
            candidates.append(
                _Candidate(
                    question=(
                        f"How do the documented details in {left.title} and {right.title} "
                        "fit together?"
                    ),
                    reference_answer=(
                        f"{_first_evidence_sentence(left.content, maximum=260)} "
                        f"{_first_evidence_sentence(right.content, maximum=260)}"
                    ),
                    category="multi_hop",
                    difficulty=Difficulty.HARD,
                    answerable=True,
                    source_chunk_ids=[left.chunk_id, right.chunk_id],
                    expected_citations=[left.chunk_id, right.chunk_id],
                    conversation_history=[],
                    source_type=left.source_type,
                    metadata={
                        "source_chunk": left_snapshot,
                        "source_chunks": [left_snapshot, right_snapshot],
                        "generation_method": "multi_hop_template",
                        "review_status": "pending",
                    },
                )
            )
    return candidates


def _global_negative_candidates(chunks: Sequence[SourceChunk]) -> list[_Candidate]:
    snapshot = _chunk_snapshot(chunks[0])
    definitions = (
        (
            "How do I configure PostgreSQL physical replication slots?",
            "out_of_scope",
            "PostgreSQL administration is outside this Kubernetes knowledge base.",
            "knowledge-base scope mismatch",
        ),
        (
            "What production value should I use for it?",
            "ambiguous",
            "The question does not identify a Kubernetes resource, field, or version.",
            "missing referent and operating context",
        ),
        (
            "Which Kubernetes release guarantees that every failed rollout repairs itself?",
            "hallucination_trap",
            "The indexed evidence contains no such guarantee.",
            "premise is not supported by the evidence",
        ),
        (
            "Are all Kubernetes workload API objects interchangeable in every situation?",
            "similar_api_distinction",
            (
                "No indexed evidence establishes that different workload API objects "
                "are interchangeable."
            ),
            "false equivalence between similar API objects",
        ),
    )
    return [
        _Candidate(
            question=question,
            reference_answer=answer,
            category=category,
            difficulty=Difficulty.HARD,
            answerable=False,
            source_chunk_ids=[],
            expected_citations=[],
            conversation_history=[],
            source_type=None,
            metadata={
                "hard_negative_source_chunks": [snapshot],
                "negative_reason": reason,
                "generation_method": "global_negative_template",
                "review_status": "pending",
            },
        )
        for question, category, answer, reason in definitions
    ]


def _category_gap_candidates(
    chunks: Sequence[SourceChunk], existing_categories: set[str]
) -> list[_Candidate]:
    """Add evidence-safe negatives for source types absent from a small input sample."""
    missing_definitions = {
        "api_field": (
            "Which exact Kubernetes API field configures {title}?",
            "The supplied source chunk does not identify an exact API field for that request.",
            "input sample contains no API-reference seed",
        ),
        "issue_status": (
            "What is the current GitHub Issue number and status for {title}?",
            "The supplied source chunk does not identify a GitHub Issue or its current status.",
            "input sample contains no issue-status seed",
        ),
        "release_change": (
            "Which Kubernetes release first introduced the behavior in {title}?",
            "The supplied source chunk does not establish the release that first introduced it.",
            "input sample contains no release-note seed",
        ),
        "version_difference": (
            "How did {title} change between Kubernetes v1.29 and v1.30?",
            "The supplied source chunk does not establish that version comparison.",
            "input sample contains no version-comparison seed",
        ),
    }
    generated: list[_Candidate] = []
    for index, category in enumerate(sorted(REQUIRED_CATEGORIES - existing_categories)):
        definition = missing_definitions.get(category)
        if definition is None:
            continue
        chunk = chunks[index % len(chunks)]
        snapshot = _chunk_snapshot(chunk)
        question_template, answer, reason = definition
        generated.append(
            _Candidate(
                question=question_template.format(title=chunk.title),
                reference_answer=answer,
                category=category,
                difficulty=Difficulty.HARD,
                answerable=False,
                source_chunk_ids=[],
                expected_citations=[],
                conversation_history=[],
                source_type=None,
                metadata={
                    "hard_negative_source_chunks": [snapshot],
                    "negative_reason": reason,
                    "generation_method": "missing-source-type-negative",
                    "review_status": "pending",
                },
            )
        )
    return generated


def _deduplicate(candidates: Iterable[_Candidate]) -> list[_Candidate]:
    unique: list[_Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _normalize_question(candidate.question)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _select_balanced(
    candidates: Sequence[_Candidate], *, count: int, rng: random.Random
) -> list[_Candidate]:
    if count < 1:
        raise DatasetBuildError("Dataset count must be positive")
    if len(candidates) < count:
        raise DatasetBuildError(
            f"Only {len(candidates)} unique candidates are available; requested {count}. "
            "Supply more source chunks or reduce --count."
        )

    groups: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.category].append(candidate)
    missing = REQUIRED_CATEGORIES.difference(groups)
    if count >= len(REQUIRED_CATEGORIES) and missing:
        raise DatasetBuildError(
            "Cannot build a category-complete dataset; missing: " + ", ".join(sorted(missing))
        )

    selected: list[_Candidate] = []
    selected_ids: set[int] = set()
    if count >= len(REQUIRED_CATEGORIES):
        for category in sorted(REQUIRED_CATEGORIES):
            choice = rng.choice(groups[category])
            selected.append(choice)
            selected_ids.add(id(choice))

    remaining = [item for item in candidates if id(item) not in selected_ids]
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
    rng.shuffle(selected)
    return selected


def build_dataset(
    chunks: Sequence[SourceChunk],
    *,
    count: int = 52,
    seed: int = 20250717,
    id_prefix: str = "k8s",
) -> list[EvaluationCase]:
    """Build a balanced, deterministic set of unreviewed evaluation cases."""
    if not chunks:
        raise DatasetBuildError("No source chunks were supplied")
    seeded = _seed_candidates(chunks)
    if len(seeded) >= count and REQUIRED_CATEGORIES.issubset(
        {candidate.category for candidate in seeded}
    ):
        pool = seeded
    else:
        generated = [
            *seeded,
            *_extractive_candidates(chunks),
            *_global_negative_candidates(chunks),
        ]
        pool = [
            *generated,
            *_category_gap_candidates(
                chunks,
                {candidate.category for candidate in generated},
            ),
        ]
    unique = _deduplicate(pool)
    selected = _select_balanced(unique, count=count, rng=random.Random(seed))

    cases: list[EvaluationCase] = []
    width = max(3, len(str(count)))
    for index, candidate in enumerate(selected, start=1):
        metadata = {
            **candidate.metadata,
            "generator": "opensource-doc-intelligence/dataset-builder-v1",
            "random_seed": seed,
        }
        cases.append(
            EvaluationCase(
                id=f"{id_prefix}-{index:0{width}d}",
                question=candidate.question,
                conversation_history=candidate.conversation_history,
                reference_answer=candidate.reference_answer,
                relevant_chunk_ids=candidate.source_chunk_ids,
                expected_citations=candidate.expected_citations,
                answerable=candidate.answerable,
                category=candidate.category,
                difficulty=candidate.difficulty,
                source_type=candidate.source_type,
                metadata=metadata,
                human_reviewed=False,
            )
        )
    return cases
