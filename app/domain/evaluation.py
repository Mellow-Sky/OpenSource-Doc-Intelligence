"""Evaluation dataset and per-case result contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvaluationRunStatus(StrEnum):
    """Durable lifecycle shared by the API queue and evaluation worker."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ConversationTurn(BaseModel):
    role: str
    content: str


class EvaluationCase(BaseModel):
    """One reproducible JSONL evaluation item."""

    id: str
    question: str
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    reference_answer: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    answerable: bool
    category: str
    difficulty: Difficulty
    source_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    human_reviewed: bool = False


class JudgeScores(BaseModel):
    factual_correctness: int = Field(ge=0, le=5)
    completeness: int = Field(ge=0, le=5)
    relevance: int = Field(ge=0, le=5)
    groundedness: int = Field(ge=0, le=5)
    rationale: str


class EvaluationCaseResult(BaseModel):
    case_id: str
    generated_answer: str
    retrieved_chunk_ids: list[str]
    predicted_answerable: bool
    metrics: dict[str, float | int | None]
    judge: JudgeScores | None = None
    error: str | None = None
