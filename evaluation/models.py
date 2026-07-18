"""Transport-neutral contracts for evaluation execution and persisted reports."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.domain.evaluation import EvaluationCase, JudgeScores


class EvaluationEvidence(BaseModel):
    chunk_id: str
    document_id: str | None = None
    title: str
    section: str = ""
    document_type: str = "unknown"
    content: str
    score: float | None = None
    rank: int = Field(ge=1)


class EvaluationCitationMarker(BaseModel):
    """Auditable outcome for one citation marker in generated-answer order."""

    number: int | None = Field(default=None, ge=1)
    chunk_id: str | None = None
    valid: bool
    support_score: float = Field(ge=0, le=1)


class EvaluationCitationSummary(BaseModel):
    citation_ids: list[str] = Field(default_factory=list)
    markers: list[EvaluationCitationMarker] = Field(default_factory=list)
    validity: list[bool] = Field(default_factory=list)
    support_scores: list[float] = Field(default_factory=list)
    claim_requires_citation: list[bool] = Field(default_factory=list)
    claim_supported: list[bool] = Field(default_factory=list)


class EvaluationResponse(BaseModel):
    generated_answer: str
    predicted_answerable: bool
    rewritten_query: str
    evidence: list[EvaluationEvidence] = Field(default_factory=list)
    citations: EvaluationCitationSummary = Field(default_factory=EvaluationCitationSummary)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, int | float | None] = Field(default_factory=dict)


class EvaluationResultRecord(BaseModel):
    case: EvaluationCase
    generated_answer: str
    rewritten_query: str
    predicted_answerable: bool
    retrieved_evidence: list[EvaluationEvidence]
    metrics: dict[str, float | int | None]
    citations: EvaluationCitationSummary
    judge: JudgeScores | None = None
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_usage: dict[str, int] = Field(default_factory=dict)
    judge_estimated_cost_usd: float | None = Field(default=None, ge=0)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, int | float | None] = Field(default_factory=dict)
    error: str | None = None


class EvaluationRunReport(BaseModel):
    run_id: str
    experiment_name: str
    dataset_name: str
    dataset_path: str
    dataset_fingerprint: str
    dataset_size: int = Field(ge=0)
    started_at: datetime
    finished_at: datetime
    elapsed_seconds: float = Field(ge=0)
    config_snapshot: dict[str, Any]
    git_commit: str | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    summary: dict[str, Any]
    category_metrics: dict[str, dict[str, float | int | None]]
    difficulty_metrics: dict[str, dict[str, float | int | None]]
    answerability_groups: dict[str, dict[str, float | int | None]]
    results: list[EvaluationResultRecord]
