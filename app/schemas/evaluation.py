"""Validated HTTP contracts for durable offline evaluation jobs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.evaluation import EvaluationRunStatus

MAX_CONFIG_SNAPSHOT_BYTES = 65_536


class EvaluationRunCreateRequest(BaseModel):
    """A non-blocking request that a separate worker can execute later."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(min_length=1, max_length=255)
    dataset_path: str = Field(min_length=1, max_length=2_048)
    experiment_name: str = Field(default="default", min_length=1, max_length=255)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dataset_name", "experiment_name")
    @classmethod
    def normalize_names(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        if _has_control_characters(stripped):
            raise ValueError("value must not contain control characters")
        return stripped

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, value: str) -> str:
        """Constrain queued worker inputs to versioned JSONL evaluation datasets."""
        normalized = value.strip().replace("\\", "/")
        if not normalized or _has_control_characters(normalized):
            raise ValueError("dataset_path is invalid")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
            raise ValueError("dataset_path must be relative to evaluation/datasets")
        parts = path.parts
        if parts[:2] == ("evaluation", "datasets"):
            parts = parts[2:]
        if not parts or PurePosixPath(*parts).suffix.lower() != ".jsonl":
            raise ValueError("dataset_path must identify a .jsonl file")
        return str(PurePosixPath("evaluation", "datasets", *parts))

    @model_validator(mode="after")
    def validate_snapshot_size(self) -> EvaluationRunCreateRequest:
        encoded = json.dumps(
            self.config_snapshot,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        if len(encoded) > MAX_CONFIG_SNAPSHOT_BYTES:
            raise ValueError(f"config_snapshot cannot exceed {MAX_CONFIG_SNAPSHOT_BYTES} bytes")
        return self


class EvaluationRunResponse(BaseModel):
    """Polling representation of a durable evaluation run."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    dataset_name: str
    status: EvaluationRunStatus
    config_snapshot: dict[str, Any]
    started_at: datetime
    finished_at: datetime | None
    summary: dict[str, Any]
    result_count: int = Field(ge=0)
    report_available: bool


def _has_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)
