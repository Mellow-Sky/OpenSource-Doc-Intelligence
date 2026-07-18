"""Validated, reproducible JSONL dataset loading and atomic writing."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from app.domain.evaluation import EvaluationCase


class DatasetError(ValueError):
    """Raised when an evaluation dataset is malformed or ambiguous."""


def load_dataset(path: Path) -> list[EvaluationCase]:
    """Load JSONL cases, rejecting malformed lines and duplicate identifiers."""
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"Unable to read dataset: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = EvaluationCase.model_validate_json(line)
        except ValidationError as exc:
            raise DatasetError(f"Invalid dataset row at line {line_number}: {exc}") from exc
        if case.id in seen_ids:
            raise DatasetError(f"Duplicate evaluation case id: {case.id}")
        seen_ids.add(case.id)
        cases.append(case)
    if not cases:
        raise DatasetError("Evaluation dataset is empty")
    return cases


def write_dataset(path: Path, cases: Iterable[EvaluationCase]) -> int:
    """Atomically write unique cases in deterministic JSONL form."""
    materialized = list(cases)
    identifiers = [case.id for case in materialized]
    if len(set(identifiers)) != len(identifiers):
        raise DatasetError("Evaluation case ids must be unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        case.model_dump_json(exclude_none=True) + "\n"
        for case in sorted(materialized, key=lambda item: item.id)
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return len(materialized)


def dataset_fingerprint(cases: Iterable[EvaluationCase]) -> str:
    """Return a stable SHA-256 over canonical JSON rows."""
    import hashlib

    digest = hashlib.sha256()
    for case in sorted(cases, key=lambda item: item.id):
        canonical = json.dumps(
            case.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
