from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.dataset import load_dataset
from evaluation.dataset_builder import (
    REQUIRED_CATEGORIES,
    DatasetBuildError,
    SourceChunk,
    build_dataset,
    load_source_chunks,
)

PROJECT_ROOT = Path(__file__).parents[2]
CATALOG = PROJECT_ROOT / "evaluation/datasets/kubernetes_source_catalog.jsonl"


def test_sample_catalog_builds_reproducible_complete_dataset() -> None:
    chunks = load_source_chunks(CATALOG)

    first = build_dataset(chunks, count=52, seed=17)
    second = build_dataset(chunks, count=52, seed=17)

    assert first == second
    assert len(first) == 52
    assert len({case.id for case in first}) == 52
    assert len({case.question.casefold() for case in first}) == 52
    assert {case.category for case in first} == REQUIRED_CATEGORIES
    assert {case.answerable for case in first} == {True, False}
    assert all(not case.human_reviewed for case in first)
    assert all("source_chunk" in case.metadata for case in first if case.answerable)
    assert all(case.relevant_chunk_ids for case in first if case.answerable)
    assert all(not case.relevant_chunk_ids for case in first if not case.answerable)


def test_loader_rejects_duplicate_chunk_ids(tmp_path: Path) -> None:
    row = {
        "chunk_id": "same",
        "title": "Title",
        "content": "Evidence.",
    }
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(DatasetBuildError, match="Duplicate source chunk"):
        load_source_chunks(path)


def test_extractable_chunks_generate_hard_negatives_and_multi_turn() -> None:
    chunks = [
        SourceChunk(
            chunk_id=f"chunk-{index}",
            title=f"Kubernetes topic {index}",
            section="Configuration",
            content=f"Topic {index} has a documented behavior. It has a second fact.",
        )
        for index in range(12)
    ]

    cases = build_dataset(chunks, count=40, seed=9, id_prefix="fixture")

    assert any(case.category == "unanswerable" for case in cases)
    assert any(case.category == "multi_turn_reference" for case in cases)
    assert any(case.category == "multi_hop" for case in cases)
    assert any(case.conversation_history for case in cases)


def test_help_command_runs_without_external_services() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_eval_dataset.py", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--seed" in result.stdout
    assert "--database-url" in result.stdout


def test_offline_sample_cli_writes_valid_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "generated.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_eval_dataset.py",
            "--input",
            str(CATALOG),
            "--output",
            str(output),
            "--count",
            "52",
            "--seed",
            "17",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Wrote 52 unreviewed cases" in result.stdout
    assert len(load_dataset(output)) == 52
