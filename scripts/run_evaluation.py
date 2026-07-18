"""Run reproducible offline RAG evaluation and write JSON, Markdown, JSONL, and CSV reports."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.engine import make_url

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.db.session import Database
from app.domain.evaluation import EvaluationCase
from app.ingestion.chunkers import create_llm_token_counter
from app.providers.base import EmbeddingProvider, LLMProvider, Provider, RerankerProvider
from app.providers.factory import (
    create_embedding_provider,
    create_judge_provider,
    create_llm_provider,
    create_reranker_provider,
)
from app.services.chat_service import ChatService
from app.services.pricing_service import PricingCatalog
from evaluation.adapters import ChatEvaluationExecutor
from evaluation.dataset import load_dataset
from evaluation.judges.llm import LLMEvaluationJudge
from evaluation.models import EvaluationRunReport
from evaluation.reporting import write_comparison, write_report
from evaluation.runner import EvaluationRunner, prompt_versions


class Experiment(BaseModel):
    """One validated evaluation configuration override."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    settings: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = Field(default=None, ge=1, le=100)


class ExperimentFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiments: list[Experiment] = Field(min_length=1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/kubernetes_eval.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation/reports/latest"))
    parser.add_argument("--experiment-name", default="configured")
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run keyword, vector, hybrid, reranked, raw-query, rewrite, and Top-K baselines",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace, *, settings: Settings | None = None) -> int:
    """Execute all requested experiments and return a process status."""
    runtime = settings or get_settings()
    cases = load_dataset(args.dataset)
    experiments = _experiments(args, runtime)
    commit = runtime.git_commit or _git_commit()
    versions = prompt_versions(runtime.prompt_directory)
    reports: list[EvaluationRunReport] = []
    for experiment in experiments:
        experiment_settings = _apply_overrides(runtime, experiment.settings)
        report = await _run_one(
            experiment,
            settings=experiment_settings,
            cases=cases,
            dataset_path=args.dataset,
            concurrency=args.concurrency,
            fail_fast=args.fail_fast,
            use_judge=not args.no_judge,
            git_commit=commit,
            versions=versions,
        )
        output = args.output / _slug(experiment.name) if len(experiments) > 1 else args.output
        paths = write_report(report, output)
        reports.append(report)
        print(
            json.dumps(
                {
                    "experiment": experiment.name,
                    "run_id": report.run_id,
                    "successful": report.summary["successful_case_count"],
                    "errors": report.summary["error_count"],
                    "report": str(paths["markdown"]),
                },
                ensure_ascii=False,
            )
        )
    if len(reports) > 1:
        comparison = write_comparison(reports, args.output)
        print(json.dumps({"comparison": str(comparison["markdown"])}, ensure_ascii=False))
    return int(any(report.summary["error_count"] for report in reports))


async def _run_one(
    experiment: Experiment,
    *,
    settings: Settings,
    cases: list[EvaluationCase],
    dataset_path: Path,
    concurrency: int,
    fail_fast: bool,
    use_judge: bool,
    git_commit: str | None,
    versions: dict[str, str],
) -> EvaluationRunReport:
    database = Database(settings)
    embedding: EmbeddingProvider | None = None
    reranker: RerankerProvider | None = None
    llm: LLMProvider | None = None
    judge_provider: LLMProvider | None = None
    providers: list[Provider] = []
    try:
        if settings.retrieval_mode in {"hybrid", "vector"}:
            embedding = create_embedding_provider(settings)
            providers.append(embedding)
        if settings.enable_reranker:
            reranker = create_reranker_provider(settings)
            providers.append(reranker)
        llm = create_llm_provider(settings)
        providers.append(llm)
        context_token_counter = await create_llm_token_counter(settings, llm)
        judge_provider = create_judge_provider(settings) if use_judge else None
        if judge_provider is not None:
            providers.append(judge_provider)
        service = ChatService(
            session_factory=database.session_factory,
            settings=settings,
            embedding_provider=embedding,
            reranker_provider=reranker,
            llm_provider=llm,
            context_token_counter=context_token_counter,
        )
        judge = (
            LLMEvaluationJudge(
                judge_provider,
                prompt_path=settings.prompt_directory / "eval_judge.md",
                pricing_catalog=PricingCatalog.from_file(settings.pricing_config_path),
                max_tokens=settings.judge_max_tokens,
                timeout_seconds=settings.judge_timeout_seconds,
            )
            if judge_provider is not None
            else None
        )
        runner = EvaluationRunner(
            ChatEvaluationExecutor(service, top_k=experiment.top_k),
            judge=judge,
            concurrency=concurrency,
            fail_fast=fail_fast,
        )
        return await runner.run(
            cases,
            experiment_name=experiment.name,
            dataset_name=dataset_path.stem,
            dataset_path=dataset_path,
            config_snapshot=_safe_snapshot(settings, experiment),
            git_commit=git_commit,
            prompt_versions=versions,
        )
    finally:
        closed: set[int] = set()
        for provider in providers:
            if id(provider) not in closed:
                await provider.close()
                closed.add(id(provider))
        await database.close()


def _experiments(args: argparse.Namespace, settings: Settings) -> list[Experiment]:
    if args.experiment_config is not None:
        try:
            raw = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))
            return ExperimentFile.model_validate(raw).experiments
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            raise ConfigurationError(f"Invalid experiment file: {args.experiment_config}") from exc
    if args.compare:
        return [
            Experiment(
                name="keyword-only",
                settings={"retrieval_mode": "keyword", "enable_reranker": False},
            ),
            Experiment(
                name="vector-only",
                settings={"retrieval_mode": "vector", "enable_reranker": False},
            ),
            Experiment(
                name="hybrid",
                settings={"retrieval_mode": "hybrid", "enable_reranker": False},
            ),
            Experiment(
                name="hybrid-reranked",
                settings={"retrieval_mode": "hybrid", "enable_reranker": True},
            ),
            Experiment(
                name="raw-query",
                settings={"enable_query_rewrite": False},
            ),
            Experiment(
                name="rewritten-query",
                settings={"enable_query_rewrite": True},
            ),
            Experiment(name="top-k-5", top_k=5),
            Experiment(name="top-k-10", top_k=10),
            Experiment(name="top-k-20", top_k=20),
        ]
    return [Experiment(name=args.experiment_name)]


def _apply_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    unknown = sorted(set(overrides).difference(Settings.model_fields))
    if unknown:
        raise ConfigurationError(f"Unknown experiment settings: {', '.join(unknown)}")
    try:
        return Settings.model_validate({**settings.model_dump(), **overrides})
    except ValidationError as exc:
        raise ConfigurationError("Invalid experiment setting overrides") from exc


def _safe_snapshot(settings: Settings, experiment: Experiment) -> dict[str, Any]:
    snapshot = settings.model_dump(mode="json")
    secret_keys = {
        "api_key",
        "admin_api_key",
        "github_token",
        "llm_api_key",
        "embedding_api_key",
        "reranker_api_key",
        "judge_api_key",
    }
    for key in secret_keys:
        value = snapshot.get(key)
        snapshot[key] = None if value is None else "[REDACTED]"
    snapshot["database_url"] = make_url(settings.database_url).render_as_string(hide_password=True)
    snapshot["experiment_top_k"] = experiment.top_k
    return snapshot


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "experiment"


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
