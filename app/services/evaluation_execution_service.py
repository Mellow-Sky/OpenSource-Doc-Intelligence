"""Execute one durable evaluation run through the production ChatService pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy.engine import make_url

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.db.models.evaluation import EvaluationRun
from app.db.session import Database
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
from evaluation.reporting import write_report
from evaluation.runner import EvaluationRunner, prompt_versions

_ALLOWED_OVERRIDES = {
    "citation_coverage_threshold",
    "enable_citation_validation",
    "enable_query_rewrite",
    "enable_reranker",
    "evidence_sufficiency_threshold",
    "fusion_mode",
    "keyword_fusion_weight",
    "keyword_top_k",
    "max_context_tokens",
    "no_answer_avg_threshold",
    "no_answer_gray_zone_lower",
    "no_answer_gray_zone_upper",
    "no_answer_margin_threshold",
    "no_answer_top1_threshold",
    "no_answer_top_k",
    "no_answer_topic_overlap_threshold",
    "rerank_top_k",
    "reranker_model",
    "reranker_score_threshold",
    "retrieval_mode",
    "rrf_k",
    "vector_top_k",
}


class EvaluationExecutionService:
    """Create isolated providers, run cases, and atomically write report artifacts."""

    def __init__(
        self,
        settings: Settings,
        *,
        dataset_root: Path = Path("evaluation/datasets"),
        report_root: Path = Path("evaluation/reports"),
    ) -> None:
        self._base_settings = settings
        self._dataset_root = dataset_root
        self._report_root = report_root

    async def execute(
        self,
        run: EvaluationRun,
        *,
        output_directory: Path | None = None,
    ) -> EvaluationRunReport:
        """Execute a claimed run and return its on-disk report model."""
        request, overrides = _snapshot_parts(run.config_snapshot)
        settings, top_k = _experiment_settings(self._base_settings, overrides)
        dataset_path = _safe_dataset_path(str(request["dataset_path"]), self._dataset_root)
        cases = await asyncio.to_thread(load_dataset, dataset_path)
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
            judge_provider = create_judge_provider(settings)
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
            report = await EvaluationRunner(
                ChatEvaluationExecutor(service, top_k=top_k),
                judge=judge,
                concurrency=settings.evaluation_concurrency,
            ).run(
                cases,
                experiment_name=str(request["experiment_name"]),
                dataset_name=run.dataset_name,
                dataset_path=dataset_path,
                config_snapshot=_safe_snapshot(settings, top_k),
                git_commit=settings.git_commit or await _git_commit(),
                prompt_versions=prompt_versions(settings.prompt_directory),
                run_id=str(run.id),
            )
            output = output_directory or self._report_root / str(run.id)
            await asyncio.to_thread(write_report, report, output)
            return report
        finally:
            closed: set[int] = set()
            for provider in providers:
                if id(provider) not in closed:
                    await provider.close()
                    closed.add(id(provider))
            await database.close()


def _snapshot_parts(
    snapshot: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = snapshot.get("request")
    overrides = snapshot.get("overrides", {})
    if not isinstance(request, dict) or not isinstance(overrides, dict):
        raise ConfigurationError("Evaluation run snapshot is invalid")
    if not isinstance(request.get("dataset_path"), str) or not isinstance(
        request.get("experiment_name"), str
    ):
        raise ConfigurationError("Evaluation run request snapshot is invalid")
    return request, overrides


def _experiment_settings(
    base: Settings,
    overrides: Mapping[str, Any],
) -> tuple[Settings, int | None]:
    unknown = sorted(set(overrides).difference(_ALLOWED_OVERRIDES | {"experiment_top_k"}))
    if unknown:
        raise ConfigurationError(f"Evaluation overrides are not allowed: {', '.join(unknown)}")
    top_k_raw = overrides.get("experiment_top_k")
    if top_k_raw is not None and (
        isinstance(top_k_raw, bool) or not isinstance(top_k_raw, int) or not 1 <= top_k_raw <= 100
    ):
        raise ConfigurationError("experiment_top_k must be an integer from 1 through 100")
    settings_overrides = {
        key: value for key, value in overrides.items() if key in _ALLOWED_OVERRIDES
    }
    try:
        settings = Settings.model_validate({**base.model_dump(), **settings_overrides})
    except ValidationError as exc:
        raise ConfigurationError("Evaluation setting overrides are invalid") from exc
    return settings, top_k_raw


def _safe_dataset_path(value: str, root: Path) -> Path:
    resolved_root = root.resolve(strict=False)
    requested = Path(value)
    candidates = (
        [requested]
        if requested.is_absolute()
        else [Path.cwd() / requested, resolved_root / requested]
    )
    resolved = next(
        (
            candidate.resolve(strict=False)
            for candidate in candidates
            if candidate.resolve(strict=False).is_relative_to(resolved_root)
        ),
        None,
    )
    if resolved is None or resolved.suffix.casefold() != ".jsonl" or not resolved.is_file():
        raise ConfigurationError("Evaluation dataset path is invalid or unavailable")
    return resolved


def _safe_snapshot(settings: Settings, top_k: int | None) -> dict[str, Any]:
    snapshot = settings.model_dump(mode="json")
    for key in (
        "api_key",
        "admin_api_key",
        "github_token",
        "llm_api_key",
        "embedding_api_key",
        "reranker_api_key",
        "judge_api_key",
    ):
        snapshot[key] = None if snapshot.get(key) is None else "[REDACTED]"
    snapshot["database_url"] = make_url(settings.database_url).render_as_string(hide_password=True)
    snapshot["experiment_top_k"] = top_k
    return snapshot


async def _git_commit() -> str | None:
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "rev-parse",
            "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5)
    except (OSError, TimeoutError):
        return None
    if process.returncode != 0:
        return None
    value = stdout.decode(errors="replace").strip()
    return value or None
