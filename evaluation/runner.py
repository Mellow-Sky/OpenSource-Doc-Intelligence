"""Asynchronous offline evaluation engine over a replaceable RAG case executor."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol
from uuid import uuid4

from app.domain.evaluation import EvaluationCase
from evaluation.dataset import dataset_fingerprint
from evaluation.judges.llm import LLMEvaluationJudge
from evaluation.metrics.answer import evaluate_answer
from evaluation.metrics.answerability import answerability_metrics
from evaluation.metrics.citations import citation_metrics
from evaluation.metrics.performance import percentile, summarize_performance
from evaluation.metrics.retrieval import DEFAULT_K_VALUES, evaluate_retrieval
from evaluation.models import (
    EvaluationResponse,
    EvaluationResultRecord,
    EvaluationRunReport,
)


class EvaluationExecutor(Protocol):
    """Execute one case using a real system or an explicit test double."""

    async def execute(self, case: EvaluationCase) -> EvaluationResponse:
        """Return the complete answer/retrieval trace for one case."""


class EvaluationRunner:
    """Run bounded concurrent cases and aggregate deterministic plus judge metrics."""

    def __init__(
        self,
        executor: EvaluationExecutor,
        *,
        judge: LLMEvaluationJudge | None = None,
        concurrency: int = 1,
        fail_fast: bool = False,
    ) -> None:
        if concurrency < 1:
            msg = "evaluation concurrency must be positive"
            raise ValueError(msg)
        self._executor = executor
        self._judge = judge
        self._semaphore = asyncio.Semaphore(concurrency)
        self._fail_fast = fail_fast

    async def run(
        self,
        cases: Sequence[EvaluationCase],
        *,
        experiment_name: str,
        dataset_name: str,
        dataset_path: Path,
        config_snapshot: Mapping[str, Any],
        git_commit: str | None = None,
        prompt_versions: Mapping[str, str] | None = None,
        run_id: str | None = None,
    ) -> EvaluationRunReport:
        """Evaluate all supplied cases and return a fully serializable report."""
        if not cases:
            msg = "evaluation requires at least one case"
            raise ValueError(msg)
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        tasks = [asyncio.create_task(self._run_case(case)) for case in cases]
        if self._fail_fast:
            results = await asyncio.gather(*tasks)
        else:
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            results = [
                _unexpected_failure(case, item) if isinstance(item, BaseException) else item
                for case, item in zip(cases, gathered, strict=True)
            ]
        elapsed = time.perf_counter() - started
        return EvaluationRunReport(
            run_id=run_id or str(uuid4()),
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            dataset_path=str(dataset_path),
            dataset_fingerprint=dataset_fingerprint(cases),
            dataset_size=len(cases),
            started_at=started_at,
            finished_at=datetime.now(UTC),
            elapsed_seconds=elapsed,
            config_snapshot=dict(config_snapshot),
            git_commit=git_commit,
            prompt_versions=dict(prompt_versions or {}),
            summary=_summarize(results, elapsed),
            category_metrics=_group_metrics(results, lambda item: item.case.category),
            difficulty_metrics=_group_metrics(results, lambda item: item.case.difficulty.value),
            answerability_groups=_group_metrics(
                results,
                lambda item: "answerable" if item.case.answerable else "unanswerable",
            ),
            results=results,
        )

    async def _run_case(self, case: EvaluationCase) -> EvaluationResultRecord:
        async with self._semaphore:
            response = await self._executor.execute(case)
            return await self._score(case, response)

    async def _score(
        self,
        case: EvaluationCase,
        response: EvaluationResponse,
    ) -> EvaluationResultRecord:
        resolved_ids = _resolved_evidence_ids(case, response)
        retrieved_ids = [resolved_ids[item.chunk_id] for item in response.evidence]
        retrieval = evaluate_retrieval(retrieved_ids, case.relevant_chunk_ids)
        answer = evaluate_answer(
            response.generated_answer,
            case.reference_answer,
            keywords=_metadata_keywords(case),
        )
        citations = citation_metrics(
            citation_validity=response.citations.validity,
            support_scores=response.citations.support_scores,
            claim_requires_citation=response.citations.claim_requires_citation,
            claim_supported=response.citations.claim_supported,
        )
        metrics: dict[str, float | int | None] = {
            **{f"recall_at_{k}": retrieval.recall_at_k[k] for k in DEFAULT_K_VALUES},
            **{
                f"relevant_set_recall_at_{k}": retrieval.relevant_set_coverage_at_k[k]
                for k in DEFAULT_K_VALUES
            },
            "mrr": retrieval.reciprocal_rank,
            "exact_match": answer.exact_match,
            "token_f1": answer.token_f1,
            "keyword_coverage": answer.keyword_coverage,
            "numeric_version_consistency": answer.numeric_version_consistency,
            "citation_precision": citations.precision,
            "citation_recall": citations.recall,
            "citation_correctness": citations.correctness,
            "citation_completeness": citations.completeness,
            "expected_citation_recall": _expected_citation_recall(
                case,
                response,
                resolved_ids,
            ),
            "answerability_correct": float(response.predicted_answerable == case.answerable),
            "query_changed": float(response.rewritten_query.strip() != case.question.strip()),
            "topic_switch_error": float(
                bool(case.metadata.get("topic_switch"))
                and response.rewritten_query.strip() != case.question.strip()
            ),
            "unnecessary_rewrite": float(
                bool(case.metadata.get("query_independent"))
                and response.rewritten_query.strip() != case.question.strip()
            ),
        }
        judge_result = None
        if self._judge is not None and response.predicted_answerable:
            judge_result = await self._judge.evaluate(
                case=case,
                generated_answer=response.generated_answer,
                evidence=[
                    {
                        "chunk_id": item.chunk_id,
                        "title": item.title,
                        "section": item.section,
                        "content": item.content,
                    }
                    for item in response.evidence
                ],
            )
            metrics["answer_correct"] = float(
                judge_result.scores.factual_correctness >= 4
                and judge_result.scores.groundedness >= 4
            )
            for name in ("factual_correctness", "completeness", "relevance", "groundedness"):
                metrics[f"judge_{name}"] = getattr(judge_result.scores, name)
        else:
            metrics["answer_correct"] = None
        return EvaluationResultRecord(
            case=case,
            generated_answer=response.generated_answer,
            rewritten_query=response.rewritten_query,
            predicted_answerable=response.predicted_answerable,
            retrieved_evidence=response.evidence,
            metrics=metrics,
            citations=response.citations,
            judge=judge_result.scores if judge_result is not None else None,
            judge_provider=judge_result.provider if judge_result is not None else None,
            judge_model=judge_result.model if judge_result is not None else None,
            judge_usage=(
                {
                    "prompt_tokens": judge_result.usage.prompt_tokens,
                    "completion_tokens": judge_result.usage.completion_tokens,
                    "total_tokens": judge_result.usage.total_tokens,
                }
                if judge_result is not None
                else {}
            ),
            judge_estimated_cost_usd=(
                judge_result.estimated_cost_usd if judge_result is not None else None
            ),
            latency_ms={
                **response.latency_ms,
                **({"judge_ms": judge_result.latency_ms} if judge_result is not None else {}),
            },
            usage=response.usage,
        )


def prompt_versions(directory: Path) -> dict[str, str]:
    """Hash every prompt file so reports are tied to exact prompt content."""
    if not directory.exists():
        return {}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.md"))
    }


def _metadata_keywords(case: EvaluationCase) -> list[str] | None:
    value = case.metadata.get("keywords")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return value


def _resolved_evidence_ids(
    case: EvaluationCase,
    response: EvaluationResponse,
) -> dict[str, str]:
    """Map retrieved UUIDs to portable dataset IDs when exact source hashes match."""
    source_chunks = case.metadata.get("source_chunks", [])
    hash_to_id = {
        str(item["content_sha256"]): str(item["chunk_id"])
        for item in source_chunks
        if isinstance(item, dict)
        and isinstance(item.get("content_sha256"), str)
        and isinstance(item.get("chunk_id"), str)
    }
    resolved: dict[str, str] = {}
    for evidence in response.evidence:
        content_hash = hashlib.sha256(evidence.content.encode("utf-8")).hexdigest()
        resolved[evidence.chunk_id] = hash_to_id.get(content_hash, evidence.chunk_id)
    return resolved


def _expected_citation_recall(
    case: EvaluationCase,
    response: EvaluationResponse,
    resolved_ids: Mapping[str, str],
) -> float:
    expected = set(case.expected_citations)
    if not expected:
        return 1.0 if not response.citations.citation_ids else 0.0
    cited = {resolved_ids.get(item, item) for item in response.citations.citation_ids}
    return len(expected.intersection(cited)) / len(expected)


def _summarize(
    results: Sequence[EvaluationResultRecord],
    elapsed_seconds: float,
) -> dict[str, Any]:
    valid = [result for result in results if result.error is None]
    summary: dict[str, Any] = _mean_metrics(valid)
    labels = [result.case.answerable for result in valid]
    predictions = [result.predicted_answerable for result in valid]
    classification = answerability_metrics(labels, predictions)
    summary.update({f"no_answer_{key}": value for key, value in asdict(classification).items()})

    latencies = [result.latency_ms.get("total_ms", 0.0) for result in valid]
    performance = summarize_performance(
        latencies,
        elapsed_seconds=elapsed_seconds,
        error_count=len(results) - len(valid),
    )
    summary.update({f"latency_{key}": value for key, value in asdict(performance).items()})
    summary.update(_stage_latency_metrics(valid))
    answer_total_tokens = sum(_integer_usage(result, "total_tokens") for result in valid)
    answer_prompt_tokens = sum(_integer_usage(result, "prompt_tokens") for result in valid)
    answer_completion_tokens = sum(_integer_usage(result, "completion_tokens") for result in valid)
    judge_prompt_tokens = sum(int(result.judge_usage.get("prompt_tokens", 0)) for result in valid)
    judge_completion_tokens = sum(
        int(result.judge_usage.get("completion_tokens", 0)) for result in valid
    )
    judge_tokens = sum(int(result.judge_usage.get("total_tokens", 0)) for result in valid)

    answer_cost_values = [result.usage.get("estimated_cost_usd") for result in valid]
    answer_costs = [float(value) for value in answer_cost_values if isinstance(value, (int, float))]
    judged = [result for result in valid if result.judge_model is not None]
    judge_costs = [
        float(result.judge_estimated_cost_usd)
        for result in judged
        if isinstance(result.judge_estimated_cost_usd, (int, float))
    ]
    # A failed case can have consumed provider tokens before raising. Because that
    # usage is not available in the result record, a run containing failures must
    # never claim that its cost total is complete.
    all_cases_succeeded = len(valid) == len(results)
    answer_cost_complete = all_cases_succeeded and len(answer_costs) == len(valid)
    judge_cost_complete = all_cases_succeeded and len(judge_costs) == len(judged)
    cost_complete = answer_cost_complete and judge_cost_complete
    complete_cost = sum(answer_costs) + sum(judge_costs)
    combined_tokens = answer_total_tokens + judge_tokens
    summary.update(
        {
            "case_count": len(results),
            "successful_case_count": len(valid),
            "error_count": len(results) - len(valid),
            "answer_prompt_tokens": answer_prompt_tokens,
            "answer_completion_tokens": answer_completion_tokens,
            "answer_total_tokens": answer_total_tokens,
            "judge_prompt_tokens": judge_prompt_tokens,
            "judge_completion_tokens": judge_completion_tokens,
            "judge_tokens": judge_tokens,
            "prompt_tokens": answer_prompt_tokens + judge_prompt_tokens,
            "completion_tokens": answer_completion_tokens + judge_completion_tokens,
            "total_tokens": combined_tokens,
            "average_tokens": combined_tokens / len(valid) if valid else 0.0,
            "answer_total_cost_usd": sum(answer_costs) if answer_cost_complete else None,
            "judge_total_cost_usd": sum(judge_costs) if judge_cost_complete else None,
            "total_cost_usd": complete_cost if cost_complete else None,
            "average_cost_usd": (complete_cost / len(valid) if cost_complete and valid else None),
            "answer_cost_complete": answer_cost_complete,
            "judge_cost_complete": judge_cost_complete,
            "cost_complete": cost_complete,
        }
    )
    return summary


def _group_metrics(
    results: Sequence[EvaluationResultRecord],
    key: Callable[[EvaluationResultRecord], str],
) -> dict[str, dict[str, float | int | None]]:
    groups: defaultdict[str, list[EvaluationResultRecord]] = defaultdict(list)
    for result in results:
        if result.error is None:
            groups[str(key(result))].append(result)
    return {
        name: {"count": len(items), **_mean_metrics(items)}
        for name, items in sorted(groups.items())
    }


def _mean_metrics(results: Sequence[EvaluationResultRecord]) -> dict[str, float | None]:
    keys = sorted({key for result in results for key in result.metrics})
    output: dict[str, float | None] = {}
    for key in keys:
        values = [
            float(value)
            for result in results
            if isinstance((value := result.metrics.get(key)), (int, float))
        ]
        output[key] = fmean(values) if values else None
    return output


def _integer_usage(result: EvaluationResultRecord, key: str) -> int:
    value = result.usage.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _stage_latency_metrics(
    results: Sequence[EvaluationResultRecord],
) -> dict[str, float]:
    output: dict[str, float] = {}
    stages = sorted({stage for result in results for stage in result.latency_ms})
    for stage in stages:
        values = [result.latency_ms[stage] for result in results if stage in result.latency_ms]
        if not values:
            continue
        prefix = stage.removesuffix("_ms")
        output.update(
            {
                f"stage_{prefix}_mean_ms": fmean(values),
                f"stage_{prefix}_p50_ms": percentile(values, 0.50),
                f"stage_{prefix}_p90_ms": percentile(values, 0.90),
                f"stage_{prefix}_p95_ms": percentile(values, 0.95),
                f"stage_{prefix}_p99_ms": percentile(values, 0.99),
                f"stage_{prefix}_min_ms": min(values),
                f"stage_{prefix}_max_ms": max(values),
            }
        )
    return output


def _unexpected_failure(case: EvaluationCase, error: BaseException) -> EvaluationResultRecord:
    return EvaluationResultRecord(
        case=case,
        generated_answer="",
        rewritten_query=case.question,
        predicted_answerable=False,
        retrieved_evidence=[],
        metrics={},
        citations={"citation_ids": []},
        error=f"{type(error).__name__}: {error}",
    )
