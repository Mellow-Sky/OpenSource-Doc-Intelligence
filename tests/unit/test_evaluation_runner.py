from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from app.domain.evaluation import Difficulty, EvaluationCase
from app.providers.base import GenerationResponse, LLMProvider, TokenUsage
from app.services.pricing_service import PricingCatalog
from evaluation.dataset import DatasetError, load_dataset, write_dataset
from evaluation.judges.llm import LLMEvaluationJudge
from evaluation.models import (
    EvaluationCitationSummary,
    EvaluationEvidence,
    EvaluationResponse,
)
from evaluation.reporting import write_comparison, write_report
from evaluation.runner import EvaluationRunner, prompt_versions


class _Executor:
    async def execute(self, case: EvaluationCase) -> EvaluationResponse:
        if case.answerable:
            return EvaluationResponse(
                generated_answer="Use kubectl rollout undo deployment/nginx [1].",
                predicted_answerable=True,
                rewritten_query=case.question,
                evidence=[
                    EvaluationEvidence(
                        chunk_id="chunk-rollout",
                        document_id="document-rollout",
                        title="Deployments",
                        section="Rolling Back",
                        document_type="official_documentation",
                        content="Run kubectl rollout undo deployment/nginx.",
                        rank=1,
                        score=0.9,
                    )
                ],
                citations=EvaluationCitationSummary(
                    citation_ids=["chunk-rollout"],
                    validity=[True],
                    support_scores=[1.0],
                    claim_requires_citation=[True],
                    claim_supported=[True],
                ),
                latency_ms={"total_ms": 50.0},
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "estimated_cost_usd": None,
                },
            )
        return EvaluationResponse(
            generated_answer="The current knowledge base does not contain enough evidence.",
            predicted_answerable=False,
            rewritten_query=case.question,
            latency_ms={"total_ms": 10.0},
            usage={
                "prompt_tokens": 3,
                "completion_tokens": 0,
                "total_tokens": 3,
                "estimated_cost_usd": None,
            },
        )


class _PricedExecutor(_Executor):
    async def execute(self, case: EvaluationCase) -> EvaluationResponse:
        response = await super().execute(case)
        return response.model_copy(
            update={
                "usage": {
                    **response.usage,
                    "estimated_cost_usd": 0.001,
                }
            }
        )


class _JudgeProvider(LLMProvider):
    @property
    def name(self) -> str:
        return "judge-test"

    async def healthcheck(self) -> None:
        return None

    async def generate(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        assert "Retrieved evidence JSON" in messages[0]["content"]
        return GenerationResponse(
            text=json.dumps(
                {
                    "factual_correctness": 5,
                    "completeness": 4,
                    "relevance": 5,
                    "groundedness": 5,
                    "rationale": "The command is directly supported.",
                }
            ),
            model="judge-model",
            usage=TokenUsage(prompt_tokens=20, completion_tokens=8),
        )

    def stream(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        return _empty_stream()


async def _empty_stream() -> AsyncIterator[str]:
    if False:
        yield ""


def _judge_pricing(tmp_path: Path) -> PricingCatalog:
    path = tmp_path / "pricing.yaml"
    path.write_text(
        """pricing:
  judge-test:
    judge-model:
      input_per_million_tokens: 1
      output_per_million_tokens: 2
""",
        encoding="utf-8",
    )
    return PricingCatalog.from_file(path)


def _cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            id="answerable",
            question="How do I roll back nginx?",
            reference_answer="Use kubectl rollout undo deployment/nginx.",
            relevant_chunk_ids=["chunk-rollout"],
            expected_citations=["chunk-rollout"],
            answerable=True,
            category="how_to",
            difficulty=Difficulty.EASY,
            source_type="official_documentation",
            human_reviewed=True,
        ),
        EvaluationCase(
            id="unanswerable",
            question="What is the weather?",
            reference_answer="Outside the Kubernetes knowledge base.",
            answerable=False,
            category="out_of_scope",
            difficulty=Difficulty.EASY,
            human_reviewed=True,
        ),
    ]


@pytest.mark.asyncio
async def test_runner_executes_real_metric_and_judge_paths(tmp_path: Path) -> None:
    judge = LLMEvaluationJudge(
        _JudgeProvider(),
        prompt_path=Path("prompts/eval_judge.md"),
        pricing_catalog=_judge_pricing(tmp_path),
    )
    report = await EvaluationRunner(_Executor(), judge=judge, concurrency=2).run(
        _cases(),
        experiment_name="hybrid-reranked",
        dataset_name="fixture",
        dataset_path=tmp_path / "fixture.jsonl",
        config_snapshot={"retrieval_mode": "hybrid"},
        prompt_versions=prompt_versions(Path("prompts")),
    )

    assert report.dataset_size == 2
    assert report.summary["recall_at_1"] == 0.5
    assert report.summary["no_answer_accuracy"] == 1.0
    assert report.summary["answer_correct"] == 1.0
    assert report.summary["total_tokens"] == 46
    assert report.summary["prompt_tokens"] == 33
    assert report.summary["completion_tokens"] == 13
    assert report.summary["judge_total_cost_usd"] == pytest.approx(0.000036)
    assert report.summary["total_cost_usd"] is None
    assert report.summary["answer_cost_complete"] is False
    assert report.summary["judge_cost_complete"] is True
    assert report.summary["cost_complete"] is False
    assert report.results[0].judge is not None
    assert report.results[0].judge.groundedness == 5
    assert report.results[0].judge_provider == "judge-test"
    assert report.results[0].judge_model == "judge-model"
    assert report.results[0].judge_usage == {
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "total_tokens": 28,
    }
    assert report.results[0].judge_estimated_cost_usd == pytest.approx(0.000036)
    assert report.results[1].judge_estimated_cost_usd is None

    paths = write_report(report, tmp_path / "report")
    report_json = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert report_json["dataset_size"] == 2
    assert report_json["results"][0]["judge_usage"]["total_tokens"] == 28
    assert report_json["results"][0]["judge_estimated_cost_usd"] == pytest.approx(0.000036)
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "recall_at_20" in markdown
    assert "judge_total_cost_usd" in markdown
    assert "case_id" in paths["csv"].read_text(encoding="utf-8")

    raw = report.model_copy(update={"experiment_name": "raw-query"}, deep=True)
    rewritten = report.model_copy(update={"experiment_name": "rewritten-query"}, deep=True)
    comparison_paths = write_comparison([raw, rewritten], tmp_path / "comparison")
    comparison = json.loads(comparison_paths["json"].read_text(encoding="utf-8"))
    assert comparison["rewrite_analysis"]["recall_delta_at_5"] == 0.0
    assert "Query rewrite analysis" in comparison_paths["markdown"].read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_runner_combines_answer_and_judge_costs_only_when_all_prices_known(
    tmp_path: Path,
) -> None:
    known_judge = LLMEvaluationJudge(
        _JudgeProvider(),
        prompt_path=Path("prompts/eval_judge.md"),
        pricing_catalog=_judge_pricing(tmp_path),
    )
    complete = await EvaluationRunner(_PricedExecutor(), judge=known_judge).run(
        [_cases()[0]],
        experiment_name="priced",
        dataset_name="fixture",
        dataset_path=tmp_path / "fixture.jsonl",
        config_snapshot={},
    )

    assert complete.summary["answer_total_cost_usd"] == pytest.approx(0.001)
    assert complete.summary["judge_total_cost_usd"] == pytest.approx(0.000036)
    assert complete.summary["total_cost_usd"] == pytest.approx(0.001036)
    assert complete.summary["average_cost_usd"] == pytest.approx(0.001036)
    assert complete.summary["cost_complete"] is True

    unknown_judge = LLMEvaluationJudge(
        _JudgeProvider(),
        prompt_path=Path("prompts/eval_judge.md"),
        pricing_catalog=PricingCatalog(),
    )
    incomplete = await EvaluationRunner(_PricedExecutor(), judge=unknown_judge).run(
        [_cases()[0]],
        experiment_name="unknown-judge-price",
        dataset_name="fixture",
        dataset_path=tmp_path / "fixture.jsonl",
        config_snapshot={},
    )

    assert incomplete.results[0].judge_estimated_cost_usd is None
    assert incomplete.summary["answer_cost_complete"] is True
    assert incomplete.summary["judge_cost_complete"] is False
    assert incomplete.summary["cost_complete"] is False
    assert incomplete.summary["total_cost_usd"] is None
    assert incomplete.summary["average_cost_usd"] is None


@pytest.mark.asyncio
async def test_failed_case_keeps_run_cost_unknown_even_when_successful_cases_are_priced() -> None:
    class PartiallyFailingExecutor(_PricedExecutor):
        async def execute(self, case: EvaluationCase) -> EvaluationResponse:
            if not case.answerable:
                raise RuntimeError("provider response was lost after billing")
            return await super().execute(case)

    report = await EvaluationRunner(PartiallyFailingExecutor()).run(
        _cases(),
        experiment_name="partial-failure",
        dataset_name="fixture",
        dataset_path=Path("fixture.jsonl"),
        config_snapshot={},
    )

    assert report.summary["successful_case_count"] == 1
    assert report.summary["error_count"] == 1
    assert report.summary["answer_cost_complete"] is False
    assert report.summary["judge_cost_complete"] is False
    assert report.summary["cost_complete"] is False
    assert report.summary["answer_total_cost_usd"] is None
    assert report.summary["total_cost_usd"] is None
    assert report.summary["average_cost_usd"] is None


def test_dataset_jsonl_round_trip_and_duplicate_rejection(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    assert write_dataset(path, _cases()) == 2
    assert [case.id for case in load_dataset(path)] == ["answerable", "unanswerable"]

    duplicate = _cases()[0].model_dump_json() + "\n" + _cases()[0].model_dump_json()
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(DatasetError, match="Duplicate"):
        load_dataset(path)


@pytest.mark.asyncio
async def test_portable_source_hash_resolves_runtime_chunk_ids() -> None:
    content = "Portable source content about a Deployment rollback."
    case = EvaluationCase(
        id="portable",
        question="How is rollback described?",
        reference_answer=content,
        relevant_chunk_ids=["portable-chunk"],
        expected_citations=["portable-chunk"],
        answerable=True,
        category="factual",
        difficulty=Difficulty.EASY,
        metadata={
            "source_chunks": [
                {
                    "chunk_id": "portable-chunk",
                    "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                }
            ]
        },
    )

    class PortableExecutor:
        async def execute(self, _case: EvaluationCase) -> EvaluationResponse:
            return EvaluationResponse(
                generated_answer=content,
                predicted_answerable=True,
                rewritten_query=_case.question,
                evidence=[
                    EvaluationEvidence(
                        chunk_id="runtime-uuid",
                        title="Deployment",
                        content=content,
                        rank=1,
                    )
                ],
                citations=EvaluationCitationSummary(
                    citation_ids=["runtime-uuid"],
                    validity=[True],
                    support_scores=[1.0],
                    claim_requires_citation=[True],
                    claim_supported=[True],
                ),
            )

    report = await EvaluationRunner(PortableExecutor()).run(
        [case],
        experiment_name="portable",
        dataset_name="portable",
        dataset_path=Path("portable.jsonl"),
        config_snapshot={},
    )

    assert report.results[0].metrics["recall_at_1"] == 1.0
    assert report.results[0].metrics["expected_citation_recall"] == 1.0
