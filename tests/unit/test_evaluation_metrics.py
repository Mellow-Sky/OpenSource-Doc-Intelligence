from __future__ import annotations

import pytest

from evaluation.metrics.answer import (
    evaluate_answer,
    exact_match,
    numeric_version_consistency,
    token_f1,
)
from evaluation.metrics.answerability import answerability_metrics
from evaluation.metrics.citations import citation_metrics
from evaluation.metrics.performance import percentile, summarize_performance
from evaluation.metrics.retrieval import evaluate_retrieval, reciprocal_rank
from evaluation.metrics.rewrite import compare_rewrite_retrieval


def test_recall_at_k_set_coverage_and_mrr() -> None:
    metrics = evaluate_retrieval(
        ["noise", "relevant-a", "noise-2", "relevant-b"],
        {"relevant-a", "relevant-b"},
        k_values=(1, 3, 5),
    )

    assert metrics.recall_at_k == {1: 0.0, 3: 1.0, 5: 1.0}
    assert metrics.relevant_set_coverage_at_k == {1: 0.0, 3: 0.5, 5: 1.0}
    assert metrics.reciprocal_rank == 0.5
    assert reciprocal_rank(["x"], ["missing"]) == 0.0


def test_deterministic_answer_metrics_preserve_technical_identifiers() -> None:
    assert (
        exact_match(
            "kubectl rollout undo deployment/nginx", "Kubectl rollout undo Deployment/nginx"
        )
        == 1.0
    )
    assert token_f1(
        "rollout undo deployment/nginx", "kubectl rollout undo deployment/nginx"
    ) == pytest.approx(6 / 7)
    assert numeric_version_consistency("Available since v1.30", "Available since v1.30") == 1.0
    assert numeric_version_consistency("Available since v1.29", "Available since v1.30") == 0.0

    metrics = evaluate_answer("Kubernetes v1.30 supports the field", "The field exists in v1.30")
    assert metrics.keyword_coverage > 0
    assert metrics.numeric_version_consistency == 1.0


def test_citation_metrics_distinguish_validity_and_claim_coverage() -> None:
    metrics = citation_metrics(
        citation_validity=[True, False],
        support_scores=[0.9, 0.2],
        claim_requires_citation=[True, True, False],
        claim_supported=[True, False, False],
    )

    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.correctness == pytest.approx(0.55)
    assert metrics.completeness == 0.5


def test_answerability_confusion_metrics_report_false_answers_and_refusals() -> None:
    metrics = answerability_metrics(
        [True, True, False, False],
        [True, False, True, False],
    )

    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.false_positive_rate == 0.5
    assert metrics.false_negative_rate == 0.5
    assert metrics.false_answers == 1
    assert metrics.false_refusals == 1


def test_performance_summary_uses_interpolated_percentiles() -> None:
    summary = summarize_performance(
        [10.0, 20.0, 30.0, 40.0],
        elapsed_seconds=2.0,
        error_count=1,
    )

    assert percentile([10.0, 20.0], 0.5) == 15.0
    assert summary.mean_ms == 25.0
    assert summary.p50_ms == 25.0
    assert summary.throughput_per_second == 2.5
    assert summary.error_rate == 0.2


def test_rewrite_comparison_flags_topic_switch_and_unnecessary_changes() -> None:
    comparison = compare_rewrite_retrieval(
        original_results=["wrong"],
        rewritten_results=["right"],
        relevant_ids=["right"],
        k=1,
        topic_switched=True,
        query_was_independent=True,
        query_changed=True,
    )

    assert comparison.recall_delta == 1.0
    assert comparison.topic_switch_error is True
    assert comparison.unnecessary_rewrite is True
