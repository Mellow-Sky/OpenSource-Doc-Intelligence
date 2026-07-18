"""Write auditable JSON, JSONL, CSV, and Markdown evaluation reports."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation.models import EvaluationResultRecord, EvaluationRunReport


def write_report(report: EvaluationRunReport, output_directory: Path) -> dict[str, Path]:
    """Persist one report in four inspectable formats using atomic replacements."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_directory / "report.json",
        "jsonl": output_directory / "results.jsonl",
        "csv": output_directory / "results.csv",
        "markdown": output_directory / "report.md",
    }
    _atomic_write(paths["json"], report.model_dump_json(indent=2, exclude_none=False) + "\n")
    _atomic_write(
        paths["jsonl"],
        "".join(result.model_dump_json(exclude_none=False) + "\n" for result in report.results),
    )
    _atomic_write(paths["csv"], _results_csv(report.results))
    _atomic_write(paths["markdown"], render_markdown(report))
    return paths


def write_comparison(
    reports: Sequence[EvaluationRunReport],
    output_directory: Path,
) -> dict[str, Path]:
    """Persist a compact experiment comparison in JSON, CSV, and Markdown."""
    if not reports:
        msg = "comparison requires at least one report"
        raise ValueError(msg)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = [_comparison_row(report) for report in reports]
    rewrite_analysis = _rewrite_analysis(reports)
    json_path = output_directory / "comparison.json"
    csv_path = output_directory / "comparison.csv"
    markdown_path = output_directory / "comparison.md"
    _atomic_write(
        json_path,
        json.dumps(
            {"experiments": rows, "rewrite_analysis": rewrite_analysis},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    _atomic_write(csv_path, _mapping_csv(rows))
    _atomic_write(markdown_path, _comparison_markdown(rows, rewrite_analysis))
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def render_markdown(report: EvaluationRunReport) -> str:
    """Render aggregate, grouped, cost, performance, and failure analysis."""
    summary = report.summary
    lines = [
        f"# Evaluation Report: {_escape(report.experiment_name)}",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Dataset: `{_escape(report.dataset_name)}` ({report.dataset_size} cases)",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Started (UTC): `{report.started_at.isoformat()}`",
        f"- Finished (UTC): `{report.finished_at.isoformat()}`",
        f"- Git commit: `{report.git_commit or 'unknown'}`",
        "",
        "## Overall metrics",
        "",
        _metric_table(
            summary,
            (
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "recall_at_10",
                "recall_at_20",
                "mrr",
                "exact_match",
                "token_f1",
                "answer_correct",
                "citation_precision",
                "citation_recall",
                "citation_correctness",
                "citation_completeness",
                "no_answer_precision",
                "no_answer_recall",
                "no_answer_f1",
            ),
        ),
        "",
        "## Performance and cost",
        "",
        _metric_table(
            summary,
            (
                "latency_mean_ms",
                "latency_p50_ms",
                "latency_p90_ms",
                "latency_p95_ms",
                "latency_p99_ms",
                "latency_min_ms",
                "latency_max_ms",
                "latency_throughput_per_second",
                "latency_error_rate",
                "answer_prompt_tokens",
                "answer_completion_tokens",
                "answer_total_tokens",
                "judge_prompt_tokens",
                "judge_completion_tokens",
                "judge_tokens",
                "average_tokens",
                "total_tokens",
                "answer_total_cost_usd",
                "judge_total_cost_usd",
                "average_cost_usd",
                "total_cost_usd",
                "answer_cost_complete",
                "judge_cost_complete",
                "cost_complete",
            ),
        ),
        "",
        "Unknown model prices remain `null`; they are never replaced with an invented cost.",
        "",
        "## Metrics by category",
        "",
        _group_table(report.category_metrics),
        "",
        "## Metrics by difficulty",
        "",
        _group_table(report.difficulty_metrics),
        "",
        "## Answerable vs unanswerable",
        "",
        _group_table(report.answerability_groups),
        "",
        "## Failure analysis",
        "",
        *_failure_analysis(report.results),
        "",
        "## Configuration snapshot",
        "",
        "```json",
        json.dumps(report.config_snapshot, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Prompt versions",
        "",
        _mapping_table(report.prompt_versions),
        "",
    ]
    return "\n".join(lines)


def _results_csv(results: Sequence[EvaluationResultRecord]) -> str:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "case_id": result.case.id,
            "question": result.case.question,
            "category": result.case.category,
            "difficulty": result.case.difficulty.value,
            "expected_answerable": result.case.answerable,
            "predicted_answerable": result.predicted_answerable,
            "rewritten_query": result.rewritten_query,
            "generated_answer": result.generated_answer,
            "retrieved_chunk_ids": json.dumps(
                [item.chunk_id for item in result.retrieved_evidence], ensure_ascii=False
            ),
            "citation_chunk_ids": json.dumps(result.citations.citation_ids, ensure_ascii=False),
            "judge_provider": result.judge_provider,
            "judge_model": result.judge_model,
            "judge_prompt_tokens": result.judge_usage.get("prompt_tokens"),
            "judge_completion_tokens": result.judge_usage.get("completion_tokens"),
            "judge_total_tokens": result.judge_usage.get("total_tokens"),
            "judge_estimated_cost_usd": result.judge_estimated_cost_usd,
            "error": result.error,
        }
        row.update(result.metrics)
        row.update({f"latency_{key}": value for key, value in result.latency_ms.items()})
        row.update({f"usage_{key}": value for key, value in result.usage.items()})
        rows.append(row)
    return _mapping_csv(rows)


def _comparison_row(report: EvaluationRunReport) -> dict[str, Any]:
    keys = (
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "token_f1",
        "answer_correct",
        "citation_precision",
        "citation_recall",
        "no_answer_f1",
        "latency_p95_ms",
        "average_tokens",
        "average_cost_usd",
    )
    return {
        "experiment": report.experiment_name,
        "run_id": report.run_id,
        "dataset_fingerprint": report.dataset_fingerprint,
        **{key: report.summary.get(key) for key in keys},
    }


def _comparison_markdown(
    rows: Sequence[Mapping[str, Any]],
    rewrite_analysis: Mapping[str, Any] | None,
) -> str:
    headers = list(rows[0])
    lines = ["# Evaluation Experiment Comparison", "", _markdown_rows(headers, rows), ""]
    if rewrite_analysis is not None:
        lines.extend(
            [
                "## Query rewrite analysis",
                "",
                _mapping_table(rewrite_analysis),
                "",
            ]
        )
    return "\n".join(lines)


def _rewrite_analysis(
    reports: Sequence[EvaluationRunReport],
) -> dict[str, float | int] | None:
    raw = next((report for report in reports if report.experiment_name == "raw-query"), None)
    rewritten = next(
        (report for report in reports if report.experiment_name == "rewritten-query"),
        None,
    )
    if raw is None or rewritten is None:
        return None
    raw_cases = {result.case.id: result for result in raw.results if result.error is None}
    rewritten_cases = {
        result.case.id: result for result in rewritten.results if result.error is None
    }
    common = sorted(set(raw_cases).intersection(rewritten_cases))
    if not common:
        return None
    output: dict[str, float | int] = {"case_count": len(common)}
    for k in (1, 3, 5, 10, 20):
        raw_recall = sum(
            float(raw_cases[item].metrics[f"recall_at_{k}"] or 0) for item in common
        ) / len(common)
        rewritten_recall = sum(
            float(rewritten_cases[item].metrics[f"recall_at_{k}"] or 0) for item in common
        ) / len(common)
        output[f"raw_recall_at_{k}"] = raw_recall
        output[f"rewritten_recall_at_{k}"] = rewritten_recall
        output[f"recall_delta_at_{k}"] = rewritten_recall - raw_recall
    topic_switch = [
        rewritten_cases[item]
        for item in common
        if bool(rewritten_cases[item].case.metadata.get("topic_switch"))
    ]
    independent = [
        rewritten_cases[item]
        for item in common
        if bool(rewritten_cases[item].case.metadata.get("query_independent"))
    ]
    output["topic_switch_error_rate"] = (
        sum(float(item.metrics["topic_switch_error"] or 0) for item in topic_switch)
        / len(topic_switch)
        if topic_switch
        else 0.0
    )
    output["unnecessary_rewrite_rate"] = (
        sum(float(item.metrics["unnecessary_rewrite"] or 0) for item in independent)
        / len(independent)
        if independent
        else 0.0
    )
    return output


def _metric_table(summary: Mapping[str, Any], keys: Sequence[str]) -> str:
    return _markdown_rows(
        ["metric", "value"],
        [{"metric": key, "value": _format(summary.get(key))} for key in keys],
    )


def _group_table(groups: Mapping[str, Mapping[str, Any]]) -> str:
    if not groups:
        return "No successful samples."
    metric_names = ("count", "recall_at_5", "mrr", "token_f1", "answer_correct")
    rows = [
        {"group": name, **{metric: _format(values.get(metric)) for metric in metric_names}}
        for name, values in groups.items()
    ]
    return _markdown_rows(["group", *metric_names], rows)


def _mapping_table(values: Mapping[str, Any]) -> str:
    if not values:
        return "No values recorded."
    return _markdown_rows(
        ["name", "value"],
        [{"name": key, "value": value} for key, value in sorted(values.items())],
    )


def _failure_analysis(results: Sequence[EvaluationResultRecord]) -> list[str]:
    errors = [item for item in results if item.error]
    missed = [
        item for item in results if item.case.answerable and item.metrics.get("recall_at_20") == 0
    ]
    citation_errors = [
        item
        for item in results
        if item.predicted_answerable and (item.metrics.get("citation_precision") or 0) < 1
    ]
    worst = sorted(
        (item for item in results if item.error is None),
        key=lambda item: float(item.metrics.get("token_f1") or 0),
    )[:10]
    lines = [
        f"- Execution errors: {len(errors)}",
        f"- Answerable cases with no relevant chunk in Top 20: {len(missed)}",
        f"- Answers with citation precision below 1: {len(citation_errors)}",
        "",
        "### Worst queries by Token F1",
        "",
    ]
    if not worst:
        lines.append("No successful samples.")
    else:
        lines.extend(
            f"- `{_escape(item.case.id)}` ({_format(item.metrics.get('token_f1'))}): "
            f"{_escape(item.case.question)}"
            for item in worst
        )
    lines.extend(["", "### Unretrieved cases", ""])
    lines.extend(
        [f"- `{_escape(item.case.id)}`: {_escape(item.case.question)}" for item in missed]
        or ["None."]
    )
    lines.extend(["", "### Citation error cases", ""])
    lines.extend(
        [
            f"- `{_escape(item.case.id)}`: precision "
            f"{_format(item.metrics.get('citation_precision'))}"
            for item in citation_errors
        ]
        or ["None."]
    )
    return lines


def _mapping_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    headers = sorted({key for row in rows for key in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _markdown_rows(headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    line_one = "| " + " | ".join(_escape(header) for header in headers) + " |"
    line_two = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_escape(str(row.get(header, ""))) for header in headers) + " |"
        for row in rows
    ]
    return "\n".join([line_one, line_two, *body])


def _format(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).lower() if isinstance(value, bool) else str(value)


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _atomic_write(path: Path, content: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
