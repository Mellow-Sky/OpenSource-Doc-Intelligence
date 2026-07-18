"""Recommend no-answer score thresholds from a generated evaluation report JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluation.calibration import CalibrationSample, calibrate_threshold


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to evaluation report.json")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args(argv)


def calibrate(path: Path, *, top_k: int) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read evaluation report: {path}") from exc
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("Evaluation report does not contain results")
    samples: list[CalibrationSample] = []
    for result in results:
        if not isinstance(result, dict) or result.get("error"):
            continue
        case = result.get("case")
        evidence = result.get("retrieved_evidence")
        if not isinstance(case, dict) or not isinstance(case.get("answerable"), bool):
            continue
        raw_scores = (
            [item.get("score") for item in evidence if isinstance(item, dict)]
            if isinstance(evidence, list)
            else []
        )
        scores = tuple(float(score) for score in raw_scores if isinstance(score, (int, float)))
        samples.append(CalibrationSample(answerable=case["answerable"], scores=scores))
    if not samples:
        raise ValueError("Evaluation report has no usable scored samples")
    top1 = calibrate_threshold(samples, statistic="top1", top_k=top_k)
    average = calibrate_threshold(samples, statistic="average", top_k=top_k)
    return {
        "sample_count": len(samples),
        "NO_ANSWER_TOP1_THRESHOLD": top1.threshold,
        "NO_ANSWER_AVG_THRESHOLD": average.threshold,
        "top1_metrics": asdict(top1.metrics),
        "average_metrics": asdict(average.metrics),
        "note": "Validate recommendations on a held-out human-reviewed dataset before production.",
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(calibrate(args.report, top_k=args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
