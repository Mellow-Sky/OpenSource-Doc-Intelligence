from __future__ import annotations

from evaluation.calibration import CalibrationSample, calibrate_threshold


def test_threshold_calibration_separates_answerable_and_unanswerable_scores() -> None:
    samples = [
        CalibrationSample(answerable=True, scores=(0.9, 0.8)),
        CalibrationSample(answerable=True, scores=(0.7, 0.6)),
        CalibrationSample(answerable=False, scores=(0.3, 0.2)),
        CalibrationSample(answerable=False, scores=()),
    ]

    recommendation = calibrate_threshold(samples, statistic="top1")

    assert recommendation.threshold is not None
    assert 0.3 < recommendation.threshold <= 0.7
    assert recommendation.metrics.f1 == 1.0
    assert recommendation.sample_count == 4
