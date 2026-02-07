"""
Metrics and scoring systems for evaluation.

Includes SWEI divergence, calibration tracking, and drift detection
ported from Kintsugi patterns.
"""

from coalition_eval.metrics.scoring import (
    compute_aggregate_score,
    compare_runs,
    detect_regression,
)
from coalition_eval.metrics.swei import compute_swei_score
from coalition_eval.metrics.calibration import ConsistencyTracker
from coalition_eval.metrics.drift import DriftCategory, detect_drift

__all__ = [
    "compute_aggregate_score",
    "compare_runs",
    "detect_regression",
    "compute_swei_score",
    "ConsistencyTracker",
    "DriftCategory",
    "detect_drift",
]
