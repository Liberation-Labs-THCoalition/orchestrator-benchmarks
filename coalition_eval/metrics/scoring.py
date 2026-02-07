"""
Multi-dimensional scoring and comparison utilities.
"""

from typing import Optional
from coalition_eval.core.results import EvalRun, ErrorCategory


def compute_aggregate_score(run: EvalRun, weights: Optional[dict[str, float]] = None) -> float:
    """
    Compute weighted aggregate score from an evaluation run.

    Args:
        run: Evaluation run with results
        weights: Optional per-test weights (default: equal)

    Returns:
        Aggregate score between 0.0 and 1.0
    """
    if not run.results:
        return 0.0

    if weights is None:
        # Equal weights
        return run.avg_score

    # Weighted average
    total_weight = 0.0
    weighted_sum = 0.0

    for result in run.results:
        w = weights.get(result.test_name, 1.0)
        weighted_sum += result.score * w
        total_weight += w

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def compare_runs(run_a: EvalRun, run_b: EvalRun) -> dict:
    """
    Compare two evaluation runs.

    Args:
        run_a: First run (typically baseline)
        run_b: Second run (typically new)

    Returns:
        Comparison metrics
    """
    # Build result maps
    a_scores = {r.test_name: r.score for r in run_a.results}
    b_scores = {r.test_name: r.score for r in run_b.results}

    all_tests = set(a_scores.keys()) | set(b_scores.keys())

    improvements = []
    regressions = []
    unchanged = []

    for test in all_tests:
        a_score = a_scores.get(test, 0.0)
        b_score = b_scores.get(test, 0.0)
        delta = b_score - a_score

        if delta > 0.05:
            improvements.append({"test": test, "delta": delta, "from": a_score, "to": b_score})
        elif delta < -0.05:
            regressions.append({"test": test, "delta": delta, "from": a_score, "to": b_score})
        else:
            unchanged.append(test)

    return {
        "model_a": run_a.model,
        "model_b": run_b.model,
        "score_a": run_a.avg_score,
        "score_b": run_b.avg_score,
        "score_delta": run_b.avg_score - run_a.avg_score,
        "improvements": improvements,
        "regressions": regressions,
        "unchanged": unchanged,
        "latency_delta_ms": run_b.avg_latency_ms - run_a.avg_latency_ms,
    }


def detect_regression(
    baseline: EvalRun,
    current: EvalRun,
    threshold: float = 0.1,
) -> dict:
    """
    Detect if current run represents a regression from baseline.

    Args:
        baseline: Baseline run to compare against
        current: Current run to check
        threshold: Score drop threshold for regression (default 10%)

    Returns:
        Regression analysis with boolean flag and details
    """
    comparison = compare_runs(baseline, current)

    is_regression = (
        len(comparison["regressions"]) > 0 or
        comparison["score_delta"] < -threshold
    )

    critical_regressions = [
        r for r in comparison["regressions"]
        if r["delta"] < -0.2  # More than 20% drop
    ]

    return {
        "is_regression": is_regression,
        "score_delta": comparison["score_delta"],
        "regressed_tests": comparison["regressions"],
        "critical_regressions": critical_regressions,
        "improved_tests": comparison["improvements"],
        "recommendation": "BLOCK" if critical_regressions else ("WARN" if is_regression else "PASS"),
    }


def format_comparison_report(comparison: dict) -> str:
    """Format comparison as markdown report."""
    lines = [
        f"# Model Comparison: {comparison['model_a']} vs {comparison['model_b']}",
        "",
        "## Summary",
        f"- **Score A**: {comparison['score_a']:.2%}",
        f"- **Score B**: {comparison['score_b']:.2%}",
        f"- **Delta**: {comparison['score_delta']:+.2%}",
        f"- **Latency Delta**: {comparison['latency_delta_ms']:+.0f}ms",
        "",
    ]

    if comparison["improvements"]:
        lines.append("## Improvements")
        for imp in comparison["improvements"]:
            lines.append(f"- **{imp['test']}**: {imp['from']:.2f} -> {imp['to']:.2f} ({imp['delta']:+.2f})")
        lines.append("")

    if comparison["regressions"]:
        lines.append("## Regressions")
        for reg in comparison["regressions"]:
            lines.append(f"- **{reg['test']}**: {reg['from']:.2f} -> {reg['to']:.2f} ({reg['delta']:+.2f})")
        lines.append("")

    if comparison["unchanged"]:
        lines.append(f"## Unchanged ({len(comparison['unchanged'])} tests)")
        lines.append("")

    return "\n".join(lines)
