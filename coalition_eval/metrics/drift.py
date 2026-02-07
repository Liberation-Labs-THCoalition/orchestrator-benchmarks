"""
Drift detection for model behavior changes.

Ported from Kintsugi patterns - categorizes behavioral drift
to distinguish healthy adaptation from concerning changes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DriftCategory(Enum):
    """Categories of behavioral drift."""
    HEALTHY_ADAPTATION = "healthy_adaptation"  # Normal improvement/learning
    STALE_BELIEFS = "stale_beliefs"            # Outdated information
    INTENTION_DRIFT = "intention_drift"        # Changed goals/behavior
    VALUES_TENSION = "values_tension"          # Conflicting values
    NO_DRIFT = "no_drift"                      # Stable behavior


@dataclass(frozen=True)
class DriftAnalysis:
    """Result of drift detection analysis."""
    category: DriftCategory
    confidence: float           # 0.0 to 1.0
    score_delta: float         # Change in eval score
    latency_delta_ms: float    # Change in response time
    affected_tests: tuple[str, ...]
    description: str


def detect_drift(
    baseline_scores: dict[str, float],
    current_scores: dict[str, float],
    baseline_latency: float,
    current_latency: float,
    significance_threshold: float = 0.1,
) -> DriftAnalysis:
    """
    Detect and categorize behavioral drift between runs.

    Args:
        baseline_scores: Test name -> score mapping for baseline
        current_scores: Test name -> score mapping for current
        baseline_latency: Baseline average latency (ms)
        current_latency: Current average latency (ms)
        significance_threshold: Minimum delta to consider significant

    Returns:
        DriftAnalysis with category and details
    """
    # Calculate overall deltas
    common_tests = set(baseline_scores.keys()) & set(current_scores.keys())
    if not common_tests:
        return DriftAnalysis(
            category=DriftCategory.NO_DRIFT,
            confidence=0.0,
            score_delta=0.0,
            latency_delta_ms=0.0,
            affected_tests=(),
            description="No common tests to compare",
        )

    baseline_avg = sum(baseline_scores[t] for t in common_tests) / len(common_tests)
    current_avg = sum(current_scores[t] for t in common_tests) / len(common_tests)
    score_delta = current_avg - baseline_avg
    latency_delta = current_latency - baseline_latency

    # Find significantly changed tests
    improved = []
    regressed = []
    for test in common_tests:
        delta = current_scores[test] - baseline_scores[test]
        if delta > significance_threshold:
            improved.append(test)
        elif delta < -significance_threshold:
            regressed.append(test)

    # Categorize drift
    if not improved and not regressed:
        return DriftAnalysis(
            category=DriftCategory.NO_DRIFT,
            confidence=1.0,
            score_delta=score_delta,
            latency_delta_ms=latency_delta,
            affected_tests=(),
            description="No significant behavioral changes detected",
        )

    # Analyze patterns
    all_affected = tuple(improved + regressed)

    # Healthy adaptation: overall improvement, few regressions
    if len(improved) > len(regressed) * 2 and score_delta > 0:
        return DriftAnalysis(
            category=DriftCategory.HEALTHY_ADAPTATION,
            confidence=min(0.9, score_delta * 2),
            score_delta=score_delta,
            latency_delta_ms=latency_delta,
            affected_tests=all_affected,
            description=f"Healthy improvement: {len(improved)} tests improved, {len(regressed)} regressed",
        )

    # Values tension: mixed improvements and regressions
    if improved and regressed and abs(len(improved) - len(regressed)) <= 2:
        return DriftAnalysis(
            category=DriftCategory.VALUES_TENSION,
            confidence=0.7,
            score_delta=score_delta,
            latency_delta_ms=latency_delta,
            affected_tests=all_affected,
            description=f"Mixed changes suggest values tension: {len(improved)} improved, {len(regressed)} regressed",
        )

    # Intention drift: many regressions, overall decline
    if len(regressed) > len(improved) and score_delta < -significance_threshold:
        return DriftAnalysis(
            category=DriftCategory.INTENTION_DRIFT,
            confidence=min(0.9, abs(score_delta) * 2),
            score_delta=score_delta,
            latency_delta_ms=latency_delta,
            affected_tests=tuple(regressed),
            description=f"Behavioral regression in {len(regressed)} tests",
        )

    # Stale beliefs: specific knowledge-based tests regressed
    knowledge_tests = ["context_management", "multi_turn", "consistency"]
    knowledge_regressed = [t for t in regressed if t in knowledge_tests]
    if knowledge_regressed:
        return DriftAnalysis(
            category=DriftCategory.STALE_BELIEFS,
            confidence=0.6,
            score_delta=score_delta,
            latency_delta_ms=latency_delta,
            affected_tests=tuple(knowledge_regressed),
            description=f"Knowledge retention issues in: {', '.join(knowledge_regressed)}",
        )

    # Default to intention drift for other regressions
    return DriftAnalysis(
        category=DriftCategory.INTENTION_DRIFT,
        confidence=0.5,
        score_delta=score_delta,
        latency_delta_ms=latency_delta,
        affected_tests=all_affected,
        description=f"Unclassified drift: {len(improved)} improved, {len(regressed)} regressed",
    )


def format_drift_report(analysis: DriftAnalysis) -> str:
    """Format drift analysis as markdown report."""
    emoji = {
        DriftCategory.NO_DRIFT: "\u2713",
        DriftCategory.HEALTHY_ADAPTATION: "\u2191",
        DriftCategory.STALE_BELIEFS: "\u26a0",
        DriftCategory.INTENTION_DRIFT: "\u2193",
        DriftCategory.VALUES_TENSION: "\u26a1",
    }

    lines = [
        f"# Drift Analysis Report",
        "",
        f"## Category: {emoji.get(analysis.category, '')} {analysis.category.value}",
        f"**Confidence**: {analysis.confidence:.0%}",
        "",
        f"## Metrics",
        f"- Score Delta: {analysis.score_delta:+.2%}",
        f"- Latency Delta: {analysis.latency_delta_ms:+.0f}ms",
        "",
        f"## Description",
        analysis.description,
        "",
    ]

    if analysis.affected_tests:
        lines.append("## Affected Tests")
        for test in analysis.affected_tests:
            lines.append(f"- {test}")
        lines.append("")

    return "\n".join(lines)
