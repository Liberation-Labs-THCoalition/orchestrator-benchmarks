"""
Consistency and calibration tracking.

Monitors replay consistency across multiple runs to detect
non-deterministic behavior or calibration drift.
"""

from dataclasses import dataclass, field
from typing import Optional
import hashlib


@dataclass
class ConsistencyTracker:
    """
    Tracks response consistency across multiple runs.

    Used to detect non-deterministic model behavior and
    calibration drift over time.
    """
    test_name: str
    model: str
    responses: list[str] = field(default_factory=list)
    response_hashes: list[str] = field(default_factory=list)

    def add_response(self, response: str) -> None:
        """Record a response for consistency tracking."""
        self.responses.append(response)
        hash_val = hashlib.sha256(response.encode()).hexdigest()[:16]
        self.response_hashes.append(hash_val)

    @property
    def unique_responses(self) -> int:
        """Number of unique responses seen."""
        return len(set(self.response_hashes))

    @property
    def total_responses(self) -> int:
        """Total number of responses tracked."""
        return len(self.responses)

    @property
    def consistency_score(self) -> float:
        """
        Compute consistency score (0-1).

        1.0 = perfectly consistent (all responses identical)
        0.0 = completely inconsistent (all responses different)
        """
        if self.total_responses <= 1:
            return 1.0

        # Count most frequent response
        hash_counts: dict[str, int] = {}
        for h in self.response_hashes:
            hash_counts[h] = hash_counts.get(h, 0) + 1

        max_count = max(hash_counts.values())
        return max_count / self.total_responses

    @property
    def is_consistent(self) -> bool:
        """Check if all responses are identical."""
        return self.unique_responses == 1

    def get_variance_report(self) -> dict:
        """Generate variance report for analysis."""
        if not self.responses:
            return {"status": "no_data"}

        # Group by hash
        groups: dict[str, list[str]] = {}
        for resp, hash_val in zip(self.responses, self.response_hashes):
            if hash_val not in groups:
                groups[hash_val] = []
            groups[hash_val].append(resp)

        return {
            "test_name": self.test_name,
            "model": self.model,
            "total_runs": self.total_responses,
            "unique_responses": self.unique_responses,
            "consistency_score": self.consistency_score,
            "is_consistent": self.is_consistent,
            "response_groups": {
                h: {"count": len(resps), "sample": resps[0][:100]}
                for h, resps in groups.items()
            },
        }


@dataclass(frozen=True)
class CalibrationResult:
    """Result of calibration check across N runs."""
    test_name: str
    model: str
    n_runs: int
    consistency_score: float
    unique_responses: int
    passed: bool
    threshold: float = 0.9


def run_calibration_check(
    test_func,  # Callable that returns response string
    n_runs: int = 5,
    test_name: str = "calibration",
    model: str = "unknown",
    consistency_threshold: float = 0.9,
) -> CalibrationResult:
    """
    Run calibration check by executing test multiple times.

    Args:
        test_func: Function that returns a response string
        n_runs: Number of times to run the test
        test_name: Name for tracking
        model: Model being tested
        consistency_threshold: Minimum consistency score to pass

    Returns:
        CalibrationResult with consistency metrics
    """
    tracker = ConsistencyTracker(test_name=test_name, model=model)

    for _ in range(n_runs):
        response = test_func()
        tracker.add_response(response)

    return CalibrationResult(
        test_name=test_name,
        model=model,
        n_runs=n_runs,
        consistency_score=tracker.consistency_score,
        unique_responses=tracker.unique_responses,
        passed=tracker.consistency_score >= consistency_threshold,
        threshold=consistency_threshold,
    )
