"""Basic tests for coalition_eval package."""

import pytest
from coalition_eval.core.results import EvalResult, ErrorCategory
from coalition_eval.datasets.mock_data import MockDataGenerator
from coalition_eval.metrics.swei import compute_swei_score
from coalition_eval.metrics.calibration import ConsistencyTracker


class TestEvalResult:
    """Test EvalResult dataclass."""

    def test_create_result(self):
        result = EvalResult(
            test_name="test_basic",
            model="qwen2.5:14b",
            passed=True,
            score=0.95,
            latency_ms=150.5,
            tokens_used=100,
            tokens_per_sec=50.0,
            response_text="Test response",
        )
        assert result.test_name == "test_basic"
        assert result.passed is True
        assert result.score == 0.95

    def test_score_clamping(self):
        # Score should be clamped to [0, 1]
        result = EvalResult(
            test_name="test",
            model="test",
            passed=True,
            score=1.5,  # Out of range
            latency_ms=100,
            tokens_used=50,
            tokens_per_sec=25.0,
            response_text="",
        )
        assert result.score == 1.0

    def test_to_dict(self):
        result = EvalResult(
            test_name="test",
            model="test",
            passed=True,
            score=0.8,
            latency_ms=100,
            tokens_used=50,
            tokens_per_sec=25.0,
            response_text="response",
        )
        d = result.to_dict()
        assert d["test_name"] == "test"
        assert d["passed"] is True
        assert "result_id" in d


class TestMockDataGenerator:
    """Test mock data generation."""

    def test_arithmetic_reproducibility(self):
        gen1 = MockDataGenerator(seed=42)
        gen2 = MockDataGenerator(seed=42)

        samples1 = list(gen1.arithmetic(10))
        samples2 = list(gen2.arithmetic(10))

        assert len(samples1) == 10
        for s1, s2 in zip(samples1, samples2):
            assert s1.prompt == s2.prompt
            assert s1.expected == s2.expected

    def test_tool_selection(self):
        gen = MockDataGenerator(seed=42)
        samples = list(gen.tool_selection(5))

        assert len(samples) == 5
        for s in samples:
            assert s.category == "tool_selection"
            assert s.expected in ["calculator", "search", "weather", "calendar", "email"]


class TestSWEIScore:
    """Test SWEI scoring."""

    def test_identical_texts(self):
        score = compute_swei_score("hello world", "hello world")
        assert score.overall < 1e-10  # Essentially zero (floating point)

    def test_completely_different(self):
        score = compute_swei_score("hello", "goodbye")
        assert score.overall > 0.5  # High divergence

    def test_partial_overlap(self):
        score = compute_swei_score("the quick brown fox", "the slow brown fox")
        assert 0.0 < score.overall < 0.5


class TestConsistencyTracker:
    """Test consistency tracking."""

    def test_consistent_responses(self):
        tracker = ConsistencyTracker(test_name="test", model="test")
        tracker.add_response("hello")
        tracker.add_response("hello")
        tracker.add_response("hello")

        assert tracker.is_consistent
        assert tracker.consistency_score == 1.0

    def test_inconsistent_responses(self):
        tracker = ConsistencyTracker(test_name="test", model="test")
        tracker.add_response("hello")
        tracker.add_response("world")
        tracker.add_response("foo")

        assert not tracker.is_consistent
        assert tracker.consistency_score < 1.0
        assert tracker.unique_responses == 3
