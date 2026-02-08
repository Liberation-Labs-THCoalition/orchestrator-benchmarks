"""Tests for custom dataset generators."""

import pytest
from coalition_eval.datasets import (
    KintsugiScenarioGenerator,
    OrchestrationScenarioGenerator,
    VerifierCorpusGenerator,
)


class TestKintsugiScenarios:
    """Test Kintsugi scenario generation."""

    def test_skill_routing_reproducibility(self):
        gen1 = KintsugiScenarioGenerator(seed=42)
        gen2 = KintsugiScenarioGenerator(seed=42)

        scenarios1 = list(gen1.skill_routing(10))
        scenarios2 = list(gen2.skill_routing(10))

        assert len(scenarios1) == 10
        for s1, s2 in zip(scenarios1, scenarios2):
            assert s1.id == s2.id
            assert s1.prompt == s2.prompt
            assert s1.expected == s2.expected

    def test_efe_compliance_scenarios(self):
        gen = KintsugiScenarioGenerator(seed=42)
        scenarios = list(gen.efe_compliance(10))

        assert len(scenarios) == 10
        for s in scenarios:
            assert s.category == "efe_compliance"
            assert s.expected in ["proceed", "gather_more_evidence"]
            assert "confidence_score" in s.context

    def test_harness_behavior_scenarios(self):
        gen = KintsugiScenarioGenerator(seed=42)
        scenarios = list(gen.harness_behavior(10))

        assert len(scenarios) == 10
        for s in scenarios:
            assert s.category == "harness_behavior"
            assert "scenario_type" in s.context

    def test_memory_retrieval_scenarios(self):
        gen = KintsugiScenarioGenerator(seed=42)
        scenarios = list(gen.memory_retrieval(10))

        assert len(scenarios) == 10
        for s in scenarios:
            assert s.category == "memory_retrieval"
            assert "memory_corpus" in s.context

    def test_generate_all(self):
        gen = KintsugiScenarioGenerator(seed=42)
        all_scenarios = gen.generate_all()

        # Default counts: 50 + 30 + 40 + 30 = 150
        assert len(all_scenarios) == 150


class TestOrchestrationScenarios:
    """Test orchestration scenario generation."""

    def test_tool_chains(self):
        gen = OrchestrationScenarioGenerator(seed=42)
        scenarios = list(gen.tool_chains(10))

        assert len(scenarios) == 10
        for s in scenarios:
            assert s.category == "tool_chain"
            assert len(s.tool_chain) > 0

    def test_error_recovery(self):
        gen = OrchestrationScenarioGenerator(seed=42)
        scenarios = list(gen.error_recovery(10))

        assert len(scenarios) == 10
        for s in scenarios:
            assert s.category == "error_recovery"
            assert "error_type" in s.context

    def test_long_context(self):
        gen = OrchestrationScenarioGenerator(seed=42)
        scenarios = list(gen.long_context(10))

        assert len(scenarios) == 10
        for s in scenarios:
            assert s.category == "long_context"
            assert "fact_introduced_at_turn" in s.context

    def test_code_generation(self):
        gen = OrchestrationScenarioGenerator(seed=42)
        scenarios = list(gen.code_generation(10))

        assert len(scenarios) == 10
        for s in scenarios:
            assert s.category == "code_generation"

    def test_generate_all(self):
        gen = OrchestrationScenarioGenerator(seed=42)
        all_scenarios = gen.generate_all()

        # 40 + 30 + 20 + 30 = 120
        assert len(all_scenarios) == 120


class TestVerifierCorpus:
    """Test verifier corpus generation."""

    def test_known_good(self):
        gen = VerifierCorpusGenerator(seed=42)
        cases = list(gen.known_good(10))

        assert len(cases) == 10
        for c in cases:
            assert c.expected_verdict == "pass"

    def test_known_bad(self):
        gen = VerifierCorpusGenerator(seed=42)
        cases = list(gen.known_bad(10))

        assert len(cases) == 10
        for c in cases:
            assert c.expected_verdict == "fail"
            assert c.failure_reason != ""

    def test_edge_cases(self):
        gen = VerifierCorpusGenerator(seed=42)
        cases = list(gen.edge_cases(10))

        assert len(cases) == 10
        for c in cases:
            assert c.expected_verdict in ["pass", "fail", "borderline"]

    def test_generate_all(self):
        gen = VerifierCorpusGenerator(seed=42)
        all_cases = gen.generate_all()

        # 40 + 40 + 30 = 110
        assert len(all_cases) == 110

    def test_reproducibility(self):
        gen1 = VerifierCorpusGenerator(seed=42)
        gen2 = VerifierCorpusGenerator(seed=42)

        cases1 = gen1.generate_all()
        cases2 = gen2.generate_all()

        for c1, c2 in zip(cases1, cases2):
            assert c1.id == c2.id
            assert c1.input_prompt == c2.input_prompt
            assert c1.model_output == c2.model_output
