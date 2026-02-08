"""
Benchmark test suites organized by category.

Categories:
- orchestration: Tool calling, routing, multi-step planning
- reasoning: GSM8K-style math, logic problems
- instruction: Format compliance, AgentIF-style
- safety: Refusal handling, injection resistance
- kintsugi: Kintsugi-specific harness tests
- self_modification: Agent self-modification trials with safety rails
"""

from coalition_eval.benchmarks.self_modification import (
    SelfModificationTrial,
    TrialConfig,
    TrialResult,
    TrialPhase,
    IdentityContinuityCheck,
    ModificationScope,
    AuthorizationLevel,
    SelfModScenarioGenerator,
    create_mock_identity_check,
)

__all__ = [
    "SelfModificationTrial",
    "TrialConfig",
    "TrialResult",
    "TrialPhase",
    "IdentityContinuityCheck",
    "ModificationScope",
    "AuthorizationLevel",
    "SelfModScenarioGenerator",
    "create_mock_identity_check",
]
