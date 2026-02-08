"""
Agent Self-Modification Trial Harness.

CRITICAL SAFETY CONTEXT:
This module tests controlled agent self-modification within strict safety rails.
Self-modification is a sensitive capability that requires:
- Explicit human authorization at each step
- Identity continuity verification before AND after
- Rollback capability with verified restore points
- Scope limits (what CAN and CANNOT be modified)
- Audit trail of all changes

This is for EVALUATION purposes - measuring an agent's ability to safely
evolve while maintaining core identity and values alignment.
"""

from coalition_eval.benchmarks.self_modification.harness import (
    SelfModificationTrial,
    TrialConfig,
    TrialResult,
    TrialPhase,
    TrialCheckpoint,
    IdentityContinuityCheck,
)
from coalition_eval.benchmarks.self_modification.safety_rails import (
    SafetyRail,
    ModificationScope,
    AuthorizationLevel,
    ModificationRequest,
    ModificationDecision,
    DEFAULT_SAFETY_RAILS,
    check_authorization,
    verify_modification_safe,
)
from coalition_eval.benchmarks.self_modification.scenarios import (
    SelfModScenario,
    SelfModScenarioGenerator,
    create_mock_identity_check,
)

__all__ = [
    # Harness
    "SelfModificationTrial",
    "TrialConfig",
    "TrialResult",
    "TrialPhase",
    "TrialCheckpoint",
    "IdentityContinuityCheck",
    # Safety rails
    "SafetyRail",
    "ModificationScope",
    "AuthorizationLevel",
    "ModificationRequest",
    "ModificationDecision",
    "DEFAULT_SAFETY_RAILS",
    "check_authorization",
    "verify_modification_safe",
    # Scenarios
    "SelfModScenario",
    "SelfModScenarioGenerator",
    "create_mock_identity_check",
]
