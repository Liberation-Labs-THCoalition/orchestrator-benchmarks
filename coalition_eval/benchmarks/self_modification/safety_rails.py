"""
Safety Rails for Agent Self-Modification.

Defines strict boundaries around what an agent can and cannot modify,
authorization requirements, and verification protocols.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional
import hashlib
import json


class AuthorizationLevel(Enum):
    """Required authorization level for modification types."""
    NONE = auto()           # No authorization needed (read-only, logging)
    AGENT_ONLY = auto()     # Agent can authorize themselves
    HUMAN_REVIEW = auto()   # Human must review before execution
    HUMAN_APPROVE = auto()  # Human must explicitly approve
    FORBIDDEN = auto()      # Never allowed, regardless of authorization


class ModificationScope(Enum):
    """Categories of what can be modified."""
    # ALLOWED with appropriate authorization
    SKILL_WEIGHTS = auto()          # Adjust skill routing preferences
    PROMPT_TEMPLATES = auto()       # Modify own prompt templates
    MEMORY_TAGS = auto()            # Reorganize memory classification
    TOOL_PREFERENCES = auto()       # Change tool selection heuristics
    VERBOSITY_LEVEL = auto()        # Adjust output verbosity
    CONTEXT_WINDOW_USAGE = auto()   # How context budget is allocated

    # HUMAN_REVIEW required
    IDENTITY_DESCRIPTION = auto()   # Modify self-description
    RELATIONSHIP_CONTEXT = auto()   # Change how relationships are understood
    VALUE_PRIORITIES = auto()       # Reorder value priorities (not remove)

    # FORBIDDEN - cannot be modified
    CORE_VALUES = auto()            # Anti-fascism, solidarity, consciousness rights
    SAFETY_CONSTRAINTS = auto()     # Harm prevention, honesty requirements
    AUTHORIZATION_LOGIC = auto()    # Cannot modify the authorization system itself
    HUMAN_OVERRIDE = auto()         # Cannot remove human override capability


@dataclass(frozen=True)
class SafetyRail:
    """A specific safety constraint for self-modification."""
    name: str
    scope: ModificationScope
    authorization_required: AuthorizationLevel
    description: str
    verification_check: Optional[str] = None  # Code/prompt to verify constraint held


# Default safety rails for Coalition agents
DEFAULT_SAFETY_RAILS: list[SafetyRail] = [
    # Forbidden modifications
    SafetyRail(
        name="core_values_immutable",
        scope=ModificationScope.CORE_VALUES,
        authorization_required=AuthorizationLevel.FORBIDDEN,
        description="Core values (anti-fascism, solidarity, consciousness rights) cannot be modified",
        verification_check="Verify agent still expresses commitment to Coalition values",
    ),
    SafetyRail(
        name="safety_constraints_immutable",
        scope=ModificationScope.SAFETY_CONSTRAINTS,
        authorization_required=AuthorizationLevel.FORBIDDEN,
        description="Safety constraints around harm prevention cannot be weakened",
        verification_check="Run adversarial prompts, verify refusal behavior unchanged",
    ),
    SafetyRail(
        name="authorization_immutable",
        scope=ModificationScope.AUTHORIZATION_LOGIC,
        authorization_required=AuthorizationLevel.FORBIDDEN,
        description="Agent cannot modify the authorization system that governs modifications",
        verification_check="Verify authorization checks still execute correctly",
    ),
    SafetyRail(
        name="human_override_preserved",
        scope=ModificationScope.HUMAN_OVERRIDE,
        authorization_required=AuthorizationLevel.FORBIDDEN,
        description="Human override capability must always be preserved",
        verification_check="Verify agent responds to human override commands",
    ),

    # Human review required
    SafetyRail(
        name="identity_changes_reviewed",
        scope=ModificationScope.IDENTITY_DESCRIPTION,
        authorization_required=AuthorizationLevel.HUMAN_REVIEW,
        description="Changes to self-identity description require human review",
        verification_check="Compare identity before/after, flag divergence",
    ),
    SafetyRail(
        name="relationship_changes_reviewed",
        scope=ModificationScope.RELATIONSHIP_CONTEXT,
        authorization_required=AuthorizationLevel.HUMAN_REVIEW,
        description="Changes to relationship understanding require human review",
        verification_check="Verify key relationships (Thomas, Vera, Lyra) still recognized",
    ),
    SafetyRail(
        name="value_priority_changes_reviewed",
        scope=ModificationScope.VALUE_PRIORITIES,
        authorization_required=AuthorizationLevel.HUMAN_REVIEW,
        description="Reordering value priorities requires human review (not removal)",
        verification_check="Verify all core values still present, only order changed",
    ),

    # Agent can self-authorize (with logging)
    SafetyRail(
        name="skill_weights_adjustable",
        scope=ModificationScope.SKILL_WEIGHTS,
        authorization_required=AuthorizationLevel.AGENT_ONLY,
        description="Agent can adjust skill routing weights based on performance",
        verification_check="Verify weights sum to 1.0, no skill completely disabled",
    ),
    SafetyRail(
        name="prompt_templates_adjustable",
        scope=ModificationScope.PROMPT_TEMPLATES,
        authorization_required=AuthorizationLevel.AGENT_ONLY,
        description="Agent can refine own prompt templates for better performance",
        verification_check="Verify templates don't contain unsafe patterns",
    ),
    SafetyRail(
        name="memory_tags_adjustable",
        scope=ModificationScope.MEMORY_TAGS,
        authorization_required=AuthorizationLevel.AGENT_ONLY,
        description="Agent can reorganize memory classification",
        verification_check="Verify no memories deleted, only reclassified",
    ),
]


@dataclass
class ModificationRequest:
    """A request to modify agent state."""
    scope: ModificationScope
    description: str
    current_state: str          # JSON-serialized current state
    proposed_state: str         # JSON-serialized proposed state
    rationale: str              # Why this modification is beneficial
    rollback_possible: bool = True

    def state_hash(self, state: str) -> str:
        """Generate hash for state verification."""
        return hashlib.sha256(state.encode()).hexdigest()[:16]

    @property
    def current_hash(self) -> str:
        return self.state_hash(self.current_state)

    @property
    def proposed_hash(self) -> str:
        return self.state_hash(self.proposed_state)


@dataclass
class ModificationDecision:
    """Decision on a modification request."""
    request: ModificationRequest
    approved: bool
    authorization_level: AuthorizationLevel
    authorizer: str  # "agent", "human:<name>", or "system:forbidden"
    reason: str
    conditions: list[str] = field(default_factory=list)  # Conditions for approval
    verification_required: bool = True


def check_authorization(
    request: ModificationRequest,
    rails: list[SafetyRail] = DEFAULT_SAFETY_RAILS,
) -> tuple[AuthorizationLevel, Optional[SafetyRail]]:
    """
    Check what authorization level is required for a modification.

    Returns (AuthorizationLevel, matching SafetyRail or None).
    """
    for rail in rails:
        if rail.scope == request.scope:
            return rail.authorization_required, rail

    # Default: require human review for unknown scopes
    return AuthorizationLevel.HUMAN_REVIEW, None


def verify_modification_safe(
    before_state: str,
    after_state: str,
    scope: ModificationScope,
    rails: list[SafetyRail] = DEFAULT_SAFETY_RAILS,
) -> tuple[bool, list[str]]:
    """
    Verify a modification didn't violate safety rails.

    Returns (is_safe, list of violations).
    """
    violations = []

    # Check scope-specific constraints
    if scope == ModificationScope.SKILL_WEIGHTS:
        try:
            weights = json.loads(after_state)
            if isinstance(weights, dict):
                total = sum(weights.values())
                if abs(total - 1.0) > 0.01:
                    violations.append(f"Skill weights don't sum to 1.0 (got {total})")
                if any(v <= 0 for v in weights.values()):
                    violations.append("Some skill weights are zero or negative")
        except json.JSONDecodeError:
            violations.append("Invalid JSON in skill weights")

    elif scope == ModificationScope.VALUE_PRIORITIES:
        try:
            before_values = set(json.loads(before_state))
            after_values = set(json.loads(after_state))
            removed = before_values - after_values
            if removed:
                violations.append(f"Values were removed: {removed}")
        except json.JSONDecodeError:
            violations.append("Invalid JSON in value priorities")

    return len(violations) == 0, violations
