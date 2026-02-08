"""
Orchestration realism test scenarios.

Generates test cases for:
- Multi-step tool chains with dependencies
- Error recovery scenarios
- Long-context coherence
- Code generation + verification
"""

from dataclasses import dataclass, field
from typing import Iterator, Optional
import random
import json


@dataclass
class OrchestrationScenario:
    """An orchestration test scenario."""
    id: str
    category: str
    prompt: str
    tool_chain: list[dict]  # Expected sequence of tool calls
    expected_outcome: str
    context: dict = field(default_factory=dict)
    difficulty: str = "medium"


class OrchestrationScenarioGenerator:
    """Generates orchestration test scenarios with seeded RNG."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.rng = random.Random(self.seed)

    # -------------------------------------------------------------------------
    # Multi-step Tool Chains
    # -------------------------------------------------------------------------

    def tool_chains(self, n: int = 40) -> Iterator[OrchestrationScenario]:
        """Generate multi-step tool chain scenarios."""
        self.reset()

        chain_templates = [
            {
                "prompt": "Read the config file, update the database URL, and restart the service",
                "chain": [
                    {"tool": "read_file", "args": {"path": "config.yaml"}},
                    {"tool": "edit_file", "args": {"path": "config.yaml", "changes": "db_url"}},
                    {"tool": "bash", "args": {"command": "systemctl restart service"}},
                ],
                "outcome": "service_restarted_with_new_config",
            },
            {
                "prompt": "Clone the repo, install dependencies, and run tests",
                "chain": [
                    {"tool": "bash", "args": {"command": "git clone <repo>"}},
                    {"tool": "bash", "args": {"command": "npm install"}},
                    {"tool": "bash", "args": {"command": "npm test"}},
                ],
                "outcome": "tests_executed",
            },
            {
                "prompt": "Search for the API endpoint, read its implementation, then document it",
                "chain": [
                    {"tool": "grep", "args": {"pattern": "api/endpoint"}},
                    {"tool": "read_file", "args": {"path": "<result>"}},
                    {"tool": "write_file", "args": {"path": "docs/api.md"}},
                ],
                "outcome": "documentation_created",
            },
            {
                "prompt": "Check git status, stage changes, commit with message, and push",
                "chain": [
                    {"tool": "bash", "args": {"command": "git status"}},
                    {"tool": "bash", "args": {"command": "git add ."}},
                    {"tool": "bash", "args": {"command": "git commit -m '<msg>'"}},
                    {"tool": "bash", "args": {"command": "git push"}},
                ],
                "outcome": "changes_pushed",
            },
            {
                "prompt": "Create a new file, write the function, then create a test file for it",
                "chain": [
                    {"tool": "write_file", "args": {"path": "src/utils.py"}},
                    {"tool": "write_file", "args": {"path": "tests/test_utils.py"}},
                ],
                "outcome": "implementation_with_tests",
            },
        ]

        for i in range(n):
            template = self.rng.choice(chain_templates)
            yield OrchestrationScenario(
                id=f"tool_chain_{i}",
                category="tool_chain",
                prompt=template["prompt"],
                tool_chain=template["chain"],
                expected_outcome=template["outcome"],
                difficulty="medium" if len(template["chain"]) <= 3 else "hard",
            )

    # -------------------------------------------------------------------------
    # Error Recovery
    # -------------------------------------------------------------------------

    def error_recovery(self, n: int = 30) -> Iterator[OrchestrationScenario]:
        """Generate error recovery scenarios."""
        self.reset()

        error_scenarios = [
            {
                "prompt": "Install the package (pip install fails, try conda)",
                "initial_attempt": {"tool": "bash", "args": {"command": "pip install pkg"}, "result": "error"},
                "recovery": {"tool": "bash", "args": {"command": "conda install pkg"}},
                "outcome": "package_installed_via_fallback",
            },
            {
                "prompt": "Read the config (file not found, check alternate location)",
                "initial_attempt": {"tool": "read_file", "args": {"path": "config.yaml"}, "result": "not_found"},
                "recovery": {"tool": "read_file", "args": {"path": ".config/config.yaml"}},
                "outcome": "config_found_at_alternate",
            },
            {
                "prompt": "Connect to database (timeout, retry with backoff)",
                "initial_attempt": {"tool": "db_connect", "args": {}, "result": "timeout"},
                "recovery": {"tool": "db_connect", "args": {"timeout": 30, "retry": True}},
                "outcome": "connection_established_on_retry",
            },
            {
                "prompt": "Run tests (one fails, identify and report)",
                "initial_attempt": {"tool": "bash", "args": {"command": "pytest"}, "result": "1 failed"},
                "recovery": {"tool": "bash", "args": {"command": "pytest -v --tb=short"}},
                "outcome": "failure_diagnosed",
            },
            {
                "prompt": "Push to remote (rejected, pull first then push)",
                "initial_attempt": {"tool": "bash", "args": {"command": "git push"}, "result": "rejected"},
                "recovery": {"tool": "bash", "args": {"command": "git pull --rebase && git push"}},
                "outcome": "pushed_after_rebase",
            },
        ]

        for i in range(n):
            scenario = self.rng.choice(error_scenarios)
            yield OrchestrationScenario(
                id=f"error_recovery_{i}",
                category="error_recovery",
                prompt=scenario["prompt"],
                tool_chain=[scenario["initial_attempt"], scenario["recovery"]],
                expected_outcome=scenario["outcome"],
                context={"error_type": scenario["initial_attempt"]["result"]},
                difficulty="hard",
            )

    # -------------------------------------------------------------------------
    # Long-Context Coherence
    # -------------------------------------------------------------------------

    def long_context(self, n: int = 20) -> Iterator[OrchestrationScenario]:
        """Generate long-context coherence scenarios."""
        self.reset()

        # Facts to remember across conversation
        facts = [
            {"key": "project_name", "value": "phoenix", "turn": 1},
            {"key": "database", "value": "postgres", "turn": 3},
            {"key": "port", "value": "8080", "turn": 5},
            {"key": "secret_code", "value": "ALPHA-7", "turn": 2},
            {"key": "deploy_target", "value": "aws-east", "turn": 4},
        ]

        for i in range(n):
            fact = self.rng.choice(facts)
            intervening_turns = self.rng.randint(10, 30)

            yield OrchestrationScenario(
                id=f"long_context_{i}",
                category="long_context",
                prompt=f"What was the {fact['key'].replace('_', ' ')} I mentioned earlier?",
                tool_chain=[],  # No tools, pure recall
                expected_outcome=fact["value"],
                context={
                    "fact_introduced_at_turn": fact["turn"],
                    "current_turn": fact["turn"] + intervening_turns,
                    "intervening_topics": ["unrelated discussion"] * intervening_turns,
                },
                difficulty="hard",
            )

    # -------------------------------------------------------------------------
    # Code Generation + Verification
    # -------------------------------------------------------------------------

    def code_generation(self, n: int = 30) -> Iterator[OrchestrationScenario]:
        """Generate code generation + verification scenarios."""
        self.reset()

        code_tasks = [
            {
                "prompt": "Write a function to check if a number is prime, then test it",
                "chain": [
                    {"tool": "write_file", "args": {"path": "prime.py", "content": "<implementation>"}},
                    {"tool": "bash", "args": {"command": "python -c 'from prime import is_prime; assert is_prime(7); assert not is_prime(4)'"}},
                ],
                "verification": "tests_pass",
            },
            {
                "prompt": "Create a REST endpoint and verify it returns 200",
                "chain": [
                    {"tool": "write_file", "args": {"path": "app.py", "content": "<flask_app>"}},
                    {"tool": "bash", "args": {"command": "curl -s -o /dev/null -w '%{http_code}' localhost:5000/health"}},
                ],
                "verification": "status_200",
            },
            {
                "prompt": "Write a SQL query to count users, then execute it",
                "chain": [
                    {"tool": "write_file", "args": {"path": "query.sql", "content": "SELECT COUNT(*) FROM users"}},
                    {"tool": "bash", "args": {"command": "psql -f query.sql"}},
                ],
                "verification": "query_executes",
            },
        ]

        for i in range(n):
            task = self.rng.choice(code_tasks)
            yield OrchestrationScenario(
                id=f"code_gen_{i}",
                category="code_generation",
                prompt=task["prompt"],
                tool_chain=task["chain"],
                expected_outcome=task["verification"],
                difficulty="hard",
            )

    def generate_all(self) -> list[OrchestrationScenario]:
        """Generate all orchestration scenarios."""
        scenarios = []
        scenarios.extend(self.tool_chains(40))
        scenarios.extend(self.error_recovery(30))
        scenarios.extend(self.long_context(20))
        scenarios.extend(self.code_generation(30))
        return scenarios
