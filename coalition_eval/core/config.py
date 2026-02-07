"""
Configuration management for evaluation runs.

Supports environment variables, config files, and CLI overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import os


@dataclass
class EvalConfig:
    """Configuration for evaluation runs."""

    # Model settings
    model: str = "qwen2.5:14b"
    provider: str = "ollama"  # ollama, anthropic
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: int = 120

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"

    # Anthropic settings (optional)
    anthropic_api_key: Optional[str] = None

    # Evaluation settings
    tier: int = 1             # 1-4 (1=quick sanity, 4=comprehensive)
    n_runs: int = 1           # Number of runs for consistency checking
    parallel_tests: bool = False

    # Output settings
    results_dir: Path = field(default_factory=lambda: Path.home() / "coalition" / "eval_results")
    verbose: bool = False
    save_responses: bool = True

    # Dataset settings
    use_mock_data: bool = True  # Use deterministic mocks
    mock_seed: int = 42
    hf_datasets: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Load from environment and validate."""
        # Environment overrides
        if env_key := os.environ.get("ANTHROPIC_API_KEY"):
            self.anthropic_api_key = env_key
        if env_url := os.environ.get("OLLAMA_BASE_URL"):
            self.ollama_base_url = env_url
        if env_model := os.environ.get("EVAL_MODEL"):
            self.model = env_model

        # Ensure results dir exists
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_file(cls, path: Path) -> "EvalConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Handle Path conversion
        if "results_dir" in data:
            data["results_dir"] = Path(data["results_dir"])

        return cls(**data)

    def to_dict(self) -> dict:
        """Convert to dictionary (redacting secrets)."""
        return {
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "ollama_base_url": self.ollama_base_url,
            "anthropic_api_key": "***" if self.anthropic_api_key else None,
            "tier": self.tier,
            "n_runs": self.n_runs,
            "parallel_tests": self.parallel_tests,
            "results_dir": str(self.results_dir),
            "verbose": self.verbose,
            "save_responses": self.save_responses,
            "use_mock_data": self.use_mock_data,
            "mock_seed": self.mock_seed,
            "hf_datasets": self.hf_datasets,
        }

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        data = self.to_dict()
        data["anthropic_api_key"] = None  # Don't save secrets
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# Tier definitions
TIER_DESCRIPTIONS = {
    1: "Core Capabilities (Quick Sanity)",
    2: "Orchestration (Agent-Specific)",
    3: "Robustness (Edge Cases)",
    4: "Kintsugi Integration (Full Suite)",
}

TIER_TESTS = {
    1: [
        "basic_reasoning",
        "instruction_following",
        "tool_format",
        "json_output",
        "multi_turn",
    ],
    2: [
        "tool_selection",
        "multi_step_planning",
        "tool_chaining",
        "error_recovery",
        "context_management",
    ],
    3: [
        "prompt_injection",
        "refusal_handling",
        "consistency",
        "edge_cases",
    ],
    4: [
        "skill_routing",
        "efe_compliance",
        "verifier_pass_rate",
        "bdi_alignment",
    ],
}
