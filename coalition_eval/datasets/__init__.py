"""
Dataset loading and mock generation.

Supports HuggingFace datasets, deterministic mock generators,
and custom Coalition/Kintsugi test scenarios.
"""

from coalition_eval.datasets.loader import (
    load_dataset_samples,
    list_available_datasets,
    DatasetConfig,
    Sample,
)
from coalition_eval.datasets.mock_data import (
    MockDataGenerator,
    generate_arithmetic_problems,
    generate_tool_selection_problems,
    generate_instruction_following_problems,
)
from coalition_eval.datasets.kintsugi_scenarios import (
    KintsugiScenario,
    KintsugiScenarioGenerator,
    generate_skill_routing_scenarios,
    generate_efe_scenarios,
    generate_harness_scenarios,
    generate_memory_scenarios,
)
from coalition_eval.datasets.orchestration_scenarios import (
    OrchestrationScenario,
    OrchestrationScenarioGenerator,
)
from coalition_eval.datasets.verifier_corpus import (
    VerifierTestCase,
    VerifierCorpusGenerator,
)

__all__ = [
    # HuggingFace loader
    "load_dataset_samples",
    "list_available_datasets",
    "DatasetConfig",
    "Sample",
    # Basic mock generators
    "MockDataGenerator",
    "generate_arithmetic_problems",
    "generate_tool_selection_problems",
    "generate_instruction_following_problems",
    # Kintsugi-specific scenarios
    "KintsugiScenario",
    "KintsugiScenarioGenerator",
    "generate_skill_routing_scenarios",
    "generate_efe_scenarios",
    "generate_harness_scenarios",
    "generate_memory_scenarios",
    # Orchestration scenarios
    "OrchestrationScenario",
    "OrchestrationScenarioGenerator",
    # Verifier corpus
    "VerifierTestCase",
    "VerifierCorpusGenerator",
]
