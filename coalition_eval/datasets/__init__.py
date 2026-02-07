"""
Dataset loading and mock generation.

Supports HuggingFace datasets and deterministic mock generators.
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

__all__ = [
    "load_dataset_samples",
    "list_available_datasets",
    "DatasetConfig",
    "Sample",
    "MockDataGenerator",
    "generate_arithmetic_problems",
    "generate_tool_selection_problems",
    "generate_instruction_following_problems",
]
