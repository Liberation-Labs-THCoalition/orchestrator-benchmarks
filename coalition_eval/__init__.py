"""
Coalition Evaluation Suite

Comprehensive evaluation framework for testing LLM agents across the Coalition.
Supports local Ollama models, Anthropic Claude, and custom providers.
"""

__version__ = "0.1.0"

from coalition_eval.core.results import EvalResult, EvalRun, ErrorCategory
from coalition_eval.core.runner import TestRunner

__all__ = [
    "__version__",
    "EvalResult",
    "EvalRun",
    "ErrorCategory",
    "TestRunner",
]
