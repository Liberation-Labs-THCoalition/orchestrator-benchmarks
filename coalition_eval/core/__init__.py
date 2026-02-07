"""Core evaluation infrastructure."""

from coalition_eval.core.results import EvalResult, EvalRun, ErrorCategory
from coalition_eval.core.runner import TestRunner
from coalition_eval.core.config import EvalConfig

__all__ = [
    "EvalResult",
    "EvalRun",
    "ErrorCategory",
    "TestRunner",
    "EvalConfig",
]
