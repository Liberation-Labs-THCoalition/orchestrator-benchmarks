"""Core evaluation infrastructure."""

from coalition_eval.core.results import EvalResult, EvalRun, ErrorCategory
from coalition_eval.core.runner import TestRunner
from coalition_eval.core.config import EvalConfig
from coalition_eval.core.reporter import (
    generate_markdown_report,
    generate_html_report,
    generate_executive_summary,
    generate_comparison_report,
)

# Optional visualization support
try:
    from coalition_eval.core.visualizations import (
        EvalVisualizer,
        visualize_run,
        compare_models,
    )
    HAS_VISUALIZATIONS = True
except ImportError:
    HAS_VISUALIZATIONS = False

__all__ = [
    "EvalResult",
    "EvalRun",
    "ErrorCategory",
    "TestRunner",
    "EvalConfig",
    "generate_markdown_report",
    "generate_html_report",
    "generate_executive_summary",
    "generate_comparison_report",
]

if HAS_VISUALIZATIONS:
    __all__.extend(["EvalVisualizer", "visualize_run", "compare_models"])
