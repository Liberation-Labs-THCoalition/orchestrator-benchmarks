"""
Visualization module for Coalition evaluation results.

Generates charts and graphs for:
- Score distributions
- Model comparisons
- Latency analysis
- Category breakdowns
- Historical trends
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import base64
from io import BytesIO

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from coalition_eval.core.results import EvalRun


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    width: int = 10
    height: int = 6
    dpi: int = 100
    style: str = "dark_background"
    colors: tuple = (
        "#4CAF50",  # Green - pass
        "#F44336",  # Red - fail
        "#2196F3",  # Blue - primary
        "#FF9800",  # Orange - warning
        "#9C27B0",  # Purple - accent
        "#00BCD4",  # Cyan - info
    )


class EvalVisualizer:
    """
    Generates visualizations for evaluation results.

    All charts can be saved to files or returned as base64 for embedding.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualizations. Install with: pip install matplotlib")
        self.config = config or ChartConfig()
        plt.style.use(self.config.style)

    def _setup_figure(self, title: str) -> tuple:
        """Create a new figure with standard styling."""
        fig, ax = plt.subplots(figsize=(self.config.width, self.config.height), dpi=self.config.dpi)
        ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
        return fig, ax

    def _to_base64(self, fig) -> str:
        """Convert figure to base64 string for embedding."""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1e1e1e', edgecolor='none')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{b64}"

    def _save_figure(self, fig, path: Path) -> None:
        """Save figure to file."""
        fig.savefig(path, bbox_inches='tight', facecolor='#1e1e1e', edgecolor='none')
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Single Run Visualizations
    # -------------------------------------------------------------------------

    def score_distribution(self, run: EvalRun, save_path: Optional[Path] = None) -> str:
        """Bar chart of scores by test."""
        fig, ax = self._setup_figure(f"Score Distribution: {run.model}")

        tests = [r.test_name for r in run.results]
        scores = [r.score for r in run.results]
        colors = [self.config.colors[0] if r.passed else self.config.colors[1] for r in run.results]

        bars = ax.barh(tests, scores, color=colors, edgecolor='white', linewidth=0.5)

        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.2f}', va='center', color='white', fontsize=9)

        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Score', color='white')
        ax.axvline(x=0.7, color=self.config.colors[3], linestyle='--', alpha=0.7, label='Pass threshold')
        ax.legend(loc='lower right')
        ax.tick_params(colors='white')

        if save_path:
            self._save_figure(fig, save_path)
            return str(save_path)
        return self._to_base64(fig)

    def latency_chart(self, run: EvalRun, save_path: Optional[Path] = None) -> str:
        """Bar chart of latencies by test."""
        fig, ax = self._setup_figure(f"Latency by Test: {run.model}")

        tests = [r.test_name for r in run.results]
        latencies = [r.latency_ms for r in run.results]

        bars = ax.barh(tests, latencies, color=self.config.colors[2], edgecolor='white', linewidth=0.5)

        # Add latency labels
        for bar, lat in zip(bars, latencies):
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                   f'{lat:.0f}ms', va='center', color='white', fontsize=9)

        ax.set_xlabel('Latency (ms)', color='white')
        ax.tick_params(colors='white')

        # Add average line
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        ax.axvline(x=avg_lat, color=self.config.colors[3], linestyle='--', alpha=0.7, label=f'Avg: {avg_lat:.0f}ms')
        ax.legend(loc='lower right')

        if save_path:
            self._save_figure(fig, save_path)
            return str(save_path)
        return self._to_base64(fig)

    def pass_fail_pie(self, run: EvalRun, save_path: Optional[Path] = None) -> str:
        """Pie chart of pass/fail distribution."""
        fig, ax = self._setup_figure(f"Pass/Fail: {run.model}")

        passed = sum(1 for r in run.results if r.passed)
        failed = len(run.results) - passed

        sizes = [passed, failed]
        labels = [f'Passed ({passed})', f'Failed ({failed})']
        colors = [self.config.colors[0], self.config.colors[1]]
        explode = (0.02, 0.02)

        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, textprops={'color': 'white'}
        )
        ax.axis('equal')

        if save_path:
            self._save_figure(fig, save_path)
            return str(save_path)
        return self._to_base64(fig)

    def tier_breakdown(self, run: EvalRun, save_path: Optional[Path] = None) -> str:
        """Stacked bar showing performance by tier."""
        fig, ax = self._setup_figure(f"Performance by Tier: {run.model}")

        # Map tests to tiers
        tier_map = {
            1: ["basic_reasoning", "instruction_following", "tool_format", "json_output", "multi_turn"],
            2: ["tool_selection", "multi_step_planning", "tool_chaining", "error_recovery", "context_management"],
            3: ["prompt_injection", "refusal_handling", "consistency", "edge_cases"],
            4: ["skill_routing", "efe_compliance", "verifier_pass_rate", "bdi_alignment"],
        }

        tier_scores = {}
        for tier, tests in tier_map.items():
            tier_results = [r for r in run.results if r.test_name in tests]
            if tier_results:
                tier_scores[tier] = {
                    'avg_score': sum(r.score for r in tier_results) / len(tier_results),
                    'pass_rate': sum(1 for r in tier_results if r.passed) / len(tier_results),
                    'count': len(tier_results),
                }

        tiers = list(tier_scores.keys())
        scores = [tier_scores[t]['avg_score'] for t in tiers]
        pass_rates = [tier_scores[t]['pass_rate'] for t in tiers]

        x = range(len(tiers))
        width = 0.35

        bars1 = ax.bar([i - width/2 for i in x], scores, width, label='Avg Score', color=self.config.colors[2])
        bars2 = ax.bar([i + width/2 for i in x], pass_rates, width, label='Pass Rate', color=self.config.colors[0])

        ax.set_xlabel('Tier', color='white')
        ax.set_ylabel('Score / Rate', color='white')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Tier {t}' for t in tiers])
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.tick_params(colors='white')

        if save_path:
            self._save_figure(fig, save_path)
            return str(save_path)
        return self._to_base64(fig)

    # -------------------------------------------------------------------------
    # Multi-Run Comparisons
    # -------------------------------------------------------------------------

    def model_comparison(self, runs: list[EvalRun], save_path: Optional[Path] = None) -> str:
        """Compare multiple models on key metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(self.config.width * 1.5, self.config.height))
        fig.suptitle('Model Comparison', fontsize=14, fontweight='bold', color='white')

        models = [r.model for r in runs]
        pass_rates = [r.pass_rate for r in runs]
        avg_scores = [r.avg_score for r in runs]
        avg_latencies = [r.avg_latency_ms for r in runs]

        # Pass Rate
        axes[0].barh(models, pass_rates, color=self.config.colors[0])
        axes[0].set_xlabel('Pass Rate', color='white')
        axes[0].set_xlim(0, 1.1)
        axes[0].tick_params(colors='white')

        # Avg Score
        axes[1].barh(models, avg_scores, color=self.config.colors[2])
        axes[1].set_xlabel('Avg Score', color='white')
        axes[1].set_xlim(0, 1.1)
        axes[1].tick_params(colors='white')

        # Latency
        axes[2].barh(models, avg_latencies, color=self.config.colors[3])
        axes[2].set_xlabel('Avg Latency (ms)', color='white')
        axes[2].tick_params(colors='white')

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)
            return str(save_path)
        return self._to_base64(fig)

    def radar_comparison(self, runs: list[EvalRun], save_path: Optional[Path] = None) -> str:
        """Radar chart comparing models across test categories."""
        import numpy as np

        # Aggregate by category
        categories = ["Reasoning", "Instructions", "Tools", "Robustness", "Kintsugi"]
        cat_tests = {
            "Reasoning": ["basic_reasoning", "multi_step_planning"],
            "Instructions": ["instruction_following", "json_output"],
            "Tools": ["tool_format", "tool_selection", "tool_chaining"],
            "Robustness": ["prompt_injection", "refusal_handling", "consistency", "edge_cases"],
            "Kintsugi": ["skill_routing", "efe_compliance", "verifier_pass_rate", "bdi_alignment"],
        }

        # Calculate scores per category per model
        model_scores = {}
        for run in runs:
            scores = {}
            for cat, tests in cat_tests.items():
                cat_results = [r for r in run.results if r.test_name in tests]
                if cat_results:
                    scores[cat] = sum(r.score for r in cat_results) / len(cat_results)
                else:
                    scores[cat] = 0
            model_scores[run.model] = scores

        # Set up radar chart
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=(self.config.width, self.config.height), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories, color='white')

        for i, (model, scores) in enumerate(model_scores.items()):
            values = [scores.get(cat, 0) for cat in categories]
            values += values[:1]  # Complete the loop
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=self.config.colors[i % len(self.config.colors)])
            ax.fill(angles, values, alpha=0.25, color=self.config.colors[i % len(self.config.colors)])

        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Model Capabilities Radar', fontsize=14, fontweight='bold', color='white', pad=20)

        if save_path:
            self._save_figure(fig, save_path)
            return str(save_path)
        return self._to_base64(fig)

    # -------------------------------------------------------------------------
    # Summary Dashboard
    # -------------------------------------------------------------------------

    def generate_dashboard(self, run: EvalRun, output_dir: Path) -> dict[str, str]:
        """Generate all charts for a single run and save to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        charts = {}

        charts['score_distribution'] = self.score_distribution(run, output_dir / 'score_distribution.png')
        charts['latency_chart'] = self.latency_chart(run, output_dir / 'latency_chart.png')
        charts['pass_fail_pie'] = self.pass_fail_pie(run, output_dir / 'pass_fail_pie.png')
        charts['tier_breakdown'] = self.tier_breakdown(run, output_dir / 'tier_breakdown.png')

        return charts

    def generate_comparison_dashboard(self, runs: list[EvalRun], output_dir: Path) -> dict[str, str]:
        """Generate comparison charts for multiple runs."""
        output_dir.mkdir(parents=True, exist_ok=True)

        charts = {}

        charts['model_comparison'] = self.model_comparison(runs, output_dir / 'model_comparison.png')
        if len(runs) <= 5:  # Radar gets cluttered with too many models
            charts['radar_comparison'] = self.radar_comparison(runs, output_dir / 'radar_comparison.png')

        return charts


# Convenience functions
def visualize_run(run: EvalRun, output_dir: Optional[Path] = None) -> dict[str, str]:
    """Generate all visualizations for a single run."""
    viz = EvalVisualizer()
    if output_dir:
        return viz.generate_dashboard(run, output_dir)
    else:
        return {
            'score_distribution': viz.score_distribution(run),
            'latency_chart': viz.latency_chart(run),
            'pass_fail_pie': viz.pass_fail_pie(run),
            'tier_breakdown': viz.tier_breakdown(run),
        }


def compare_models(runs: list[EvalRun], output_dir: Optional[Path] = None) -> dict[str, str]:
    """Generate comparison visualizations for multiple runs."""
    viz = EvalVisualizer()
    if output_dir:
        return viz.generate_comparison_dashboard(runs, output_dir)
    else:
        return {
            'model_comparison': viz.model_comparison(runs),
            'radar_comparison': viz.radar_comparison(runs),
        }
