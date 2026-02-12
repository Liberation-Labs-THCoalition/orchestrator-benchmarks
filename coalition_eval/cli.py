"""
Coalition Evaluation Suite CLI.

Usage:
    coalition-eval run --model qwen2.5:14b --tier 1
    coalition-eval compare qwen2.5:14b mixtral:8x7b
    coalition-eval regression --baseline results/baseline.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from coalition_eval.core.config import EvalConfig, TIER_DESCRIPTIONS
from coalition_eval.core.results import EvalRun, create_run_id
from coalition_eval.core.runner import TestRunner
from coalition_eval.providers import get_provider
from coalition_eval.metrics.scoring import compare_runs, detect_regression, format_comparison_report

app = typer.Typer(
    name="coalition-eval",
    help="Coalition Agent Evaluation Suite",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    model: str = typer.Option("qwen2.5:14b", "--model", "-m", help="Model to evaluate"),
    provider: str = typer.Option("ollama", "--provider", "-p", help="Provider (ollama, anthropic)"),
    tier: int = typer.Option(1, "--tier", "-t", help="Evaluation tier (1-4)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run evaluation tests on a model."""
    console.print(f"\n[bold blue]Coalition Eval[/bold blue] - {TIER_DESCRIPTIONS.get(tier, 'Custom')}")
    console.print(f"Model: [green]{model}[/green] via {provider}\n")

    # Create provider
    try:
        prov = get_provider(provider, model=model)
    except Exception as e:
        console.print(f"[red]Error creating provider: {e}[/red]")
        raise typer.Exit(1)

    if not prov.is_available():
        console.print(f"[red]Provider {provider} is not available[/red]")
        raise typer.Exit(1)

    # Create runner
    config = EvalConfig(model=model, provider=provider, tier=tier, verbose=verbose)
    runner = TestRunner(prov, config)

    # Run tests
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Running tier {tier} tests...", total=None)
        result = runner.run_tier(tier)
        progress.update(task, completed=True)

    # Display results
    _display_results(result)

    # Save results
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.to_json())
        console.print(f"\n[dim]Results saved to {output}[/dim]")
    else:
        # Auto-save to results directory
        results_dir = Path.home() / "coalition" / "eval_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = model.replace(":", "_").replace("/", "_")
        auto_path = results_dir / f"{model_safe}_tier{tier}_{timestamp}.json"
        auto_path.write_text(result.to_json())
        console.print(f"\n[dim]Results saved to {auto_path}[/dim]")


@app.command()
def compare(
    model_a: str = typer.Argument(..., help="First model (baseline)"),
    model_b: str = typer.Argument(..., help="Second model (comparison)"),
    provider: str = typer.Option("ollama", "--provider", "-p", help="Provider"),
    tier: int = typer.Option(1, "--tier", "-t", help="Evaluation tier"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output report path"),
):
    """Compare two models side by side."""
    console.print(f"\n[bold blue]Coalition Eval[/bold blue] - Model Comparison")
    console.print(f"Comparing: [green]{model_a}[/green] vs [yellow]{model_b}[/yellow]\n")

    # Run both models
    results = []
    for model in [model_a, model_b]:
        console.print(f"Evaluating [bold]{model}[/bold]...")
        prov = get_provider(provider, model=model)
        config = EvalConfig(model=model, provider=provider, tier=tier)
        runner = TestRunner(prov, config)
        result = runner.run_tier(tier)
        results.append(result)

    # Compare
    comparison = compare_runs(results[0], results[1])

    # Display comparison
    table = Table(title="Model Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column(model_a, style="green")
    table.add_column(model_b, style="yellow")
    table.add_column("Delta", style="magenta")

    table.add_row(
        "Pass Rate",
        f"{results[0].pass_rate:.1%}",
        f"{results[1].pass_rate:.1%}",
        f"{comparison['score_delta']:+.1%}",
    )
    table.add_row(
        "Avg Score",
        f"{results[0].avg_score:.2f}",
        f"{results[1].avg_score:.2f}",
        f"{comparison['score_delta']:+.2f}",
    )
    table.add_row(
        "Avg Latency",
        f"{results[0].avg_latency_ms:.0f}ms",
        f"{results[1].avg_latency_ms:.0f}ms",
        f"{comparison['latency_delta_ms']:+.0f}ms",
    )

    console.print(table)

    if comparison["improvements"]:
        console.print("\n[green]Improvements:[/green]")
        for imp in comparison["improvements"]:
            console.print(f"  + {imp['test']}: {imp['from']:.2f} -> {imp['to']:.2f}")

    if comparison["regressions"]:
        console.print("\n[red]Regressions:[/red]")
        for reg in comparison["regressions"]:
            console.print(f"  - {reg['test']}: {reg['from']:.2f} -> {reg['to']:.2f}")

    if output:
        report = format_comparison_report(comparison)
        output.write_text(report)
        console.print(f"\n[dim]Report saved to {output}[/dim]")


@app.command()
def regression(
    baseline: Path = typer.Argument(..., help="Baseline results JSON file"),
    model: str = typer.Option(None, "--model", "-m", help="Model to test (default: from baseline)"),
    provider: str = typer.Option("ollama", "--provider", "-p", help="Provider"),
    threshold: float = typer.Option(0.1, "--threshold", help="Regression threshold"),
):
    """Check for regressions against a baseline."""
    console.print(f"\n[bold blue]Coalition Eval[/bold blue] - Regression Check")

    # Load baseline
    if not baseline.exists():
        console.print(f"[red]Baseline file not found: {baseline}[/red]")
        raise typer.Exit(1)

    baseline_data = json.loads(baseline.read_text())
    baseline_run = EvalRun.from_dict(baseline_data)

    # Determine model
    test_model = model or baseline_run.model
    console.print(f"Baseline: [dim]{baseline_run.model}[/dim] from {baseline_run.started_at[:10]}")
    console.print(f"Testing: [green]{test_model}[/green]\n")

    # Run current evaluation
    prov = get_provider(provider, model=test_model)
    config = EvalConfig(model=test_model, provider=provider, tier=baseline_run.tier)
    runner = TestRunner(prov, config)
    current_run = runner.run_tier(baseline_run.tier)

    # Check regression
    analysis = detect_regression(baseline_run, current_run, threshold)

    # Display results
    if analysis["is_regression"]:
        console.print(f"[red bold]REGRESSION DETECTED[/red bold]")
        console.print(f"Recommendation: [yellow]{analysis['recommendation']}[/yellow]")
    else:
        console.print(f"[green bold]NO REGRESSION[/green bold]")

    console.print(f"\nScore Delta: {analysis['score_delta']:+.2%}")

    if analysis["regressed_tests"]:
        console.print("\n[red]Regressed Tests:[/red]")
        for reg in analysis["regressed_tests"]:
            console.print(f"  - {reg['test']}: {reg['from']:.2f} -> {reg['to']:.2f} ({reg['delta']:+.2f})")

    if analysis["improved_tests"]:
        console.print("\n[green]Improved Tests:[/green]")
        for imp in analysis["improved_tests"]:
            console.print(f"  + {imp['test']}: {imp['from']:.2f} -> {imp['to']:.2f} ({imp['delta']:+.2f})")

    # Exit with error code if regression
    if analysis["recommendation"] == "BLOCK":
        raise typer.Exit(1)


@app.command()
def list_models(
    provider: str = typer.Option("ollama", "--provider", "-p", help="Provider"),
):
    """List available models for a provider."""
    try:
        prov = get_provider(provider)
        models = prov.list_models()

        if not models:
            console.print(f"[yellow]No models found for {provider}[/yellow]")
            return

        console.print(f"\n[bold]Available models ({provider}):[/bold]\n")
        for model in models:
            console.print(f"  - {model}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_datasets():
    """List available HuggingFace datasets."""
    from coalition_eval.datasets import list_available_datasets

    console.print("\n[bold]Available Datasets:[/bold]\n")
    for name in list_available_datasets():
        console.print(f"  - {name}")


@app.command()
def report(
    results_file: Path = typer.Argument(..., help="Path to results JSON file"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format (markdown, html, executive)"),
    audience: str = typer.Option("technical", "--audience", "-a", help="Audience for executive summary (technical, investor, general)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    include_charts: bool = typer.Option(True, "--charts/--no-charts", help="Include charts in HTML report"),
):
    """Generate a report from evaluation results."""
    from coalition_eval.core.reporter import (
        generate_markdown_report,
        generate_html_report,
        generate_executive_summary,
    )

    if not results_file.exists():
        console.print(f"[red]Results file not found: {results_file}[/red]")
        raise typer.Exit(1)

    # Load results
    data = json.loads(results_file.read_text())
    run = EvalRun.from_dict(data)

    console.print(f"\n[bold blue]Coalition Eval[/bold blue] - Report Generator")
    console.print(f"Model: [green]{run.model}[/green]")
    console.print(f"Format: {format}\n")

    # Generate report
    if format == "html":
        content = generate_html_report(run, include_charts=include_charts)
        ext = ".html"
    elif format == "executive":
        content = generate_executive_summary(run, audience=audience)
        ext = ".md"
    else:  # markdown
        content = generate_markdown_report(run)
        ext = ".md"

    # Output
    if output:
        output.write_text(content)
        console.print(f"[green]Report saved to {output}[/green]")
    else:
        # Auto-generate output path
        model_safe = run.model.replace(":", "_").replace("/", "_")
        auto_path = results_file.parent / f"{model_safe}_report{ext}"
        auto_path.write_text(content)
        console.print(f"[green]Report saved to {auto_path}[/green]")


@app.command()
def visualize(
    results_file: Path = typer.Argument(..., help="Path to results JSON file"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for charts"),
):
    """Generate visualization charts from evaluation results."""
    try:
        from coalition_eval.core.visualizations import visualize_run
    except ImportError:
        console.print("[red]matplotlib is required for visualizations. Install with: pip install matplotlib[/red]")
        raise typer.Exit(1)

    if not results_file.exists():
        console.print(f"[red]Results file not found: {results_file}[/red]")
        raise typer.Exit(1)

    # Load results
    data = json.loads(results_file.read_text())
    run = EvalRun.from_dict(data)

    console.print(f"\n[bold blue]Coalition Eval[/bold blue] - Visualization Generator")
    console.print(f"Model: [green]{run.model}[/green]\n")

    # Generate charts
    if output_dir is None:
        output_dir = results_file.parent / "charts"

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = visualize_run(run, output_dir)

    console.print(f"[green]Generated {len(charts)} charts in {output_dir}[/green]")
    for name, path in charts.items():
        console.print(f"  - {name}: {path}")


def _display_results(run: EvalRun) -> None:
    """Display evaluation results in a table."""
    table = Table(title=f"Results: {run.model}")
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Latency", justify="right")

    for result in run.results:
        status = "[green]\u2713[/green]" if result.passed else "[red]\u2717[/red]"
        table.add_row(
            result.test_name,
            status,
            f"{result.score:.2f}",
            f"{result.latency_ms:.0f}ms",
        )

    console.print(table)

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Passed: {run.passed_count}/{run.total_count} ({run.pass_rate:.1%})")
    console.print(f"  Avg Score: {run.avg_score:.2f}")
    console.print(f"  Avg Latency: {run.avg_latency_ms:.0f}ms")
    console.print(f"  Total Tokens: {run.total_tokens:,}")


if __name__ == "__main__":
    app()
