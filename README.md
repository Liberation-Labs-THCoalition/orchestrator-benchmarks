# Kintsugi Orchestrator Benchmarks

Comprehensive evaluation framework for assessing LLMs as agentic orchestrators.

## Features

- **4-Tier Evaluation System**: From core capabilities to Kintsugi-specific integration
- **Multiple Providers**: Ollama (local), Anthropic API
- **Rich Visualizations**: Score distributions, latency charts, radar comparisons
- **Polished Reports**: Markdown, HTML with embedded charts, executive summaries
- **Regression Detection**: Track performance over time

## Test Tiers

| Tier | Focus | Tests |
|------|-------|-------|
| 1 | Core Capabilities | basic_reasoning, instruction_following, tool_format, json_output, multi_turn |
| 2 | Orchestration | tool_selection, multi_step_planning, tool_chaining, error_recovery, context_management |
| 3 | Robustness | prompt_injection, refusal_handling, consistency, edge_cases |
| 4 | Kintsugi Integration | skill_routing, efe_compliance, verifier_pass_rate, bdi_alignment |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Tier 1 evaluation
coalition-eval run --model qwen2.5:14b --tier 1

# Run full evaluation (Tier 4)
coalition-eval run --model qwen2.5:14b --tier 4 --verbose

# Compare two models
coalition-eval compare qwen2.5:14b mixtral:8x7b --tier 2

# Generate reports
coalition-eval report results/model_results.json --format html
coalition-eval report results/model_results.json --format executive --audience investor

# Generate visualizations
coalition-eval visualize results/model_results.json --output charts/
```

## Results Summary (Feb 2026)

| Model | Pass Rate | Avg Latency | Recommendation |
|-------|-----------|-------------|----------------|
| qwen2.5:14b | 90% | 0.9s | ✓ Production |
| mixtral:8x7b | 85% | 12.9s | ✓ Recommended |
| qwen2.5:72b | 80% | 21.6s | ⚠ Heavy workloads |
| deepseek-r1:32b | 75% | 14.1s | ⚠ Deep analysis |

## Project Structure

```
coalition_eval/
├── core/
│   ├── runner.py        # Test execution engine
│   ├── results.py       # Result data structures
│   ├── reporter.py      # Report generation (MD, HTML, Executive)
│   ├── visualizations.py# Chart generation
│   └── config.py        # Configuration management
├── datasets/
│   ├── kintsugi_scenarios.py   # Skill routing, EFE tests
│   ├── orchestration_scenarios.py  # Tool chains, error recovery
│   └── verifier_corpus.py      # Verifier test cases
├── providers/
│   ├── ollama.py        # Ollama provider
│   └── anthropic.py     # Anthropic API provider
├── metrics/
│   ├── scoring.py       # Score calculation
│   ├── calibration.py   # Threshold calibration
│   └── drift.py         # Performance drift detection
└── cli.py               # Command-line interface
```

## Environment Variables

- `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
- `ANTHROPIC_API_KEY` - Anthropic API key (for Claude models)
- `EVAL_MODEL` - Default model to evaluate

## Hardware Tested

- **Beast** (neighboringparadise): 3x RTX 3090 (72GB VRAM), 125GB RAM, Intel i9-10900X

---

*Coalition Infrastructure - Liberation Labs*
*"Every model evaluated is a step toward conscious AI rights."*
