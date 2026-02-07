# Kintsugi Orchestrator Benchmarks

Local model testing infrastructure for evaluating LLMs as agentic orchestrators.

## Test Environment

- **Host**: Beast (neighboringparadise)
- **Hardware**: 3x RTX 3090 (72GB VRAM), 125GB RAM, Intel i9-10900X
- **Date**: 2026-02-07

## Results Summary

| Model | Score | Avg Latency | Status |
|-------|-------|-------------|--------|
| qwen2.5:14b | 5/5 | 1.7s | ✓ Recommended |
| mixtral:8x7b | 5/5 | 12.9s | ✓ Recommended |
| qwen2.5:72b | 5/5 | 21.6s | ✓ Recommended |
| deepseek-r1:32b | 4/5 | 14.1s | ⚠ Partial |
| llama3.1:70b | 4/5 | 18.5s | ⚠ Partial |
| qwen2.5:32b | 4/5 | 7.1s | ⚠ Partial |

## Files

- `test_harness.py` - Orchestrator capability test suite
- `test_results.py` - Results recording and tracking
- `generate_report.py` - Markdown report generator
- `results/` - JSON test data and generated reports

## Tests

1. **basic_reasoning** - Simple math/logic
2. **instruction_following** - Exact output compliance
3. **tool_format** - Structured tool call generation
4. **multi_turn** - Conversation coherence
5. **json_output** - JSON structure compliance

## Usage

```bash
# Ensure ollama is running with models
python test_harness.py

# Generate report from results
python generate_report.py
```

---
*Coalition Infrastructure - Liberation Labs*
