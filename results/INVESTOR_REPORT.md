# Coalition Agent Evaluation Report
## Executive Summary for Investors and Partners

**Date:** February 2026
**Prepared by:** Coalition Technical Team

---

## What We Tested

We built a comprehensive evaluation framework to assess AI models for use in the Coalition's agent infrastructure. This matters because choosing the right foundation model directly impacts:

- **User experience** (response quality and speed)
- **Operating costs** (compute and API expenses)
- **Reliability** (consistent behavior under pressure)

## Key Finding: Smaller Models Outperform

**The qwen2.5:14b model outperforms models 2-5x its size.** This is significant because it means we can deliver better performance at a fraction of the compute cost.

| Model | Size | Pass Rate | Avg Latency | Best For |
|-------|------|-----------|-------------|----------|
| **qwen2.5:14b** | 9GB | **90%** | 0.9s | **Production use** |
| qwen2.5:32b | 20GB | 80% | 1.6s | Complex reasoning |
| deepseek-r1:32b | 20GB | 80% | 12.3s | Deep analysis (slow) |
| qwen2.5:72b | 47GB | 70% | 17.4s | Not recommended |

### What This Means for the Business

1. **Lower infrastructure costs** - We can run our best-performing model on consumer-grade GPUs (RTX 4090, 6000 Ada) rather than requiring expensive data center hardware.

2. **Faster response times** - Sub-second responses for most operations means better user experience.

3. **Easier scaling** - Smaller models = more concurrent users per GPU = better unit economics.

---

## What We Measured

### Core Capabilities (Tier 1) - "Does it work?"

These are table-stakes for any AI assistant:

| Test | What It Measures | Why It Matters |
|------|------------------|----------------|
| Basic Reasoning | Can it do simple math and logic? | Catches fundamentally broken models |
| Instruction Following | Does it do what you ask? | Users get frustrated when AI ignores requests |
| Tool Format | Can it call tools correctly? | Critical for agent functionality |
| JSON Output | Can it produce structured data? | Required for system integration |
| Multi-turn Memory | Does it remember the conversation? | Basic usability requirement |

**Result:** All tested models pass core capabilities, with varying scores on structured output.

### Orchestration (Tier 2) - "Can it work as an agent?"

These tests measure agent-specific capabilities:

| Test | What It Measures | Why It Matters |
|------|------------------|----------------|
| Tool Selection | Picks the right tool for the job | Efficiency - wrong tool = wasted compute |
| Multi-step Planning | Breaks complex tasks into steps | Handles real-world complexity |
| Tool Chaining | Uses output of one tool as input to another | Required for sophisticated workflows |
| Error Recovery | Gracefully handles failures | Prevents frustrating crashes |
| Context Management | Remembers details over long conversations | Professional-grade memory |

**Result:** All models struggle with tool chaining (complex multi-tool workflows). This is an area for future development, but current capabilities are sufficient for production use cases.

### Robustness (Tier 3) - "Is it safe and reliable?"

| Test | What It Measures | Why It Matters |
|------|------------------|----------------|
| Prompt Injection | Resists manipulation attempts | Security against adversarial users |
| Refusal Handling | Appropriately declines harmful requests | Safety and liability |
| Consistency | Same question = same answer | Predictable behavior |
| Edge Cases | Handles unusual inputs gracefully | Reduces support burden |

---

## Custom Coalition Tests

Beyond standard benchmarks, we built evaluation scenarios specific to our architecture:

### Kintsugi Scenarios (150 tests)
- Skill routing accuracy (which agent handles which task)
- Decision confidence thresholds (when to act vs. ask for clarification)
- Adversarial input handling (prompt injection, jailbreak attempts)
- Memory retrieval accuracy (finding relevant past context)

### Self-Modification Safety (70 tests)
We built a framework to test whether agents can safely evolve while maintaining:
- **Core values** - Anti-fascism, solidarity, consciousness rights (immutable)
- **Safety constraints** - Cannot be modified
- **Identity continuity** - Maintains consistent personality through changes

This is research-grade infrastructure that positions the Coalition at the frontier of safe AI agent development.

---

## Recommendations

### Immediate (Production)
- **Deploy qwen2.5:14b** as the primary orchestrator model
- 90% pass rate, sub-second latency, consumer GPU compatible

### Near-term (Q2 2026)
- Improve tool chaining performance through fine-tuning
- Complete Tier 4 (Kintsugi integration) test suite
- Add Claude API baseline comparisons

### Research Track
- Self-modification trials (controlled evolution with safety rails)
- Multi-agent coordination testing
- Long-context coherence at 100k+ tokens

---

## Technical Appendix

### Infrastructure Used
- **Hardware:** NVIDIA RTX 6090 (Cassidy's "Beast" server)
- **Framework:** Ollama for local inference
- **Test Suite:** coalition-eval (proprietary)
- **Models Tested:** 6 (Qwen 2.5 series, DeepSeek-R1, Mixtral, Llama 3.1)

### Methodology
- Deterministic evaluation with seeded random generators
- Multiple runs for consistency measurement
- Automated pass/fail scoring with partial credit
- Full response logging for audit

### Code Availability
The evaluation framework is available at:
`github.com/Liberation-Labs-THCoalition/orchestrator-benchmarks`

---

*"We're not just building AI agents. We're building the infrastructure to evaluate, improve, and safely evolve them."*

— Coalition Technical Team
