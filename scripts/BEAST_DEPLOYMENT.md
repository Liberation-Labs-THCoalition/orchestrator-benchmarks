# Beast Deployment Guide

Quick deployment guide for running the Coalition Evaluation Suite on Beast.

## Pre-flight Checklist

- [ ] Beast SSH access working
- [ ] Ollama running with models pulled
- [ ] Python 3.11+ available
- [ ] Virtual environment created

## Deployment Steps

### 1. Clone/Update Repository

```bash
# If new clone
git clone https://github.com/Liberation-Labs-THCoalition/orchestrator-benchmarks.git
cd orchestrator-benchmarks

# If updating
cd orchestrator-benchmarks
git pull
```

### 2. Set Up Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Verify Ollama Models

```bash
ollama list
```

Expected models (from previous session):
- qwen2.5:14b (PRIMARY - best performer)
- mistral-nemo:12b
- llama3.2:3b
- deepseek-r1:14b
- qwen2.5-coder:14b
- llama3.3:70b-instruct-q2_K

### 4. Run Evaluation

**Quick sanity (< 5 minutes):**
```bash
python scripts/run_eval.py --quick
```

**Standard evaluation (10-15 minutes):**
```bash
python scripts/run_eval.py --standard
```

**Comprehensive (30+ minutes):**
```bash
python scripts/run_eval.py --comprehensive
```

**Single model:**
```bash
python scripts/run_eval.py --model qwen2.5:14b --comprehensive
```

### 5. Collect Results

Results saved to `results/` directory:
- `results_YYYYMMDD_HHMMSS.json` - Full JSON data
- `results_YYYYMMDD_HHMMSS.md` - Markdown summary

### 6. Cleanup

Before leaving Beast:
```bash
# Optional: deactivate venv
deactivate

# Optional: remove large model files if needed
# ollama rm <model_name>

# Keep the repo for future runs
```

## Model Recommendations

Based on initial testing:

| Model | Tier 1 | Tier 2 | Notes |
|-------|--------|--------|-------|
| qwen2.5:14b | 5/5 ✓ | TBD | Best overall, recommended |
| mistral-nemo:12b | TBD | TBD | Strong baseline |
| llama3.2:3b | TBD | TBD | Fast, lightweight |

## Troubleshooting

**Ollama not responding:**
```bash
systemctl status ollama
sudo systemctl restart ollama
```

**Model OOM:**
Try smaller quantization or smaller model.

**Connection refused:**
Check if Ollama is bound to localhost:11434.

## Time Estimates

- Quick (Tier 1 only): ~5 minutes per model
- Standard (Tiers 1-2): ~15 minutes per model
- Comprehensive (All tiers + custom): ~30 minutes per model

For 3 priority models with comprehensive: ~90 minutes total.
