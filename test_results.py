"""
Kintsugi Local Model Test Results Tracker
Generates professional reports for client demonstrations.
"""
import json
import os
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path.home() / "coalition" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def get_system_info():
    """Capture hardware specs for reproducibility."""
    import subprocess
    
    info = {
        "timestamp": datetime.now().isoformat(),
        "hostname": os.uname().nodename,
    }
    
    # GPU info
    try:
        gpu_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True
        )
        info["gpus"] = [line.strip() for line in gpu_output.strip().split("\n")]
    except:
        info["gpus"] = ["Unable to query"]
    
    # Memory
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["ram_gb"] = int(line.split()[1]) // (1024 * 1024)
                    break
    except:
        info["ram_gb"] = "Unknown"
    
    # CPU
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":")[1].strip()
                    break
    except:
        info["cpu"] = "Unknown"
    
    return info

def record_test_run(model: str, results: list, system_info: dict = None):
    """Record a complete test run with all metadata."""
    if system_info is None:
        system_info = get_system_info()
    
    run_data = {
        "model": model,
        "system": system_info,
        "results": results,
        "summary": {
            "passed": sum(1 for r in results if r.get("passed")),
            "total": len(results),
            "avg_latency_ms": sum(r.get("latency_ms", 0) for r in results) / len(results) if results else 0,
        }
    }
    
    # Save individual run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model.replace(":", "_").replace("/", "_")
    filename = RESULTS_DIR / f"{model_safe}_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(run_data, f, indent=2)
    
    print(f"Results saved to {filename}")
    return run_data

def generate_report():
    """Generate markdown report from all test runs."""
    runs = []
    for f in RESULTS_DIR.glob("*.json"):
        with open(f) as fp:
            runs.append(json.load(fp))
    
    if not runs:
        return "No test results found."
    
    # Sort by model name
    runs.sort(key=lambda x: x["model"])
    
    # Get system info from first run
    system = runs[0].get("system", {})
    
    report = f"""# Kintsugi Local Model Orchestrator Test Results

## Test Environment
- **Host**: {system.get('hostname', 'Unknown')}
- **CPU**: {system.get('cpu', 'Unknown')}
- **RAM**: {system.get('ram_gb', 'Unknown')} GB
- **GPUs**: 
"""
    for gpu in system.get('gpus', []):
        report += f"  - {gpu}\n"
    
    report += f"""
- **Test Date**: {runs[0].get(system, {}).get(timestamp, Unknown)[:10]}

## Summary

| Model | Score | Avg Latency (ms) |
|-------|-------|------------------|
"""
    
    for run in runs:
        summary = run.get("summary", {})
        score = f"{summary.get(passed, 0)}/{summary.get(total, 0)}"
        latency = f"{summary.get(avg_latency_ms, 0):.0f}"
        report += f"| {run[model]} | {score} | {latency} |\n"
    
    report += """
## Detailed Results

"""
    
    for run in runs:
        report += f"### {run[model]}\n\n"
        for result in run.get("results", []):
            status = "✓" if result.get("passed") else "✗"
            report += f"- {status} **{result.get(test, Unknown)}**: {result.get(latency_ms, 0)}ms\n"
            if result.get("notes"):
                report += f"  - {result[notes]}\n"
        report += "\n"
    
    # Save report
    report_file = RESULTS_DIR / "REPORT.md"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"Report saved to {report_file}")
    return report

if __name__ == "__main__":
    print(json.dumps(get_system_info(), indent=2))
