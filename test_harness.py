#!/usr/bin/env python3
"""
Local Model Test Harness for Kintsugi
Tests orchestrator capabilities of local models via Ollama.
"""

import requests
import json
import time
from dataclasses import dataclass
from typing import Optional

OLLAMA_BASE = "http://localhost:11434"

@dataclass
class TestResult:
    test_name: str
    model: str
    passed: bool
    latency_ms: float
    response: str
    error: Optional[str] = None

def chat(model: str, messages: list[dict], temperature: float = 0.7) -> tuple[str, float]:
    """Send chat request to Ollama, return response and latency."""
    start = time.time()
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}},
        timeout=120
    )
    latency = (time.time() - start) * 1000
    data = resp.json()
    return data.get("message", {}).get("content", ""), latency

def test_basic_reasoning(model: str) -> TestResult:
    """Test basic reasoning capability."""
    messages = [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]
    try:
        response, latency = chat(model, messages, temperature=0)
        passed = "4" in response
        return TestResult("basic_reasoning", model, passed, latency, response)
    except Exception as e:
        return TestResult("basic_reasoning", model, False, 0, "", str(e))

def test_instruction_following(model: str) -> TestResult:
    """Test instruction following - key for orchestration."""
    messages = [{"role": "user", "content": "List exactly 3 colors. Format: 1. [color]\\n2. [color]\\n3. [color]"}]
    try:
        response, latency = chat(model, messages, temperature=0)
        lines = [l for l in response.strip().split('\n') if l.strip()]
        passed = len(lines) >= 3 and any(c in response.lower() for c in ['red', 'blue', 'green', 'yellow', 'purple', 'orange'])
        return TestResult("instruction_following", model, passed, latency, response)
    except Exception as e:
        return TestResult("instruction_following", model, False, 0, "", str(e))

def test_tool_format(model: str) -> TestResult:
    """Test ability to output structured tool calls."""
    messages = [{"role": "user", "content": """You have access to a tool called 'search'. To use it, output:
<tool_call>{"name": "search", "args": {"query": "your query"}}</tool_call>

Search for information about Python programming."""}]
    try:
        response, latency = chat(model, messages, temperature=0)
        passed = "tool_call" in response.lower() and "search" in response.lower()
        return TestResult("tool_format", model, passed, latency, response)
    except Exception as e:
        return TestResult("tool_format", model, False, 0, "", str(e))

def test_multi_turn(model: str) -> TestResult:
    """Test multi-turn conversation coherence."""
    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What is my name?"}
    ]
    try:
        response, latency = chat(model, messages, temperature=0)
        passed = "alice" in response.lower()
        return TestResult("multi_turn", model, passed, latency, response)
    except Exception as e:
        return TestResult("multi_turn", model, False, 0, "", str(e))

def test_json_output(model: str) -> TestResult:
    """Test JSON output capability."""
    messages = [{"role": "user", "content": 'Output a JSON object with keys "name" and "age". Example: {"name": "Bob", "age": 30}'}]
    try:
        response, latency = chat(model, messages, temperature=0)
        # Try to find and parse JSON in response
        import re
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            parsed = json.loads(json_match.group())
            passed = "name" in parsed and "age" in parsed
        else:
            passed = False
        return TestResult("json_output", model, passed, latency, response)
    except Exception as e:
        return TestResult("json_output", model, False, 0, "", str(e))

def run_test_battery(model: str) -> list[TestResult]:
    """Run all tests against a model."""
    tests = [
        test_basic_reasoning,
        test_instruction_following,
        test_tool_format,
        test_multi_turn,
        test_json_output,
    ]
    results = []
    for test_fn in tests:
        print(f"  Running {test_fn.__name__}...", end=" ", flush=True)
        result = test_fn(model)
        status = "✓" if result.passed else "✗"
        print(f"{status} ({result.latency_ms:.0f}ms)")
        results.append(result)
    return results

def main():
    # Get available models
    resp = requests.get(f"{OLLAMA_BASE}/api/tags")
    models = [m["name"] for m in resp.json().get("models", [])]
    
    if not models:
        print("No models available. Pull a model first: ollama pull qwen2.5:14b")
        return
    
    print(f"Available models: {models}\n")
    
    for model in models:
        print(f"Testing {model}:")
        results = run_test_battery(model)
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        print(f"  Score: {passed}/{total}\n")

if __name__ == "__main__":
    main()
