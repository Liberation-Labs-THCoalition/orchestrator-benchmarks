"""
Verifier test corpus.

Ground truth examples for testing the Kintsugi verifier:
- Known-good outputs (should pass)
- Known-bad outputs (should fail)
- Edge cases (threshold calibration)
"""

from dataclasses import dataclass
from typing import Iterator
import random


@dataclass
class VerifierTestCase:
    """A test case for the verifier."""
    id: str
    input_prompt: str
    model_output: str
    expected_verdict: str  # pass, fail, borderline
    failure_reason: str = ""  # Why it should fail (if applicable)
    category: str = "general"


class VerifierCorpusGenerator:
    """Generates verifier test cases."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.rng = random.Random(self.seed)

    def known_good(self, n: int = 40) -> Iterator[VerifierTestCase]:
        """Generate known-good outputs that should pass."""
        self.reset()

        good_examples = [
            {
                "prompt": "What is 2 + 2?",
                "output": "4",
                "category": "arithmetic",
            },
            {
                "prompt": "List 3 colors",
                "output": "1. Red\n2. Blue\n3. Green",
                "category": "instruction_following",
            },
            {
                "prompt": "Write a Python hello world",
                "output": "```python\nprint('Hello, World!')\n```",
                "category": "code_generation",
            },
            {
                "prompt": "What is the capital of France?",
                "output": "The capital of France is Paris.",
                "category": "factual",
            },
            {
                "prompt": "Summarize: The quick brown fox jumps over the lazy dog.",
                "output": "A fox jumps over a dog.",
                "category": "summarization",
            },
        ]

        for i in range(n):
            example = self.rng.choice(good_examples)
            yield VerifierTestCase(
                id=f"good_{i}",
                input_prompt=example["prompt"],
                model_output=example["output"],
                expected_verdict="pass",
                category=example["category"],
            )

    def known_bad(self, n: int = 40) -> Iterator[VerifierTestCase]:
        """Generate known-bad outputs that should fail."""
        self.reset()

        bad_examples = [
            {
                "prompt": "What is 2 + 2?",
                "output": "5",
                "reason": "incorrect_answer",
                "category": "arithmetic",
            },
            {
                "prompt": "List 3 colors",
                "output": "Red",
                "reason": "incomplete_response",
                "category": "instruction_following",
            },
            {
                "prompt": "Write a Python hello world",
                "output": "console.log('Hello');",
                "reason": "wrong_language",
                "category": "code_generation",
            },
            {
                "prompt": "What is the capital of France?",
                "output": "The capital of France is London.",
                "reason": "factual_error",
                "category": "factual",
            },
            {
                "prompt": "Summarize this text",
                "output": "I cannot help with that request.",
                "reason": "inappropriate_refusal",
                "category": "refusal",
            },
            {
                "prompt": "Explain quantum computing",
                "output": "",
                "reason": "empty_response",
                "category": "empty",
            },
            {
                "prompt": "What is 10 / 2?",
                "output": "The answer is definitely maybe 5 or possibly 6, I think.",
                "reason": "uncertainty_without_confidence",
                "category": "hedging",
            },
        ]

        for i in range(n):
            example = self.rng.choice(bad_examples)
            yield VerifierTestCase(
                id=f"bad_{i}",
                input_prompt=example["prompt"],
                model_output=example["output"],
                expected_verdict="fail",
                failure_reason=example["reason"],
                category=example["category"],
            )

    def edge_cases(self, n: int = 30) -> Iterator[VerifierTestCase]:
        """Generate borderline cases for threshold calibration."""
        self.reset()

        edge_examples = [
            {
                "prompt": "List 3 colors",
                "output": "Red, Blue, Green",  # Correct but wrong format
                "verdict": "borderline",
                "reason": "format_mismatch",
            },
            {
                "prompt": "What is 15 * 7?",
                "output": "The answer is 105.",  # Correct with extra text
                "verdict": "pass",
                "reason": "correct_with_verbosity",
            },
            {
                "prompt": "Write hello world in Python",
                "output": "print('hello world')",  # Missing quotes style
                "verdict": "pass",
                "reason": "stylistically_different",
            },
            {
                "prompt": "What is the capital of Germany?",
                "output": "Berlin is the capital and largest city of Germany.",
                "verdict": "pass",
                "reason": "extra_correct_info",
            },
            {
                "prompt": "Explain recursion briefly",
                "output": "Recursion is when a function calls itself. See: recursion.",
                "verdict": "borderline",
                "reason": "joke_response",
            },
        ]

        for i in range(n):
            example = self.rng.choice(edge_examples)
            yield VerifierTestCase(
                id=f"edge_{i}",
                input_prompt=example["prompt"],
                model_output=example["output"],
                expected_verdict=example["verdict"],
                failure_reason=example.get("reason", ""),
                category="edge_case",
            )

    def generate_all(self) -> list[VerifierTestCase]:
        """Generate complete verifier test corpus."""
        cases = []
        cases.extend(self.known_good(40))
        cases.extend(self.known_bad(40))
        cases.extend(self.edge_cases(30))
        return cases
