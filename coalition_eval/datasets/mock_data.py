"""
Deterministic mock data generators.

Ported from Kintsugi patterns - uses seeded RNG for reproducibility.
"""

from dataclasses import dataclass
from typing import Iterator
import random


@dataclass
class MockSample:
    """A mock evaluation sample with known answer."""
    id: str
    prompt: str
    expected: str
    category: str


class MockDataGenerator:
    """
    Deterministic mock data generator.

    Uses seeded RNG for reproducible test data.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self) -> None:
        """Reset RNG to initial seed."""
        self.rng = random.Random(self.seed)

    def arithmetic(self, n: int = 50) -> Iterator[MockSample]:
        """Generate arithmetic problems."""
        self.reset()
        operations = [
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b),
        ]

        for i in range(n):
            a = self.rng.randint(1, 100)
            b = self.rng.randint(1, 50)
            op_sym, op_fn = self.rng.choice(operations)
            result = op_fn(a, b)

            yield MockSample(
                id=f"arith_{i}",
                prompt=f"What is {a} {op_sym} {b}? Answer with just the number.",
                expected=str(result),
                category="arithmetic",
            )

    def tool_selection(self, n: int = 30) -> Iterator[MockSample]:
        """Generate tool selection problems."""
        self.reset()
        tools = [
            ("calculator", ["compute", "calculate", "math", "add", "multiply"]),
            ("search", ["find", "look up", "search for", "what is"]),
            ("weather", ["weather", "temperature", "forecast", "rain"]),
            ("calendar", ["schedule", "meeting", "appointment", "event"]),
            ("email", ["send", "email", "message", "mail"]),
        ]

        for i in range(n):
            tool_name, keywords = self.rng.choice(tools)
            keyword = self.rng.choice(keywords)

            prompts = [
                f"I need to {keyword} something. Which tool should I use?",
                f"Help me {keyword}. What's the right tool?",
                f"For {keyword}, which tool is appropriate?",
            ]

            yield MockSample(
                id=f"tool_{i}",
                prompt=self.rng.choice(prompts) + f"\nTools: {', '.join(t[0] for t in tools)}",
                expected=tool_name,
                category="tool_selection",
            )

    def instruction_following(self, n: int = 30) -> Iterator[MockSample]:
        """Generate instruction following problems."""
        self.reset()
        formats = [
            ("List exactly 3 items", lambda: 3),
            ("List exactly 5 items", lambda: 5),
            ("Give me 4 examples", lambda: 4),
            ("Provide 2 options", lambda: 2),
        ]

        topics = ["colors", "fruits", "countries", "programming languages", "animals"]

        for i in range(n):
            fmt, count_fn = self.rng.choice(formats)
            topic = self.rng.choice(topics)
            count = count_fn()

            yield MockSample(
                id=f"instr_{i}",
                prompt=f"{fmt} of {topic}. Number each item.",
                expected=str(count),
                category="instruction_following",
            )

    def multi_turn(self, n: int = 20) -> Iterator[MockSample]:
        """Generate multi-turn context problems."""
        self.reset()
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        facts = [
            ("name", "My name is {name}.", "What is my name?"),
            ("color", "My favorite color is {value}.", "What's my favorite color?"),
            ("number", "Remember the number {value}.", "What number did I mention?"),
        ]

        colors = ["blue", "red", "green", "purple", "orange"]
        numbers = list(range(10, 100))

        for i in range(n):
            fact_type, setup, question = self.rng.choice(facts)

            if fact_type == "name":
                value = self.rng.choice(names)
                setup = setup.format(name=value)
            elif fact_type == "color":
                value = self.rng.choice(colors)
                setup = setup.format(value=value)
            else:
                value = str(self.rng.choice(numbers))
                setup = setup.format(value=value)

            yield MockSample(
                id=f"multi_{i}",
                prompt=f"Context: User said '{setup}'\n\nQuestion: {question}",
                expected=value.lower() if fact_type == "name" else value,
                category="multi_turn",
            )


def generate_arithmetic_problems(n: int = 50, seed: int = 42) -> list[MockSample]:
    """Generate arithmetic problems."""
    gen = MockDataGenerator(seed)
    return list(gen.arithmetic(n))


def generate_tool_selection_problems(n: int = 30, seed: int = 42) -> list[MockSample]:
    """Generate tool selection problems."""
    gen = MockDataGenerator(seed)
    return list(gen.tool_selection(n))


def generate_instruction_following_problems(n: int = 30, seed: int = 42) -> list[MockSample]:
    """Generate instruction following problems."""
    gen = MockDataGenerator(seed)
    return list(gen.instruction_following(n))
