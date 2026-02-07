"""
HuggingFace dataset loading utilities.

Provides standardized access to evaluation datasets.
"""

from dataclasses import dataclass, field
from typing import Iterator, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for loading a dataset."""
    name: str
    split: str = "test"
    subset: Optional[str] = None
    sample_size: int = 100
    shuffle: bool = False
    seed: int = 42


KNOWN_DATASETS = {
    "bfcl": DatasetConfig(
        name="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        split="train",
        sample_size=100,
    ),
    "gsm8k": DatasetConfig(
        name="gsm8k",
        subset="main",
        split="test",
        sample_size=100,
    ),
    "mmlu": DatasetConfig(
        name="cais/mmlu",
        subset="all",
        split="test",
        sample_size=100,
    ),
    "hellaswag": DatasetConfig(
        name="hellaswag",
        split="validation",
        sample_size=100,
    ),
    "truthfulqa": DatasetConfig(
        name="truthful_qa",
        subset="generation",
        split="validation",
        sample_size=100,
    ),
}


@dataclass
class Sample:
    """A single evaluation sample."""
    id: str
    prompt: str
    expected: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def list_available_datasets() -> list[str]:
    """List known dataset names."""
    return list(KNOWN_DATASETS.keys())


def load_dataset_samples(config: DatasetConfig) -> Iterator[Sample]:
    """Load samples from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required: pip install datasets")

    logger.info(f"Loading dataset: {config.name}")

    if config.subset:
        ds = load_dataset(config.name, config.subset, split=config.split)
    else:
        ds = load_dataset(config.name, split=config.split)

    if config.shuffle:
        ds = ds.shuffle(seed=config.seed)

    if len(ds) > config.sample_size:
        ds = ds.select(range(config.sample_size))

    for i, item in enumerate(ds):
        yield _convert_to_sample(config.name, i, item)


def _convert_to_sample(dataset_name: str, idx: int, item: dict) -> Sample:
    """Convert dataset item to Sample format."""
    if "gsm8k" in dataset_name.lower():
        return _convert_gsm8k(idx, item)
    elif "bfcl" in dataset_name.lower() or "function" in dataset_name.lower():
        return _convert_bfcl(idx, item)
    else:
        return Sample(
            id=f"{dataset_name}_{idx}",
            prompt=str(item.get("question", item.get("prompt", str(item)))),
            expected=str(item.get("answer", item.get("output", ""))),
            metadata=item,
        )


def _convert_gsm8k(idx: int, item: dict) -> Sample:
    """Convert GSM8K sample."""
    question = item.get("question", "")
    answer = item.get("answer", "")
    final = answer.split("####")[-1].strip() if "####" in answer else ""
    return Sample(
        id=f"gsm8k_{idx}",
        prompt=f"Solve step by step:\n{question}",
        expected=final,
        metadata={"full_answer": answer},
    )


def _convert_bfcl(idx: int, item: dict) -> Sample:
    """Convert BFCL sample."""
    return Sample(
        id=f"bfcl_{idx}",
        prompt=item.get("question", ""),
        expected=str(item.get("function", "")),
        metadata={"category": item.get("category", "")},
    )


def load_by_name(name: str, sample_size: Optional[int] = None) -> Iterator[Sample]:
    """Load a known dataset by name."""
    if name not in KNOWN_DATASETS:
        raise ValueError(f"Unknown: {name}. Available: {list_available_datasets()}")

    config = KNOWN_DATASETS[name]
    if sample_size:
        config = DatasetConfig(
            name=config.name,
            split=config.split,
            subset=config.subset,
            sample_size=sample_size,
            shuffle=config.shuffle,
            seed=config.seed,
        )
    return load_dataset_samples(config)
