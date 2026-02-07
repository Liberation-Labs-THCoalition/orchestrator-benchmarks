"""
SWEI (Semantic Weighted Evaluation Index) scoring.

Ported from Kintsugi verifier patterns - measures behavioral divergence
between expected and actual outputs using multiple similarity metrics.
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass(frozen=True)
class SWEIScore:
    """SWEI divergence score with component breakdown."""
    overall: float          # 0.0 (identical) to 1.0 (completely different)
    jaccard: float          # Token overlap similarity
    semantic: float         # Semantic similarity (if embeddings available)
    structural: float       # Structure matching (JSON, format)
    length_ratio: float     # Output length comparison


def tokenize(text: str) -> set[str]:
    """Simple word tokenization."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts."""
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union)


def structural_similarity(expected: str, actual: str) -> float:
    """
    Compare structural elements between texts.

    Checks for:
    - JSON structure preservation
    - List/numbering format
    - Key-value patterns
    """
    score = 0.0
    checks = 0

    # JSON structure
    expected_has_json = "{" in expected and "}" in expected
    actual_has_json = "{" in actual and "}" in actual
    if expected_has_json:
        checks += 1
        if actual_has_json:
            score += 1.0

    # Numbered lists
    expected_numbered = bool(re.search(r'^\d+\.', expected, re.MULTILINE))
    actual_numbered = bool(re.search(r'^\d+\.', actual, re.MULTILINE))
    if expected_numbered:
        checks += 1
        if actual_numbered:
            score += 1.0

    # Bullet points
    expected_bullets = bool(re.search(r'^[-*]', expected, re.MULTILINE))
    actual_bullets = bool(re.search(r'^[-*]', actual, re.MULTILINE))
    if expected_bullets:
        checks += 1
        if actual_bullets:
            score += 1.0

    return score / checks if checks > 0 else 1.0


def compute_swei_score(
    expected: str,
    actual: str,
    semantic_similarity: Optional[float] = None,
    weights: Optional[dict[str, float]] = None,
) -> SWEIScore:
    """
    Compute SWEI divergence score between expected and actual outputs.

    Args:
        expected: Expected/reference output
        actual: Actual model output
        semantic_similarity: Pre-computed semantic similarity (0-1)
        weights: Component weights (default: equal)

    Returns:
        SWEIScore with overall divergence and component breakdown
    """
    if weights is None:
        weights = {
            "jaccard": 0.4,
            "semantic": 0.3,
            "structural": 0.2,
            "length": 0.1,
        }

    # Jaccard similarity
    jaccard = jaccard_similarity(expected, actual)

    # Semantic similarity (use jaccard as fallback if not provided)
    semantic = semantic_similarity if semantic_similarity is not None else jaccard

    # Structural similarity
    structural = structural_similarity(expected, actual)

    # Length ratio
    len_expected = len(expected)
    len_actual = len(actual)
    if len_expected == 0 and len_actual == 0:
        length_ratio = 1.0
    elif len_expected == 0 or len_actual == 0:
        length_ratio = 0.0
    else:
        length_ratio = min(len_expected, len_actual) / max(len_expected, len_actual)

    # Weighted overall score (similarity, not divergence)
    similarity = (
        weights["jaccard"] * jaccard +
        weights["semantic"] * semantic +
        weights["structural"] * structural +
        weights["length"] * length_ratio
    )

    # Convert to divergence (0 = identical, 1 = completely different)
    divergence = 1.0 - similarity

    return SWEIScore(
        overall=divergence,
        jaccard=1.0 - jaccard,
        semantic=1.0 - semantic,
        structural=1.0 - structural,
        length_ratio=1.0 - length_ratio,
    )


def is_within_tolerance(
    swei: SWEIScore,
    threshold: float = 0.3,
) -> bool:
    """Check if SWEI score is within acceptable tolerance."""
    return swei.overall <= threshold
