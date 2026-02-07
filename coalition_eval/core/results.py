"""
Evaluation result tracking with immutable dataclasses.

Follows Kintsugi patterns:
- Frozen dataclasses for immutability
- Multi-score verdicts (not just pass/fail)
- Full response storage for audit
- Error categorization
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import hashlib
import json


class ErrorCategory(Enum):
    """Categorization of evaluation failures for analysis."""
    NONE = "none"
    FORMAT = "format"          # Output format incorrect
    TIMEOUT = "timeout"        # Request timed out
    REFUSED = "refused"        # Model refused to respond
    HALLUCINATION = "hallucination"  # Factually incorrect
    INCOMPLETE = "incomplete"  # Partial/truncated response
    EXCEPTION = "exception"    # Runtime error
    WRONG_ANSWER = "wrong_answer"  # Correct format, wrong content


@dataclass(frozen=True)
class EvalResult:
    """
    Immutable result from a single evaluation test.

    Uses frozen=True for Kintsugi-style immutability guarantees.
    Stores full response text for audit and debugging.
    """
    test_name: str
    model: str
    passed: bool
    score: float              # 0.0-1.0 partial credit
    latency_ms: float
    tokens_used: int
    tokens_per_sec: float
    response_text: str        # Full response for audit
    error_category: ErrorCategory = ErrorCategory.NONE
    error_message: Optional[str] = None
    metadata: tuple = ()      # Frozen dict alternative

    def __post_init__(self):
        """Validate score is in range."""
        if not 0.0 <= self.score <= 1.0:
            object.__setattr__(self, 'score', max(0.0, min(1.0, self.score)))

    @property
    def result_id(self) -> str:
        """Generate unique ID for this result."""
        content = f"{self.test_name}:{self.model}:{self.response_text[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_name": self.test_name,
            "model": self.model,
            "passed": self.passed,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "tokens_per_sec": self.tokens_per_sec,
            "response_text": self.response_text,
            "error_category": self.error_category.value,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
            "result_id": self.result_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalResult":
        """Create from dictionary."""
        return cls(
            test_name=data["test_name"],
            model=data["model"],
            passed=data["passed"],
            score=data["score"],
            latency_ms=data["latency_ms"],
            tokens_used=data.get("tokens_used", 0),
            tokens_per_sec=data.get("tokens_per_sec", 0.0),
            response_text=data["response_text"],
            error_category=ErrorCategory(data.get("error_category", "none")),
            error_message=data.get("error_message"),
            metadata=tuple(data.get("metadata", {}).items()),
        )


@dataclass(frozen=True)
class SystemInfo:
    """Captured system information for reproducibility."""
    hostname: str
    cpu: str
    ram_gb: int
    gpus: tuple[str, ...]
    timestamp: str

    @classmethod
    def capture(cls) -> "SystemInfo":
        """Capture current system information."""
        import os
        import subprocess

        hostname = os.uname().nodename
        cpu = "Unknown"
        ram_gb = 0
        gpus: list[str] = []

        # CPU
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu = line.split(":")[1].strip()
                        break
        except Exception:
            pass

        # RAM
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        ram_gb = int(line.split()[1]) // (1024 * 1024)
                        break
        except Exception:
            pass

        # GPUs
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            gpus = [line.strip() for line in output.strip().split("\n") if line.strip()]
        except Exception:
            pass

        return cls(
            hostname=hostname,
            cpu=cpu,
            ram_gb=ram_gb,
            gpus=tuple(gpus),
            timestamp=datetime.now().isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hostname": self.hostname,
            "cpu": self.cpu,
            "ram_gb": self.ram_gb,
            "gpus": list(self.gpus),
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class EvalRun:
    """
    Complete evaluation run with all results and metadata.

    Immutable aggregate of EvalResults with computed summary statistics.
    """
    run_id: str
    model: str
    tier: int                  # 1-4 evaluation tier
    results: tuple[EvalResult, ...]
    system_info: SystemInfo
    started_at: str
    completed_at: str

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def avg_score(self) -> float:
        return sum(r.score for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_used for r in self.results)

    @property
    def error_breakdown(self) -> dict[str, int]:
        """Count of each error category."""
        breakdown: dict[str, int] = {}
        for r in self.results:
            cat = r.error_category.value
            breakdown[cat] = breakdown.get(cat, 0) + 1
        return breakdown

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "model": self.model,
            "tier": self.tier,
            "results": [r.to_dict() for r in self.results],
            "system_info": self.system_info.to_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "summary": {
                "passed": self.passed_count,
                "total": self.total_count,
                "pass_rate": self.pass_rate,
                "avg_score": self.avg_score,
                "avg_latency_ms": self.avg_latency_ms,
                "total_tokens": self.total_tokens,
                "error_breakdown": self.error_breakdown,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalRun":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            model=data["model"],
            tier=data.get("tier", 1),
            results=tuple(EvalResult.from_dict(r) for r in data["results"]),
            system_info=SystemInfo(
                hostname=data["system_info"]["hostname"],
                cpu=data["system_info"]["cpu"],
                ram_gb=data["system_info"]["ram_gb"],
                gpus=tuple(data["system_info"]["gpus"]),
                timestamp=data["system_info"]["timestamp"],
            ),
            started_at=data["started_at"],
            completed_at=data["completed_at"],
        )


def create_run_id() -> str:
    """Generate unique run ID."""
    import uuid
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
