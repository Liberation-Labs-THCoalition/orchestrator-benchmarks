"""
Base provider interface for LLM backends.

All providers must implement this interface to ensure consistent evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ChatMessage:
    """A single message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass(frozen=True)
class ChatResponse:
    """Response from a provider."""
    content: str
    latency_ms: float
    tokens_input: int
    tokens_output: int
    model: str
    stop_reason: Optional[str] = None

    @property
    def tokens_total(self) -> int:
        return self.tokens_input + self.tokens_output

    @property
    def tokens_per_sec(self) -> float:
        if self.latency_ms <= 0:
            return 0.0
        return (self.tokens_output / self.latency_ms) * 1000


class Provider(ABC):
    """
    Abstract base class for LLM providers.

    Implementations must handle:
    - Connection/authentication
    - Message formatting
    - Token counting
    - Latency measurement
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'ollama', 'anthropic')."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Currently configured model."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ) -> ChatResponse:
        """
        Send a chat request and get a response.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds

        Returns:
            ChatResponse with content, latency, and token counts
        """
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available models for this provider."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        ...

    def chat_simple(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        """Convenience method for single-turn prompts."""
        return self.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=temperature,
            max_tokens=max_tokens,
        )
