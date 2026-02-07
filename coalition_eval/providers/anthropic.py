"""
Anthropic provider for Claude API evaluation.

Requires the anthropic package: pip install anthropic
Used for baseline comparisons against local models.
"""

import os
import time
from typing import Optional

from coalition_eval.providers.base import Provider, ChatMessage, ChatResponse


class AnthropicProvider(Provider):
    """
    Provider for Anthropic Claude models.

    Requires ANTHROPIC_API_KEY environment variable or explicit api_key.
    """

    # Available Claude models
    MODELS = [
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
    ]

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self._api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key."
            )

        self._client = anthropic.Anthropic(api_key=self._api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ) -> ChatResponse:
        """Send chat request to Anthropic API."""
        # Separate system message from conversation
        system_content = ""
        conversation = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                conversation.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        start = time.time()
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_content if system_content else None,
                messages=conversation,
            )
        except Exception as e:
            return ChatResponse(
                content=f"Error: {e}",
                latency_ms=(time.time() - start) * 1000,
                tokens_input=0,
                tokens_output=0,
                model=self._model,
                stop_reason="error",
            )

        latency_ms = (time.time() - start) * 1000

        # Extract content from response
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return ChatResponse(
            content=content,
            latency_ms=latency_ms,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            model=self._model,
            stop_reason=response.stop_reason,
        )

    def list_models(self) -> list[str]:
        """List available Claude models."""
        return self.MODELS.copy()

    def is_available(self) -> bool:
        """Check if API key is set and valid."""
        if not self._api_key:
            return False
        try:
            # Make a minimal request to verify credentials
            self._client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False
