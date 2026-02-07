"""
Ollama provider for local model evaluation.

Connects to Ollama's OpenAI-compatible API at localhost:11434.
"""

import time
from typing import Optional

import requests

from coalition_eval.providers.base import Provider, ChatMessage, ChatResponse


class OllamaProvider(Provider):
    """
    Provider for Ollama local models.

    Uses the /api/chat endpoint with OpenAI-compatible message format.
    """

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

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
        """Send chat request to Ollama."""
        # Convert messages to Ollama format
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        payload = {
            "model": self._model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        start = time.time()
        try:
            response = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return ChatResponse(
                content="",
                latency_ms=(time.time() - start) * 1000,
                tokens_input=0,
                tokens_output=0,
                model=self._model,
                stop_reason="timeout",
            )
        except requests.exceptions.RequestException as e:
            return ChatResponse(
                content=f"Error: {e}",
                latency_ms=(time.time() - start) * 1000,
                tokens_input=0,
                tokens_output=0,
                model=self._model,
                stop_reason="error",
            )

        latency_ms = (time.time() - start) * 1000
        data = response.json()

        content = data.get("message", {}).get("content", "")

        # Ollama provides token counts in response
        tokens_input = data.get("prompt_eval_count", 0)
        tokens_output = data.get("eval_count", 0)

        return ChatResponse(
            content=content,
            latency_ms=latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=self._model,
            stop_reason=data.get("done_reason", "stop"),
        )

    def list_models(self) -> list[str]:
        """List available Ollama models."""
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except requests.exceptions.RequestException:
            return []

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            response = requests.post(
                f"{self._base_url}/api/pull",
                json={"name": model},
                timeout=3600,  # Models can take a while to download
                stream=True,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
