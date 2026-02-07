"""
LLM Provider Abstraction

Supports multiple backends (Ollama, Anthropic, etc.) through a unified interface.
"""

from coalition_eval.providers.base import Provider, ChatMessage, ChatResponse
from coalition_eval.providers.ollama import OllamaProvider

__all__ = [
    "Provider",
    "ChatMessage",
    "ChatResponse",
    "OllamaProvider",
]

# Optional Anthropic import
try:
    from coalition_eval.providers.anthropic import AnthropicProvider
    __all__.append("AnthropicProvider")
except ImportError:
    pass  # anthropic package not installed


def get_provider(name: str, **kwargs) -> Provider:
    """Factory function to get a provider by name."""
    providers = {
        "ollama": OllamaProvider,
    }

    try:
        from coalition_eval.providers.anthropic import AnthropicProvider
        providers["anthropic"] = AnthropicProvider
    except ImportError:
        pass

    if name not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown provider: {name}. Available: {available}")

    return providers[name](**kwargs)
