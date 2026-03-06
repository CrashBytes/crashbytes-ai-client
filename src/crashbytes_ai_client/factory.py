"""Factory function for creating LLM clients."""

from __future__ import annotations

from typing import Any

from crashbytes_ai_client.client import LLMClient
from crashbytes_ai_client.models import LLMProvider


def create_client(provider_or_name: LLMProvider | str, **kwargs: Any) -> LLMClient:
    """Create an :class:`LLMClient` from a provider enum or string name.

    Args:
        provider_or_name: An :class:`LLMProvider` member or a string such as
            ``"openai"``, ``"anthropic"``, or ``"ollama"`` (case-insensitive).
        **kwargs: Forwarded to :class:`LLMClient.__init__`.

    Returns:
        A configured :class:`LLMClient` instance.

    Raises:
        ValueError: If the provider name is not recognised.
    """
    if isinstance(provider_or_name, LLMProvider):
        provider = provider_or_name
    else:
        name = provider_or_name.strip().lower()
        try:
            provider = LLMProvider(name)
        except ValueError:
            valid = ", ".join(p.value for p in LLMProvider)
            raise ValueError(
                f"Unknown provider {provider_or_name!r}. Valid providers: {valid}"
            ) from None

    return LLMClient(provider, **kwargs)
