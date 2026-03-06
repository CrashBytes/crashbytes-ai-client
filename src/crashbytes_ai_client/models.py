"""Data models for the unified LLM client."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class ChatMessage:
    """A single chat message with a role and content."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to a plain dictionary."""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class TokenUsage:
    """Token usage information from an API response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ChatResponse:
    """Unified response from any LLM provider."""

    content: str
    model: str
    usage: TokenUsage
    provider: LLMProvider
    raw_response: dict[str, Any] = field(default_factory=dict)
