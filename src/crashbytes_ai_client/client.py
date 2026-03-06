"""Unified LLM client for OpenAI, Anthropic, and Ollama."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from crashbytes_ai_client.models import (
    ChatMessage,
    ChatResponse,
    LLMProvider,
    TokenUsage,
)

_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProvider.OLLAMA: "llama3",
}

_DEFAULT_BASE_URLS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "https://api.openai.com",
    LLMProvider.ANTHROPIC: "https://api.anthropic.com",
    LLMProvider.OLLAMA: "http://localhost:11434",
}


class LLMClientError(Exception):
    """Raised when an LLM API call fails."""


class LLMClient:
    """Unified client for calling LLM provider APIs.

    Supports OpenAI, Anthropic, and Ollama with a consistent interface.
    Uses only stdlib ``urllib`` for HTTP — no third-party dependencies.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.api_key = api_key or self._resolve_api_key(provider)
        self.base_url = (base_url or _DEFAULT_BASE_URLS[provider]).rstrip("/")
        self.model = model or _DEFAULT_MODELS[provider]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Send a chat completion request and return a unified response."""
        resolved_model = model or self.model
        if self.provider == LLMProvider.OPENAI:
            return self._chat_openai(messages, resolved_model, temperature, max_tokens)
        if self.provider == LLMProvider.ANTHROPIC:
            return self._chat_anthropic(messages, resolved_model, temperature, max_tokens)
        return self._chat_ollama(messages, resolved_model, temperature, max_tokens)

    def complete(self, prompt: str, **kwargs: Any) -> ChatResponse:
        """Convenience method: send a single user prompt and get a response."""
        messages = [ChatMessage(role="user", content=prompt)]
        return self.chat(messages, **kwargs)

    # ------------------------------------------------------------------
    # Provider-specific implementations
    # ------------------------------------------------------------------

    def _chat_openai(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        url = f"{self.base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = self._request(url, payload, headers)
        content = data["choices"][0]["message"]["content"]
        usage_raw = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )
        return ChatResponse(
            content=content,
            model=data.get("model", model),
            usage=usage,
            provider=LLMProvider.OPENAI,
            raw_response=data,
        )

    def _chat_anthropic(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        url = f"{self.base_url}/v1/messages"

        # Anthropic requires system messages to be passed separately.
        system_text: str | None = None
        api_messages: list[dict[str, str]] = []
        for m in messages:
            if m.role == "system":
                system_text = m.content
            else:
                api_messages.append(m.to_dict())

        payload: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_text is not None:
            payload["system"] = system_text

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
        }
        data = self._request(url, payload, headers)
        content_blocks = data.get("content", [])
        content = content_blocks[0]["text"] if content_blocks else ""
        usage_raw = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_raw.get("input_tokens", 0),
            completion_tokens=usage_raw.get("output_tokens", 0),
            total_tokens=usage_raw.get("input_tokens", 0)
            + usage_raw.get("output_tokens", 0),
        )
        return ChatResponse(
            content=content,
            model=data.get("model", model),
            usage=usage,
            provider=LLMProvider.ANTHROPIC,
            raw_response=data,
        )

    def _chat_ollama(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        headers = {"Content-Type": "application/json"}
        data = self._request(url, payload, headers)
        message = data.get("message", {})
        content: str = message.get("content", "")
        usage = TokenUsage(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        )
        return ChatResponse(
            content=content,
            model=data.get("model", model),
            usage=usage,
            provider=LLMProvider.OLLAMA,
            raw_response=data,
        )

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _request(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Make a JSON POST request using stdlib urllib."""
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw: bytes = resp.read()
                result: dict[str, Any] = json.loads(raw)
                return result
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(
                f"{self.provider.value} API error {exc.code}: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMClientError(
                f"Connection error for {self.provider.value}: {exc.reason}"
            ) from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_api_key(provider: LLMProvider) -> str | None:
        """Attempt to read an API key from environment variables."""
        env_map: dict[LLMProvider, str] = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        }
        var = env_map.get(provider)
        if var is not None:
            return os.environ.get(var)
        return None  # Ollama doesn't need a key
