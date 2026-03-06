"""Tests for data models."""

from __future__ import annotations

from crashbytes_ai_client.models import (
    ChatMessage,
    ChatResponse,
    LLMProvider,
    TokenUsage,
)


class TestLLMProvider:
    def test_values(self) -> None:
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.OLLAMA.value == "ollama"

    def test_from_value(self) -> None:
        assert LLMProvider("openai") is LLMProvider.OPENAI
        assert LLMProvider("anthropic") is LLMProvider.ANTHROPIC
        assert LLMProvider("ollama") is LLMProvider.OLLAMA

    def test_members_count(self) -> None:
        assert len(LLMProvider) == 3


class TestChatMessage:
    def test_creation(self) -> None:
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_to_dict(self) -> None:
        msg = ChatMessage(role="assistant", content="Hi there")
        assert msg.to_dict() == {"role": "assistant", "content": "Hi there"}

    def test_frozen(self) -> None:
        msg = ChatMessage(role="user", content="test")
        try:
            msg.role = "system"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass

    def test_equality(self) -> None:
        a = ChatMessage(role="user", content="Hi")
        b = ChatMessage(role="user", content="Hi")
        assert a == b

    def test_inequality(self) -> None:
        a = ChatMessage(role="user", content="Hi")
        b = ChatMessage(role="assistant", content="Hi")
        assert a != b


class TestTokenUsage:
    def test_defaults(self) -> None:
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_values(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


class TestChatResponse:
    def test_creation(self) -> None:
        usage = TokenUsage(prompt_tokens=5, completion_tokens=15, total_tokens=20)
        resp = ChatResponse(
            content="Hello!",
            model="gpt-4o",
            usage=usage,
            provider=LLMProvider.OPENAI,
        )
        assert resp.content == "Hello!"
        assert resp.model == "gpt-4o"
        assert resp.usage.total_tokens == 20
        assert resp.provider is LLMProvider.OPENAI
        assert resp.raw_response == {}

    def test_with_raw_response(self) -> None:
        raw = {"id": "chatcmpl-123", "object": "chat.completion"}
        resp = ChatResponse(
            content="Hi",
            model="gpt-4o",
            usage=TokenUsage(),
            provider=LLMProvider.OPENAI,
            raw_response=raw,
        )
        assert resp.raw_response == raw

    def test_frozen(self) -> None:
        resp = ChatResponse(
            content="x",
            model="m",
            usage=TokenUsage(),
            provider=LLMProvider.OPENAI,
        )
        try:
            resp.content = "y"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass
