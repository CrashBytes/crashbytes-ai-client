"""Tests for the create_client factory function."""

from __future__ import annotations

import pytest

from crashbytes_ai_client.client import LLMClient
from crashbytes_ai_client.factory import create_client
from crashbytes_ai_client.models import LLMProvider


class TestCreateClient:
    def test_from_enum(self) -> None:
        client = create_client(LLMProvider.OPENAI, api_key="sk-test")
        assert isinstance(client, LLMClient)
        assert client.provider is LLMProvider.OPENAI
        assert client.api_key == "sk-test"

    def test_from_string_openai(self) -> None:
        client = create_client("openai", api_key="sk-test")
        assert client.provider is LLMProvider.OPENAI

    def test_from_string_anthropic(self) -> None:
        client = create_client("anthropic", api_key="ant-test")
        assert client.provider is LLMProvider.ANTHROPIC

    def test_from_string_ollama(self) -> None:
        client = create_client("ollama")
        assert client.provider is LLMProvider.OLLAMA

    def test_case_insensitive(self) -> None:
        client = create_client("OpenAI", api_key="k")
        assert client.provider is LLMProvider.OPENAI

    def test_with_whitespace(self) -> None:
        client = create_client("  ollama  ")
        assert client.provider is LLMProvider.OLLAMA

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            create_client("grok")

    def test_forwards_kwargs(self) -> None:
        client = create_client(
            "openai",
            api_key="sk-test",
            base_url="https://proxy.example.com",
            model="gpt-3.5-turbo",
        )
        assert client.api_key == "sk-test"
        assert client.base_url == "https://proxy.example.com"
        assert client.model == "gpt-3.5-turbo"

    def test_all_enum_values_accepted(self) -> None:
        for provider in LLMProvider:
            client = create_client(provider, api_key="test-key")
            assert client.provider is provider

    def test_all_string_values_accepted(self) -> None:
        for provider in LLMProvider:
            client = create_client(provider.value, api_key="test-key")
            assert client.provider is provider
