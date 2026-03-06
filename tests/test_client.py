"""Tests for LLMClient — mocks all HTTP calls."""

from __future__ import annotations

import io
import json
import os
import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crashbytes_ai_client.client import LLMClient, LLMClientError
from crashbytes_ai_client.models import ChatMessage, LLMProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_response(data: dict[str, Any], status: int = 200) -> MagicMock:
    """Build a mock that behaves like an ``http.client.HTTPResponse``."""
    body = json.dumps(data).encode("utf-8")
    mock = MagicMock()
    mock.read.return_value = body
    mock.status = status
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestClientInit:
    def test_defaults_openai(self) -> None:
        client = LLMClient(LLMProvider.OPENAI, api_key="sk-test")
        assert client.provider is LLMProvider.OPENAI
        assert client.api_key == "sk-test"
        assert "openai.com" in client.base_url
        assert client.model == "gpt-4o"

    def test_defaults_anthropic(self) -> None:
        client = LLMClient(LLMProvider.ANTHROPIC, api_key="ant-test")
        assert client.provider is LLMProvider.ANTHROPIC
        assert "anthropic.com" in client.base_url

    def test_defaults_ollama(self) -> None:
        client = LLMClient(LLMProvider.OLLAMA)
        assert client.api_key is None
        assert "localhost" in client.base_url
        assert client.model == "llama3"

    def test_custom_base_url(self) -> None:
        client = LLMClient(LLMProvider.OPENAI, api_key="k", base_url="https://my-proxy.example")
        assert client.base_url == "https://my-proxy.example"

    def test_trailing_slash_stripped(self) -> None:
        client = LLMClient(LLMProvider.OPENAI, api_key="k", base_url="https://example.com/")
        assert not client.base_url.endswith("/")

    def test_custom_model(self) -> None:
        client = LLMClient(LLMProvider.OPENAI, api_key="k", model="gpt-3.5-turbo")
        assert client.model == "gpt-3.5-turbo"

    def test_api_key_from_env_openai(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            client = LLMClient(LLMProvider.OPENAI)
            assert client.api_key == "env-key"

    def test_api_key_from_env_anthropic(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-ant"}):
            client = LLMClient(LLMProvider.ANTHROPIC)
            assert client.api_key == "env-ant"

    def test_explicit_key_overrides_env(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            client = LLMClient(LLMProvider.OPENAI, api_key="explicit")
            assert client.api_key == "explicit"


# ---------------------------------------------------------------------------
# OpenAI chat
# ---------------------------------------------------------------------------

class TestOpenAIChat:
    @patch("crashbytes_ai_client.client.urllib.request.urlopen")
    def test_basic_chat(self, mock_urlopen: MagicMock) -> None:
        response_data = {
            "choices": [{"message": {"content": "Hello back!"}}],
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_urlopen.return_value = _fake_response(response_data)

        client = LLMClient(LLMProvider.OPENAI, api_key="sk-test")
        result = client.chat([ChatMessage(role="user", content="Hi")])

        assert result.content == "Hello back!"
        assert result.model == "gpt-4o"
        assert result.usage.total_tokens == 15
        assert result.provider is LLMProvider.OPENAI

        # Verify the request was built correctly
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "/v1/chat/completions" in req.full_url
        body = json.loads(req.data)
        assert body["model"] == "gpt-4o"
        assert body["messages"] == [{"role": "user", "content": "Hi"}]

    @patch("crashbytes_ai_client.client.urllib.request.urlopen")
    def test_model_override(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _fake_response({
            "choices": [{"message": {"content": "ok"}}],
            "model": "gpt-3.5-turbo",
            "usage": {},
        })
        client = LLMClient(LLMProvider.OPENAI, api_key="sk-test")
        client.chat([ChatMessage(role="user", content="Hi")], model="gpt-3.5-turbo")

        body = json.loads(mock_urlopen.call_args[0][0].data)
        assert body["model"] == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# Anthropic chat
# ---------------------------------------------------------------------------

class TestAnthropicChat:
    @patch("crashbytes_ai_client.client.urllib.request.urlopen")
    def test_basic_chat(self, mock_urlopen: MagicMock) -> None:
        response_data = {
            "content": [{"type": "text", "text": "Hi from Claude"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 8, "output_tokens": 12},
        }
        mock_urlopen.return_value = _fake_response(response_data)

        client = LLMClient(LLMProvider.ANTHROPIC, api_key="ant-key")
        result = client.chat([ChatMessage(role="user", content="Hello")])

        assert result.content == "Hi from Claude"
        assert result.provider is LLMProvider.ANTHROPIC
        assert result.usage.prompt_tokens == 8
        assert result.usage.completion_tokens == 12
        assert result.usage.total_tokens == 20

    @patch("crashbytes_ai_client.client.urllib.request.urlopen")
    def test_system_message_extracted(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _fake_response({
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {},
        })

        client = LLMClient(LLMProvider.ANTHROPIC, api_key="ant-key")
        client.chat([
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hi"),
        ])

        body = json.loads(mock_urlopen.call_args[0][0].data)
        assert body["system"] == "You are helpful."
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"

    @patch("crashbytes_ai_client.client.urllib.request.urlopen")
    def test_anthropic_headers(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _fake_response({
            "content": [], "model": "m", "usage": {},
        })
        client = LLMClient(LLMProvider.ANTHROPIC, api_key="ant-key")
        client.chat([ChatMessage(role="user", content="x")])

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("X-api-key") == "ant-key"
        assert req.get_header("Anthropic-version") == "2023-06-01"

    @patch("crashbytes_ai_client.client.urllib.request.urlopen")
    def test_empty_content_blocks(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _fake_response({
            "content": [], "model": "m", "usage": {},
        })
        client = LLMClient(LLMProvider.ANTHROPIC, api_key="k")
        result = client.chat([ChatMessage(role="user", content="x")])
        assert result.content == ""


# ---------------------------------------------------------------------------
# Ollama chat
# ---------------------------------------------------------------------------

class TestOllamaChat:
    @patch("crashbytes_ai_client.client.urllib.request.urlopen")
    def test_basic_chat(self, mock_urlopen: MagicMock) -> None:
        response_data = {
            "message": {"role": "assistant", "content": "Ollama says hi"},
            "model": "llama3",
            "prompt_eval_count": 6,
            "eval_count": 10,
        }
        mock_urlopen.return_value = _fake_response(response_data)

        client = LLMClient(LLMProvider.OLLAMA)
        result = client.chat([ChatMessage(role="user", content="Hello")])

        assert result.content == "Ollama says hi"
        assert result.provider is LLMProvider.OLLAMA
        assert result.usage.prompt_tokens == 6
        assert result.usage.completion_tokens == 10
        assert result.usage.total_tokens == 16

    @patch("crashbytes_ai_client.client.urllib.request.urlopen")
    def test_ollama_request_format(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _fake_response({
            "message": {"content": "ok"}, "model": "llama3",
        })
        client = LLMClient(LLMProvider.OLLAMA)
        client.chat(
            [ChatMessage(role="user", content="Hi")],
            temperature=0.5,
            max_tokens=512,
        )

        body = json.loads(mock_urlopen.call_args[0][0].data)
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0.5
        assert body["options"]["num_predict"] == 512


# ---------------------------------------------------------------------------
# complete() convenience method
# ---------------------------------------------------------------------------

class TestComplete:
    @patch("crashbytes_ai_client.client.urllib.request.urlopen")
    def test_complete(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _fake_response({
            "choices": [{"message": {"content": "Done"}}],
            "model": "gpt-4o",
            "usage": {},
        })
        client = LLMClient(LLMProvider.OPENAI, api_key="k")
        result = client.complete("Summarize this.")
        assert result.content == "Done"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    @patch("crashbytes_ai_client.client.urllib.request.urlopen")
    def test_http_error(self, mock_urlopen: MagicMock) -> None:
        error_body = io.BytesIO(b'{"error": "bad request"}')
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=error_body,
        )
        client = LLMClient(LLMProvider.OPENAI, api_key="k")
        with pytest.raises(LLMClientError, match="API error 400"):
            client.complete("fail")

    @patch("crashbytes_ai_client.client.urllib.request.urlopen")
    def test_url_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        client = LLMClient(LLMProvider.OLLAMA)
        with pytest.raises(LLMClientError, match="Connection error"):
            client.complete("fail")
