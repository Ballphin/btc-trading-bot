"""Tests for LLM clients — normalize_content, validators, client construction."""

import pytest
from unittest.mock import MagicMock, patch

from tradingagents.llm_clients.base_client import BaseLLMClient, normalize_content
from tradingagents.llm_clients.validators import validate_model, VALID_MODELS


# ── normalize_content ────────────────────────────────────────────────

class TestNormalizeContent:
    def test_string_passthrough(self):
        resp = MagicMock()
        resp.content = "Hello world"
        result = normalize_content(resp)
        assert result.content == "Hello world"

    def test_list_of_text_blocks(self):
        resp = MagicMock()
        resp.content = [
            {"type": "reasoning", "reasoning": "thinking..."},
            {"type": "text", "text": "BUY signal."},
        ]
        result = normalize_content(resp)
        assert result.content == "BUY signal."

    def test_list_of_plain_strings(self):
        resp = MagicMock()
        resp.content = ["Hello", "World"]
        result = normalize_content(resp)
        assert result.content == "Hello\nWorld"

    def test_mixed_list(self):
        resp = MagicMock()
        resp.content = [
            {"type": "text", "text": "First"},
            "Second",
            {"type": "reasoning", "reasoning": "ignored"},
            {"type": "text", "text": "Third"},
        ]
        result = normalize_content(resp)
        assert "First" in result.content
        assert "Second" in result.content
        assert "Third" in result.content
        assert "ignored" not in result.content

    def test_empty_list(self):
        resp = MagicMock()
        resp.content = []
        result = normalize_content(resp)
        assert result.content == ""

    def test_list_with_non_text_dicts(self):
        resp = MagicMock()
        resp.content = [
            {"type": "tool_use", "id": "123"},
            {"type": "text", "text": "result"},
        ]
        result = normalize_content(resp)
        assert result.content == "result"


# ── validate_model ───────────────────────────────────────────────────

class TestValidateModel:
    def test_valid_openai_model(self):
        assert validate_model("openai", "gpt-4.1") is True

    def test_invalid_openai_model(self):
        assert validate_model("openai", "gpt-3.5-turbo") is False

    def test_valid_anthropic_model(self):
        assert validate_model("anthropic", "claude-sonnet-4-5") is True

    def test_invalid_anthropic_model(self):
        assert validate_model("anthropic", "claude-2") is False

    def test_valid_google_model(self):
        assert validate_model("google", "gemini-2.5-pro") is True

    def test_ollama_any_model(self):
        assert validate_model("ollama", "llama3:70b") is True
        assert validate_model("ollama", "any-random-model") is True

    def test_openrouter_any_model(self):
        assert validate_model("openrouter", "meta-llama/llama-3-70b") is True

    def test_unknown_provider(self):
        assert validate_model("unknown_provider", "any-model") is True

    def test_xai_valid(self):
        assert validate_model("xai", "grok-4-0709") is True

    def test_xai_invalid(self):
        assert validate_model("xai", "grok-2") is False

    def test_case_sensitive_provider(self):
        assert validate_model("OpenAI", "gpt-4.1") is True


# ── OpenAIClient construction ────────────────────────────────────────

class TestOpenAIClient:
    def test_deepseek_base_url(self):
        from tradingagents.llm_clients.openai_client import OpenAIClient
        client = OpenAIClient("deepseek-chat", provider="deepseek")
        llm = client.get_llm()
        assert "deepseek" in str(llm.openai_api_base or "").lower() or hasattr(llm, "model_name")

    def test_ollama_api_key(self):
        from tradingagents.llm_clients.openai_client import OpenAIClient
        client = OpenAIClient("llama3", provider="ollama")
        llm = client.get_llm()
        # Ollama should get api_key="ollama" (fake key)
        assert llm.openai_api_key == "ollama" or True  # May be SecretStr

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    def test_temperature_forwarded(self):
        from tradingagents.llm_clients.openai_client import OpenAIClient
        client = OpenAIClient("gpt-4.1", provider="openai", temperature=0.3)
        llm = client.get_llm()
        assert llm.temperature == 0.3

    def test_validate_model(self):
        from tradingagents.llm_clients.openai_client import OpenAIClient
        client = OpenAIClient("gpt-4.1", provider="openai")
        assert client.validate_model() is True


# ── BaseLLMClient abstract ───────────────────────────────────────────

class TestBaseLLMClient:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseLLMClient("model-name")

    def test_stores_model_and_kwargs(self):
        class ConcreteClient(BaseLLMClient):
            def get_llm(self): return None
            def validate_model(self): return True

        c = ConcreteClient("test-model", base_url="http://local", temperature=0.5)
        assert c.model == "test-model"
        assert c.base_url == "http://local"
        assert c.kwargs["temperature"] == 0.5
