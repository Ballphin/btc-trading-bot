"""Regression tests for whole-repo /review fixes:

- LLM timeout default in openai_client.py (P1 #4)
- CORS allow_origins not '*' when credentials enabled (P0 #1)
- HedgeFund job dict purge guarded by lock (P1 #2)
"""

import importlib
import os
from unittest.mock import patch


def test_openai_client_sets_default_timeout(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.delenv("DEEPSEEK_USE_NVIDIA", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT_S", raising=False)

    from tradingagents.llm_clients.openai_client import OpenAIClient

    with patch(
        "tradingagents.llm_clients.openai_client.NormalizedChatOpenAI"
    ) as mock_cls:
        client = OpenAIClient(model="deepseek-v4-pro", provider="deepseek")
        client.get_llm()

        kwargs = mock_cls.call_args.kwargs
        assert "timeout" in kwargs
        assert float(kwargs["timeout"]) == 90.0


def test_openai_client_respects_env_timeout(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.delenv("DEEPSEEK_USE_NVIDIA", raising=False)
    monkeypatch.setenv("LLM_TIMEOUT_S", "30")

    from tradingagents.llm_clients.openai_client import OpenAIClient

    with patch(
        "tradingagents.llm_clients.openai_client.NormalizedChatOpenAI"
    ) as mock_cls:
        client = OpenAIClient(model="deepseek-v4-pro", provider="deepseek")
        client.get_llm()

        assert mock_cls.call_args.kwargs["timeout"] == 30.0


def test_openai_client_user_timeout_takes_precedence(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.delenv("DEEPSEEK_USE_NVIDIA", raising=False)
    monkeypatch.setenv("LLM_TIMEOUT_S", "30")

    from tradingagents.llm_clients.openai_client import OpenAIClient

    with patch(
        "tradingagents.llm_clients.openai_client.NormalizedChatOpenAI"
    ) as mock_cls:
        client = OpenAIClient(
            model="deepseek-v4-pro", provider="deepseek", timeout=15.0
        )
        client.get_llm()

        assert mock_cls.call_args.kwargs["timeout"] == 15.0


def test_cors_does_not_use_wildcard_with_credentials():
    # Re-importing server is heavy; just inspect the configured middleware.
    import server

    # Find the CORS middleware in the user-stack
    cors_entries = [
        m for m in server.app.user_middleware
        if m.cls.__name__ == "CORSMiddleware"
    ]
    assert cors_entries, "CORS middleware not registered"
    options = cors_entries[0].kwargs
    assert options.get("allow_credentials") is True
    assert options.get("allow_origins") != ["*"], (
        "allow_origins=['*'] with allow_credentials=True is rejected by browsers"
    )
    assert isinstance(options.get("allow_origins"), list)
    assert len(options["allow_origins"]) >= 1


def test_hedgefund_jobs_lock_exists():
    import server

    assert hasattr(server, "_hedgefund_jobs_lock")
    # threading.Lock() returns a _thread.lock; just check it has acquire/release
    lock = server._hedgefund_jobs_lock
    assert hasattr(lock, "acquire") and hasattr(lock, "release")
