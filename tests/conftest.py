"""Shared fixtures, marker registration, and SSE parser for all tests."""

import json
import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime


# ── Marker registration & CLI hooks ─────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.live (require network + API keys)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires network + API keys")
    config.addinivalue_line("markers", "slow: tests that take >5s")
    config.addinivalue_line("markers", "e2e: frontend E2E tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-live"):
        skip_live = pytest.mark.skip(reason="need --run-live to run")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)


# ── Shared fixtures ─────────────────────────────────────────────────

@pytest.fixture
def tmp_eval_results(tmp_path):
    """Provide a temporary eval_results directory tree."""
    shadow = tmp_path / "shadow"
    shadow.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def test_config():
    """Override default_config with safe no-network test settings."""
    return {
        "llm_provider": "openai",
        "deep_think_llm": "gpt-4o",
        "quick_think_llm": "gpt-4o-mini",
        "llm_temperature": 0.0,
        "backtest_mode": True,
    }


@pytest.fixture
def mock_llm():
    """A MagicMock LLM that returns canned responses."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Test LLM response")
    return llm


@pytest.fixture
def mock_portfolio():
    """A Portfolio instance with default test settings."""
    from tradingagents.backtesting.portfolio import Portfolio
    return Portfolio(
        initial_capital=100_000.0,
        position_size_pct=0.25,
        leverage=1.0,
        slippage_bps=5.0,
    )


# ── SSE parser helper for server integration tests ──────────────────

def parse_sse_events(text: str) -> list:
    """Parse raw SSE text/event-stream into a list of dicts.

    Each SSE event looks like:
        event: <type>
        data: <json>

    Returns list of {"event": str|None, "data": dict|str}.
    """
    events = []
    current_event = None
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            raw = line[5:].strip()
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                data = raw
            events.append({"event": current_event, "data": data})
            current_event = None
        elif line == "":
            current_event = None
    return events


async def consume_sse(client, url, max_events=10, timeout=5):
    """Consume SSE events from an httpx AsyncClient stream.

    Args:
        client: httpx.AsyncClient instance
        url: URL to stream from
        max_events: Stop after this many events
        timeout: Seconds before giving up

    Returns:
        List of parsed event dicts.
    """
    import anyio

    events = []
    async with anyio.fail_after(timeout):
        async with client.stream("GET", url) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        events.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        events.append({"raw": line[6:]})
                    if len(events) >= max_events:
                        break
    return events
