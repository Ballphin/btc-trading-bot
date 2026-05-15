"""Regression test: cfg must be bound before verified-outcome persistence in _score_pending_pulses.

The bug: when only NEUTRAL entries existed, `modified` became True but `cfg`
was only assigned inside the per-entry scoring branch, causing an UnboundLocalError
at the post-write persistence path.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_score_pending_pulses_neutral_only_does_not_crash(tmp_path):
    """When all entries are NEUTRAL, _score_pending_pulses must not raise UnboundLocalError on cfg."""
    # Build a minimal pulse directory with one NEUTRAL entry (unscored)
    ticker_dir = tmp_path / "BTC-USD"
    ticker_dir.mkdir()
    pulse_path = ticker_dir / "pulse.jsonl"

    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    entry = {"ts": two_hours_ago, "signal": "NEUTRAL", "confidence": 0.0, "price": 100000}
    pulse_path.write_text(json.dumps(entry) + "\n")

    # Patch PULSE_DIR to our tmp_path and HyperliquidClient to avoid network
    with patch("server.PULSE_DIR", tmp_path), \
         patch("tradingagents.dataflows.hyperliquid_client.HyperliquidClient"):
        import server
        coro = server._score_pending_pulses()
        asyncio.run(coro)

    # Re-read: the NEUTRAL entry should now be marked scored
    reloaded = [json.loads(l) for l in pulse_path.read_text().strip().splitlines()]
    assert reloaded[0]["scored"] is True
