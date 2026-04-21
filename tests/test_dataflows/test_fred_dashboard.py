"""Stage 2 Commit L — FRED macro dashboard tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from tradingagents.dataflows.fred_client import (
    FREDClient, _is_us_market_hours, get_fred_macro_dashboard,
    market_hours_cache_ttl_sec,
)


class TestMarketHoursTTL:
    def test_inside_market_hours_gets_short_ttl(self):
        # Wed Apr 15 2026 15:00 UTC = 11:00 ET (inside 9:30-16:00)
        dt = datetime(2026, 4, 15, 15, 0, tzinfo=timezone.utc)
        assert _is_us_market_hours(dt) is True
        assert market_hours_cache_ttl_sec(dt) == 15 * 60

    def test_outside_market_hours_gets_long_ttl(self):
        # Wed Apr 15 2026 03:00 UTC = 23:00 ET (outside)
        dt = datetime(2026, 4, 15, 3, 0, tzinfo=timezone.utc)
        assert _is_us_market_hours(dt) is False
        assert market_hours_cache_ttl_sec(dt) == 60 * 60

    def test_weekend_gets_long_ttl(self):
        dt = datetime(2026, 4, 18, 15, 0, tzinfo=timezone.utc)  # Saturday
        assert _is_us_market_hours(dt) is False


class TestFredMacroDashboard:
    def test_no_api_key_graceful_degradation(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        out = get_fred_macro_dashboard("2026-04-15", api_key=None)
        assert "[FRED: no API key" in out

    def test_dashboard_series_includes_commit_L_series(self):
        """Spec: DFF, DGS10, T10Y2Y, DEXUSEU, DCOILWTICO, UNRATE, CPIAUCSL."""
        expected = {"dff", "dgs10", "t10y2y", "dexuseu", "dcoilwtico",
                    "unrate", "cpiaucsl"}
        assert set(FREDClient.DASHBOARD_SERIES) == expected

    @patch("tradingagents.dataflows.fred_client.FREDClient.get_series")
    def test_dashboard_renders_table(self, mock_get, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "x")
        # Provide a small time series so 30d change can be computed.
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-04-01"]),
            "value": [100.0, 101.0, 103.0],
        })
        mock_get.return_value = df
        out = get_fred_macro_dashboard("2026-04-15", api_key="x")
        assert "FRED Macro Dashboard" in out
        for s in FREDClient.DASHBOARD_SERIES:
            assert s.upper() in out
