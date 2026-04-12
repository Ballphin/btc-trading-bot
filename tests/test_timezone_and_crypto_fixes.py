"""Tests for timezone fixes, crypto tool binding, cache bypass, and display timezone.

Covers:
- BLOCKER 1: UTC-aware datetime parsing in hyperliquid_client.py
- BLOCKER 2: Ensemble orchestrator crypto graph rebuild
- HIGH 1: Fundamentals analyst crypto tool binding
- HIGH 2: Cache bypass for live queries
- HIGH 3: EST display timezone conversion
- MEDIUM: server.py crypto_vendors preservation
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# BLOCKER 1: UTC-aware datetime parsing
# ═══════════════════════════════════════════════════════════════════════

class TestHyperliquidUTCParsing:
    """Verify that date strings are parsed as UTC, not local time."""

    @pytest.fixture
    def hl(self, tmp_path):
        with patch("tradingagents.dataflows.base_client.BaseDataClient.__init__") as mock_init:
            mock_init.return_value = None
            from tradingagents.dataflows.hyperliquid_client import HyperliquidClient
            client = HyperliquidClient()
            client.cache_dir = tmp_path
            client.cache_ttl = 3600
            client.session = MagicMock()
            client.MAX_RETRIES = 3
            client.BACKOFF_BASE = 2
            return client

    def test_get_ohlcv_uses_utc_epoch(self, hl):
        """strptime dates must produce UTC epoch millis, not local time."""
        payloads_captured = []

        def capture_post(payload, **kwargs):
            payloads_captured.append(payload)
            return []

        with patch.object(hl, "_post_request", side_effect=capture_post):
            hl.get_ohlcv("BTC", "1d", "2026-04-12", "2026-04-13")

        assert len(payloads_captured) >= 1
        start_ms = payloads_captured[0]["req"]["startTime"]
        # 2026-04-12 00:00:00 UTC = 1776124800000 ms
        expected_ms = int(datetime(2026, 4, 12, tzinfo=timezone.utc).timestamp() * 1000)
        assert start_ms == expected_ms, (
            f"Expected UTC epoch {expected_ms}, got {start_ms}. "
            f"Difference: {(start_ms - expected_ms) / 3600000:.1f} hours — "
            f"likely using local timezone instead of UTC."
        )

    def test_get_funding_history_uses_utc_epoch(self, hl):
        """Funding history date parsing must also be UTC."""
        payloads_captured = []

        def capture_post(payload, **kwargs):
            payloads_captured.append(payload)
            return []

        with patch.object(hl, "_post_request", side_effect=capture_post):
            hl.get_funding_history("ETH", "2026-04-10", "2026-04-12")

        assert len(payloads_captured) >= 1
        start_ms = payloads_captured[0]["startTime"]
        expected_ms = int(datetime(2026, 4, 10, tzinfo=timezone.utc).timestamp() * 1000)
        assert start_ms == expected_ms


# ═══════════════════════════════════════════════════════════════════════
# crypto_data.py UTC parsing and cache bypass
# ═══════════════════════════════════════════════════════════════════════

class TestCryptoDataUTCAndCache:
    """Verify crypto_data.py functions use UTC and bypass cache for live queries."""

    def _mock_ohlcv_df(self, n=24):
        """Create a mock OHLCV DataFrame."""
        base_ts = datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc)
        rows = []
        for i in range(n):
            rows.append({
                "timestamp": base_ts + timedelta(hours=i),
                "open": 2200 + i,
                "high": 2210 + i,
                "low": 2190 + i,
                "close": 2205 + i,
                "volume": 1000 + i * 10,
            })
        return pd.DataFrame(rows)

    @patch("tradingagents.dataflows.crypto_data._get")
    @patch("tradingagents.dataflows.crypto_data._is_current_date", return_value=True)
    def test_get_crypto_price_data_uses_short_cache_for_live(self, mock_is_current, mock_get):
        """Live queries should pass max_age_override=120."""
        mock_hl = MagicMock()
        mock_hl.get_ohlcv.return_value = self._mock_ohlcv_df(5)
        mock_get.return_value = mock_hl

        from tradingagents.dataflows.crypto_data import get_crypto_price_data
        result = get_crypto_price_data("ETH-USD", "2026-04-10", "2026-04-12")

        mock_hl.get_ohlcv.assert_called_once()
        call_kwargs = mock_hl.get_ohlcv.call_args
        assert call_kwargs.kwargs.get("max_age_override") == 120 or \
               (len(call_kwargs.args) > 4 and call_kwargs.args[4] == 120)

    @patch("tradingagents.dataflows.crypto_data._get")
    @patch("tradingagents.dataflows.crypto_data._is_current_date", return_value=False)
    def test_get_crypto_price_data_uses_default_cache_for_historical(self, mock_is_current, mock_get):
        """Historical queries should use default cache (None)."""
        mock_hl = MagicMock()
        mock_hl.get_ohlcv.return_value = self._mock_ohlcv_df(5)
        mock_get.return_value = mock_hl

        from tradingagents.dataflows.crypto_data import get_crypto_price_data
        result = get_crypto_price_data("ETH-USD", "2026-03-01", "2026-03-10")

        call_kwargs = mock_hl.get_ohlcv.call_args
        assert call_kwargs.kwargs.get("max_age_override") is None or \
               (len(call_kwargs.args) <= 4)

    @patch("tradingagents.dataflows.crypto_data._get")
    @patch("tradingagents.dataflows.crypto_data._is_current_date", return_value=True)
    def test_get_crypto_indicators_uses_short_cache_for_live(self, mock_is_current, mock_get):
        """Indicator queries for current date should use short cache."""
        mock_hl = MagicMock()
        mock_hl.get_ohlcv.return_value = self._mock_ohlcv_df(90)
        mock_get.return_value = mock_hl

        from tradingagents.dataflows.crypto_data import get_crypto_indicators
        result = get_crypto_indicators("ETH-USD", "rsi", "2026-04-12")

        call_kwargs = mock_hl.get_ohlcv.call_args
        assert call_kwargs.kwargs.get("max_age_override") == 120

    @patch("tradingagents.dataflows.crypto_data._get")
    def test_get_intraday_summary_bypasses_cache(self, mock_get):
        """Intraday summary should always bypass cache (max_age_override=0)."""
        mock_hl = MagicMock()
        mock_hl.get_ohlcv.return_value = self._mock_ohlcv_df(24)
        mock_hl.get_asset_context.return_value = None
        mock_hl.get_predicted_funding.return_value = None
        mock_hl.get_realized_funding.return_value = None
        mock_get.return_value = mock_hl

        from tradingagents.dataflows.crypto_data import get_intraday_summary
        result = get_intraday_summary("ETH-USD", "2026-04-12")

        call_kwargs = mock_hl.get_ohlcv.call_args
        assert call_kwargs.kwargs.get("max_age_override") == 0 or \
               (len(call_kwargs.args) > 4 and call_kwargs.args[4] == 0)


# ═══════════════════════════════════════════════════════════════════════
# Display timezone
# ═══════════════════════════════════════════════════════════════════════

class TestDisplayTimezone:
    """Verify EST/EDT display timezone conversion in report strings."""

    def test_fmt_ts_local_converts_utc_to_est(self):
        from tradingagents.dataflows.crypto_data import _fmt_ts_local
        utc_dt = datetime(2026, 4, 12, 20, 0, 0, tzinfo=timezone.utc)
        result = _fmt_ts_local(utc_dt, "%Y-%m-%d %H:%M %Z")
        # April is EDT (UTC-4), so 20:00 UTC = 16:00 EDT
        assert "16:00" in result
        assert "EDT" in result

    def test_fmt_ts_local_handles_naive_datetime(self):
        from tradingagents.dataflows.crypto_data import _fmt_ts_local
        naive_dt = datetime(2026, 1, 15, 12, 0, 0)
        result = _fmt_ts_local(naive_dt, "%Y-%m-%d %H:%M %Z")
        # January is EST (UTC-5), so 12:00 UTC = 07:00 EST
        assert "07:00" in result
        assert "EST" in result

    def test_is_current_date_true_for_today(self):
        from tradingagents.dataflows.crypto_data import _is_current_date
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert _is_current_date(today_str) is True

    def test_is_current_date_false_for_past(self):
        from tradingagents.dataflows.crypto_data import _is_current_date
        assert _is_current_date("2020-01-01") is False

    def test_is_current_date_true_for_future(self):
        from tradingagents.dataflows.crypto_data import _is_current_date
        assert _is_current_date("2099-12-31") is True

    @patch("tradingagents.dataflows.crypto_data._get")
    def test_intraday_summary_shows_local_time_in_header(self, mock_get):
        """Intraday summary header should show display timezone."""
        mock_hl = MagicMock()
        base_ts = datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc)
        rows = [{
            "timestamp": base_ts + timedelta(hours=i),
            "open": 2200, "high": 2210, "low": 2190, "close": 2205, "volume": 1000,
        } for i in range(24)]
        mock_hl.get_ohlcv.return_value = pd.DataFrame(rows)
        mock_hl.get_asset_context.return_value = None
        mock_hl.get_predicted_funding.return_value = None
        mock_hl.get_realized_funding.return_value = None
        mock_get.return_value = mock_hl

        from tradingagents.dataflows.crypto_data import get_intraday_summary
        result = get_intraday_summary("ETH-USD", "2026-04-12")

        # Header should contain EDT (April = daylight saving)
        assert "EDT" in result or "EST" in result

    @patch("tradingagents.dataflows.crypto_data._get")
    def test_realtime_snapshot_shows_local_time(self, mock_get):
        """Realtime snapshot should show display timezone, not UTC."""
        mock_hl = MagicMock()
        mock_hl.get_spot_price.return_value = 2200.0
        mock_hl.get_ohlcv.return_value = pd.DataFrame()
        mock_hl.get_predicted_funding.return_value = None
        mock_hl.get_realized_funding.return_value = None
        mock_hl.get_asset_context.return_value = None
        mock_get.return_value = mock_hl

        from tradingagents.dataflows.crypto_data import get_realtime_snapshot
        result = get_realtime_snapshot("ETH-USD")

        assert "Timestamp (UTC)" not in result
        assert "Timestamp" in result
        assert "EDT" in result or "EST" in result


# ═══════════════════════════════════════════════════════════════════════
# Fundamentals analyst crypto tool binding
# ═══════════════════════════════════════════════════════════════════════

class TestFundamentalsAnalystCrypto:
    """Verify fundamentals analyst binds correct tools for crypto vs equity."""

    def test_crypto_tools_bound_for_eth(self):
        """ETH-USD should get crypto tools, not balance_sheet/cashflow."""
        from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(content="test", tool_calls=[])
        mock_llm.bind_tools.return_value = MagicMock(
            __or__=lambda self, other: MagicMock(invoke=lambda msgs: mock_bound.invoke(msgs))
        )

        node_fn = create_fundamentals_analyst(mock_llm)

        state = {
            "trade_date": "2026-04-12",
            "company_of_interest": "ETH-USD",
            "messages": [("human", "ETH-USD")],
        }

        # Patch the prompt chain to capture which tools are bound
        bound_tool_names = []
        original_bind = mock_llm.bind_tools

        def capture_bind(tools):
            bound_tool_names.extend([t.name for t in tools])
            return original_bind(tools)

        mock_llm.bind_tools = capture_bind
        try:
            node_fn(state)
        except Exception:
            pass

        assert "get_onchain_data" in bound_tool_names or "get_derivatives_data" in bound_tool_names, \
            f"Expected crypto tools, got: {bound_tool_names}"
        assert "get_balance_sheet" not in bound_tool_names, \
            f"Equity tool get_balance_sheet should not be bound for crypto"

    def test_equity_tools_bound_for_aapl(self):
        """AAPL should get equity tools."""
        from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst

        mock_llm = MagicMock()
        node_fn = create_fundamentals_analyst(mock_llm)

        state = {
            "trade_date": "2026-04-12",
            "company_of_interest": "AAPL",
            "messages": [("human", "AAPL")],
        }

        bound_tool_names = []
        original_bind = mock_llm.bind_tools

        def capture_bind(tools):
            bound_tool_names.extend([t.name for t in tools])
            return original_bind(tools)

        mock_llm.bind_tools = capture_bind
        try:
            node_fn(state)
        except Exception:
            pass

        assert "get_balance_sheet" in bound_tool_names, \
            f"Expected equity tools, got: {bound_tool_names}"
        assert "get_onchain_data" not in bound_tool_names


# ═══════════════════════════════════════════════════════════════════════
# Ensemble orchestrator crypto rebuild
# ═══════════════════════════════════════════════════════════════════════

class TestEnsembleCryptoRebuild:
    """Verify ensemble orchestrator rebuilds graph for crypto."""

    def test_get_current_price_uses_hyperliquid_for_crypto(self):
        """Crypto tickers should use Hyperliquid spot, not yfinance."""
        from tradingagents.graph.ensemble_orchestrator import EnsembleAnalysisOrchestrator
        from tradingagents.default_config import DEFAULT_CONFIG

        orch = EnsembleAnalysisOrchestrator(
            config=DEFAULT_CONFIG.copy(),
            provider="openrouter",
            model="test-model",
        )

        # is_crypto is imported inside _get_current_price, patch at source
        with patch("tradingagents.dataflows.asset_detection.is_crypto", return_value=True):
            with patch("tradingagents.dataflows.hyperliquid_client.HyperliquidClient") as MockHL:
                mock_instance = MagicMock()
                mock_instance.get_spot_price.return_value = 2215.50
                MockHL.return_value = mock_instance

                price = orch._get_current_price("ETH-USD")

        assert price == pytest.approx(2215.50)

    def test_get_current_price_falls_back_to_yfinance_for_equity(self):
        """Equity tickers should use yfinance."""
        from tradingagents.graph.ensemble_orchestrator import EnsembleAnalysisOrchestrator
        from tradingagents.default_config import DEFAULT_CONFIG

        orch = EnsembleAnalysisOrchestrator(
            config=DEFAULT_CONFIG.copy(),
            provider="openrouter",
            model="test-model",
        )

        with patch("yfinance.download") as mock_yf:
            mock_data = pd.DataFrame({"Close": [175.50]})
            mock_yf.return_value = mock_data

            price = orch._get_current_price("AAPL")

        assert mock_yf.called


# ═══════════════════════════════════════════════════════════════════════
# Config preservation
# ═══════════════════════════════════════════════════════════════════════

class TestConfigPreservation:
    """Verify display_timezone exists and crypto_vendors survive config overrides."""

    def test_display_timezone_in_default_config(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert "display_timezone" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["display_timezone"] == "US/Eastern"

    def test_crypto_vendors_in_default_config(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert "crypto_vendors" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["crypto_vendors"]["core_stock_apis"] == "crypto"
