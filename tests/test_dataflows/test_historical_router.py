"""Unit tests for :mod:`tradingagents.dataflows.historical_router`.

These tests NEVER hit the network — all client dependencies are replaced
with ``unittest.mock`` fakes whose return values are constructed from
realistic Binance/HL row shapes. This keeps CI deterministic and fast.

Covered cases:
    * Era routing: pre-HL → Binance, post-HL → HL, straddle → stitched.
    * ``force_source`` bypasses the era check.
    * Ticker mapping: internal ``BTC-USD`` → Binance / HL / Coinbase forms.
    * Stitch overlap validation passes (p99 ≤ 0.5%) and fails (> 0.5%).
    * Coinbase cross-check raises on p99 divergence > 0.5%.
    * Funding router mirrors the OHLCV routing.
    * Binance kline interval validation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tradingagents.dataflows import historical_router as hr
from tradingagents.dataflows.historical_router import (
    DataSourceDriftError,
    FetchResult,
    HL_LAUNCH_UTC,
    StitchValidationError,
    _to_binance_symbol,
    _to_coinbase_product,
    _to_hl_coin,
    fetch_funding_historical,
    fetch_ohlcv_historical,
)


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_ohlcv(start: str, end: str, interval_hours: int = 1,
                base_close: float = 30000.0) -> pd.DataFrame:
    """Synthetic OHLCV series — constant close for easy stitch/validation math."""
    ts = pd.date_range(start, end, freq=f"{interval_hours}h", inclusive="left")
    return pd.DataFrame({
        "timestamp": ts,
        "open": base_close,
        "high": base_close,
        "low": base_close,
        "close": base_close,
        "volume": 1.0,
    })


@pytest.fixture
def hl_client_mock():
    m = MagicMock()
    m.get_ohlcv.return_value = _make_ohlcv("2023-02-15", "2023-02-20")
    m.get_funding_history.return_value = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-02-16 00:00", "2023-02-16 08:00"]),
        "funding_rate": [0.0001, 0.0002],
    })
    return m


@pytest.fixture
def binance_client_mock():
    m = MagicMock()
    m.get_klines.return_value = _make_ohlcv("2022-06-01", "2022-06-05").assign(
        quote_volume=30000.0, n_trades=100,
    )
    m.get_funding_rates.return_value = pd.DataFrame({
        "fundingTime": pd.to_datetime(["2022-06-01 00:00", "2022-06-01 08:00"]),
        "fundingRate": [0.00005, -0.00001],
    })
    return m


@pytest.fixture
def coinbase_client_mock():
    m = MagicMock()
    m.get_ohlcv.return_value = _make_ohlcv("2023-02-15", "2023-02-17")
    return m


# ── Ticker mapping ───────────────────────────────────────────────────

class TestTickerMapping:
    def test_btc_usd_to_binance(self):
        assert _to_binance_symbol("BTC-USD") == "BTCUSDT"

    def test_btc_usdt_passthrough(self):
        assert _to_binance_symbol("BTCUSDT") == "BTCUSDT"

    def test_lowercase_input_normalized(self):
        assert _to_binance_symbol("btc-usd") == "BTCUSDT"

    def test_btc_usd_to_hl(self):
        assert _to_hl_coin("BTC-USD") == "BTC"

    def test_eth_usd_to_hl(self):
        assert _to_hl_coin("ETH-USD") == "ETH"

    def test_binance_symbol_to_coinbase(self):
        assert _to_coinbase_product("BTCUSDT") == "BTC-USD"

    def test_already_coinbase_passthrough(self):
        assert _to_coinbase_product("BTC-USD") == "BTC-USD"


# ── Era routing ──────────────────────────────────────────────────────

class TestEraRouting:
    def test_pre_hl_routes_to_binance(self, hl_client_mock, binance_client_mock):
        res = fetch_ohlcv_historical(
            "BTC-USD", "1h", "2022-06-01", "2022-06-05",
            hl_client=hl_client_mock, binance_client=binance_client_mock,
        )
        assert res.source == "binance_futures"
        hl_client_mock.get_ohlcv.assert_not_called()
        binance_client_mock.get_klines.assert_called_once()

    def test_post_hl_routes_to_hyperliquid(self, hl_client_mock, binance_client_mock):
        res = fetch_ohlcv_historical(
            "BTC-USD", "1h", "2024-01-01", "2024-01-05",
            hl_client=hl_client_mock, binance_client=binance_client_mock,
        )
        assert res.source == "hyperliquid"
        binance_client_mock.get_klines.assert_not_called()
        hl_client_mock.get_ohlcv.assert_called_once()

    def test_exact_launch_date_routes_to_hl(self, hl_client_mock, binance_client_mock):
        """end == HL_LAUNCH is a boundary case; must NOT stitch."""
        res = fetch_ohlcv_historical(
            "BTC-USD", "1h", "2022-12-01", "2023-02-15",
            hl_client=hl_client_mock, binance_client=binance_client_mock,
        )
        # end <= HL_LAUNCH → Binance only
        assert res.source == "binance_futures"

    def test_force_source_overrides_era(self, hl_client_mock, binance_client_mock):
        # Post-HL dates but force Binance.
        res = fetch_ohlcv_historical(
            "BTC-USD", "1h", "2024-01-01", "2024-01-05",
            force_source="binance_futures",
            hl_client=hl_client_mock, binance_client=binance_client_mock,
        )
        assert res.source == "binance_futures"

    def test_invalid_interval_on_hl_raises(self, hl_client_mock, binance_client_mock):
        with pytest.raises(ValueError, match="interval"):
            fetch_ohlcv_historical(
                "BTC-USD", "7m", "2024-01-01", "2024-01-05",
                hl_client=hl_client_mock, binance_client=binance_client_mock,
            )

    def test_end_before_start_raises(self, hl_client_mock, binance_client_mock):
        with pytest.raises(ValueError, match="end"):
            fetch_ohlcv_historical(
                "BTC-USD", "1h", "2024-01-05", "2024-01-01",
                hl_client=hl_client_mock, binance_client=binance_client_mock,
            )


# ── Stitching ────────────────────────────────────────────────────────

class TestStitchOverlapValidation:
    def _mocks_with_matching_overlap(self, base_close=30000.0):
        hl = MagicMock()
        bn = MagicMock()

        # Stitch window fetch: Binance returns pre-launch part
        # HL returns post-launch part. Overlap validation fetch returns
        # identical closes → p99 divergence = 0.
        def hl_ohlcv(coin, interval, start=None, end=None):
            return _make_ohlcv(start or "2023-02-15", end or "2023-03-01", base_close=base_close)

        def bn_klines(symbol, interval, start=None, end=None):
            return _make_ohlcv(
                start or "2022-01-01", end or "2023-03-01", base_close=base_close,
            ).assign(quote_volume=base_close, n_trades=100)

        hl.get_ohlcv.side_effect = hl_ohlcv
        bn.get_klines.side_effect = bn_klines
        return hl, bn

    def test_stitched_straddles_launch(self):
        hl, bn = self._mocks_with_matching_overlap()
        res = fetch_ohlcv_historical(
            "BTC-USD", "1h", "2023-01-01", "2023-03-01",
            hl_client=hl, binance_client=bn,
        )
        assert res.source == "binance+hl_stitched"
        assert res.overlap_report is not None
        assert res.overlap_report["status"] == "ok"
        # Overlap report exposes stats for UI display.
        assert "p99" in res.overlap_report
        assert "n_bars" in res.overlap_report

    def test_stitch_fails_on_large_divergence(self):
        hl = MagicMock()
        bn = MagicMock()

        # HL reports prices 2% above Binance on every bar → p99 ≈ 2%.
        def hl_ohlcv(coin, interval, start=None, end=None):
            return _make_ohlcv(start or "2023-02-15", end or "2023-03-01", base_close=30600.0)

        def bn_klines(symbol, interval, start=None, end=None):
            return _make_ohlcv(
                start or "2022-01-01", end or "2023-03-01", base_close=30000.0,
            ).assign(quote_volume=30000.0, n_trades=100)

        hl.get_ohlcv.side_effect = hl_ohlcv
        bn.get_klines.side_effect = bn_klines
        with pytest.raises(StitchValidationError):
            fetch_ohlcv_historical(
                "BTC-USD", "1h", "2023-01-01", "2023-03-01",
                hl_client=hl, binance_client=bn,
            )

    def test_stitch_boundary_dedup(self):
        """Bars exactly at the boundary should not duplicate."""
        hl, bn = self._mocks_with_matching_overlap()
        res = fetch_ohlcv_historical(
            "BTC-USD", "1h", "2023-01-01", "2023-03-01",
            hl_client=hl, binance_client=bn,
        )
        assert res.df["timestamp"].is_unique


# ── Coinbase validation ──────────────────────────────────────────────

class TestCoinbaseValidation:
    def test_passes_when_prices_match(self, hl_client_mock, binance_client_mock,
                                       coinbase_client_mock):
        # Both sources return close=30000 → zero divergence.
        fetch_ohlcv_historical(
            "BTC-USD", "1h", "2023-02-15", "2023-02-20",
            validate_with_coinbase=True,
            hl_client=hl_client_mock, binance_client=binance_client_mock,
            coinbase_client=coinbase_client_mock,
        )  # no raise = pass

    def test_raises_on_large_divergence(self, hl_client_mock, binance_client_mock):
        cb = MagicMock()
        # Coinbase reports 3% higher than source → guaranteed > 0.5%.
        cb.get_ohlcv.return_value = _make_ohlcv(
            "2023-02-15", "2023-02-20", base_close=30900.0,
        )
        with pytest.raises(DataSourceDriftError, match="Coinbase validation"):
            fetch_ohlcv_historical(
                "BTC-USD", "1h", "2023-02-15", "2023-02-20",
                validate_with_coinbase=True,
                hl_client=hl_client_mock, binance_client=binance_client_mock,
                coinbase_client=cb,
            )

    def test_skipped_for_unsupported_interval(self, hl_client_mock, binance_client_mock):
        cb = MagicMock()
        cb.get_ohlcv.side_effect = AssertionError("should not be called")
        # 4h is not in _COINBASE_GRANULARITY_MAP → validation skipped.
        fetch_ohlcv_historical(
            "BTC-USD", "4h", "2024-01-01", "2024-01-05",
            validate_with_coinbase=True,
            hl_client=hl_client_mock, binance_client=binance_client_mock,
            coinbase_client=cb,
        )
        cb.get_ohlcv.assert_not_called()


# ── Funding router ───────────────────────────────────────────────────

class TestFundingRouting:
    def test_pre_hl_uses_binance_funding(self, hl_client_mock, binance_client_mock):
        df, src = fetch_funding_historical(
            "BTC-USD", "2022-06-01", "2022-06-05",
            hl_client=hl_client_mock, binance_client=binance_client_mock,
        )
        assert src == "binance_futures"
        assert not df.empty
        # Normalized column names
        assert list(df.columns) == ["timestamp", "funding_rate"]

    def test_post_hl_uses_hyperliquid_funding(self, hl_client_mock, binance_client_mock):
        df, src = fetch_funding_historical(
            "BTC-USD", "2024-01-01", "2024-01-05",
            hl_client=hl_client_mock, binance_client=binance_client_mock,
        )
        assert src == "hyperliquid"
        assert not df.empty
        assert list(df.columns) == ["timestamp", "funding_rate"]

    def test_funding_straddles_launch(self):
        hl = MagicMock()
        bn = MagicMock()
        hl.get_funding_history.return_value = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-02-16 00:00"]),
            "funding_rate": [0.0001],
        })
        bn.get_funding_rates.return_value = pd.DataFrame({
            "fundingTime": pd.to_datetime(["2022-12-01 00:00"]),
            "fundingRate": [0.00005],
        })
        df, src = fetch_funding_historical(
            "BTC-USD", "2022-12-01", "2023-03-01",
            hl_client=hl, binance_client=bn,
        )
        assert src == "binance+hl_stitched"
        # Both entries present, one from each source.
        assert len(df) == 2


# ── BinanceClient.get_klines directly ────────────────────────────────

class TestBinanceKlines:
    def test_interval_validation(self):
        from tradingagents.dataflows.binance_client import BinanceClient
        bn = BinanceClient()
        with pytest.raises(ValueError, match="Unsupported interval"):
            bn.get_klines("BTCUSDT", interval="3s")

    def test_empty_response_returns_empty_df(self, monkeypatch):
        from tradingagents.dataflows.binance_client import BinanceClient
        bn = BinanceClient()

        monkeypatch.setattr(bn, "_request", lambda *a, **k: [])
        df = bn.get_klines("BTCUSDT", "1h", start="2022-01-01", end="2022-01-02")
        assert df.empty
        assert list(df.columns) == [
            "timestamp", "open", "high", "low", "close",
            "volume", "quote_volume", "n_trades",
        ]

    def test_parses_binance_row_layout(self, monkeypatch):
        from tradingagents.dataflows.binance_client import BinanceClient
        bn = BinanceClient()

        # Two synthetic rows with Binance's 12-column layout.
        fake = [
            [1640995200000, "47000.0", "47500.0", "46800.0", "47100.0",
             "1234.5", 1640998800000, "58023000.0", 9876,
             "600.0", "28200000.0", "0"],
            [1640998800000, "47100.0", "47200.0", "46900.0", "47050.0",
             "500.0", 1641002400000, "23525000.0", 4321,
             "250.0", "11762500.0", "0"],
        ]
        call_count = {"n": 0}
        def fake_request(*a, **k):
            call_count["n"] += 1
            # Return data once, empty thereafter so the loop terminates.
            return fake if call_count["n"] == 1 else []

        monkeypatch.setattr(bn, "_request", fake_request)
        df = bn.get_klines("BTCUSDT", "1h", start="2022-01-01", end="2022-01-02")
        assert len(df) == 2
        assert float(df.iloc[0]["close"]) == 47100.0
        assert int(df.iloc[0]["n_trades"]) == 9876
        assert df["timestamp"].is_monotonic_increasing
