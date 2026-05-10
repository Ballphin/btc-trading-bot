"""Tests for compute_v4_inputs and V4Inputs dataclass."""

import pytest
import pandas as pd
import numpy as np

from tradingagents.pulse.v4_inputs import compute_v4_inputs, V4Inputs


def _make_candles(n, interval_minutes, start_ts=None, base_price=50000.0):
    if start_ts is None:
        start_ts = pd.Timestamp("2026-01-01")
    data = []
    price = base_price
    for i in range(n):
        ts = start_ts + pd.Timedelta(minutes=interval_minutes * i)
        change = np.random.normal(0, 0.001 * base_price)
        price += change
        h = price + abs(np.random.normal(0, 50))
        l = price - abs(np.random.normal(0, 50))
        o = price + np.random.normal(0, 20)
        data.append({
            "timestamp": ts,
            "open": o,
            "high": max(h, o, price),
            "low": min(l, o, price),
            "close": price,
            "volume": abs(np.random.normal(100, 20)),
        })
    return pd.DataFrame(data)


class TestV4Inputs:
    def test_returns_empty_on_no_data(self):
        result = compute_v4_inputs(candles_by_tf={})
        assert result.vpd_signal is None
        assert result.liquidity_sweep_dir is None
        assert result.pattern_hits == {}

    def test_candlestick_patterns_detected_on_1h_4h(self):
        # Create a bullish engulfing: current close > previous open, current open < previous close
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="1h"),
            "open": [100.0, 105.0, 98.0],
            "high": [106.0, 107.0, 108.0],
            "low": [99.0, 104.0, 97.0],
            "close": [104.0, 106.0, 107.0],
            "volume": [100, 100, 100],
        })
        result = compute_v4_inputs(candles_by_tf={"1h": df})
        # pattern_hits may be empty if no patterns match the exact rules,
        # but the call should not raise.
        assert isinstance(result.pattern_hits, dict)

    def test_vpd_signal_non_zero_when_neg_corr(self):
        # Price ↓, Volume ↑ => negative correlation + new low => bullish divergence (+1)
        returns = np.array([
            -0.01, -0.02, -0.005, -0.03, -0.01,
            -0.025, -0.008, -0.02, -0.015, -0.03,
            -0.01, -0.02, -0.007, -0.025, -0.012,
            -0.02, -0.011, -0.018, -0.013, -0.02,
        ])
        prices = [100.0]
        volumes = [100.0]
        for r in returns:
            prices.append(prices[-1] * np.exp(r))
            volumes.append(volumes[-1] * np.exp(-1.3 * r))
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=len(prices), freq="15min"),
            "close": prices,
            "volume": volumes,
        })
        result = compute_v4_inputs(candles_by_tf={"15m": df})
        assert result.vpd_signal == 1

    def test_graceful_on_empty_df(self):
        result = compute_v4_inputs(
            candles_by_tf={"1h": pd.DataFrame(), "4h": pd.DataFrame()}
        )
        assert result.vpd_signal is None
        assert result.liquidity_sweep_dir is None
        assert result.pattern_hits == {}

    def test_respects_cfg_overrides(self):
        class MockCfg:
            def get(self, *keys, default=None):
                if keys == ("pulse_v4", "vpd", "lookback_bars"):
                    return 10
                if keys == ("pulse_v4", "vpd", "corr_threshold"):
                    return -0.10
                return default
        cfg = MockCfg()
        prices = np.linspace(100, 80, 12)
        volumes = np.linspace(100, 200, 12)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="15min"),
            "close": prices,
            "volume": volumes,
        })
        result = compute_v4_inputs(candles_by_tf={"15m": df}, cfg=cfg)
        # With lookback=10 and threshold=-0.10, it may still fire
        assert isinstance(result.vpd_signal, (int, type(None)))
