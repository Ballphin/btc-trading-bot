"""Tests for tradingagents/pulse/regime.py — regime detection."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.pulse.regime import RegimeResult, detect_regime


def _ar1_closes(n: int, phi: float, drift: float, sigma: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = np.zeros(n)
    for i in range(1, n):
        rets[i] = phi * rets[i - 1] + drift + rng.normal(0, sigma)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"close": close})


class TestDetectRegime:
    def test_trend_ar1_positive_rho(self):
        """AR(1) with phi≈0.08 — the target for trend regime."""
        df = _ar1_closes(7000, phi=0.08, drift=0.001, sigma=0.004, seed=10)
        r = detect_regime(df)
        assert r.mode in ("trend", "high_vol_trend")
        assert r.returns_acf1 > 0.03

    def test_chop_ar1_negative_rho(self):
        df = _ar1_closes(7000, phi=-0.05, drift=0.0, sigma=0.004, seed=11)
        r = detect_regime(df)
        assert r.mode in ("chop", "mixed")
        assert r.returns_acf1 < 0.01

    def test_mixed_on_iid(self):
        rng = np.random.default_rng(12)
        rets = 0.0005 + rng.normal(0, 0.004, 7000)
        close = 100 * np.exp(np.cumsum(rets))
        r = detect_regime(pd.DataFrame({"close": close}))
        # IID with small drift → acf1 ≈ 0 → mixed
        assert r.mode == "mixed"

    def test_insufficient_history_flagged(self):
        df = _ar1_closes(100, phi=0.08, drift=0.001, sigma=0.004, seed=13)
        r = detect_regime(df, min_bars_required=500)
        assert r.insufficient_history
        assert r.mode == "mixed"

    def test_empty_returns_mixed(self):
        r = detect_regime(pd.DataFrame())
        assert r.mode == "mixed"
        assert r.insufficient_history

    def test_directional_bias_from_4h(self):
        """EMA50 > EMA200 on 4h → bias = +1."""
        df_1h = _ar1_closes(3000, phi=0.02, drift=0.0002, sigma=0.004, seed=14)
        # 4h closes forming a clear uptrend: 400 bars, linear+noise
        rng = np.random.default_rng(15)
        closes_4h = 100 * np.exp(np.cumsum(0.002 + rng.normal(0, 0.004, 400)))
        df_4h = pd.DataFrame({"close": closes_4h})
        r = detect_regime(df_1h, candles_4h=df_4h)
        assert r.directional_bias == 1

    def test_directional_bias_downtrend(self):
        df_1h = _ar1_closes(3000, phi=0.02, drift=-0.0002, sigma=0.004, seed=16)
        rng = np.random.default_rng(17)
        closes_4h = 100 * np.exp(np.cumsum(-0.002 + rng.normal(0, 0.004, 400)))
        df_4h = pd.DataFrame({"close": closes_4h})
        r = detect_regime(df_1h, candles_4h=df_4h)
        assert r.directional_bias == -1

    def test_percentile_clamp_caps_extreme_vol_z(self):
        """When recent vol is an extreme outlier, clamped z should stay finite."""
        rng = np.random.default_rng(18)
        n = 3000
        rets = np.zeros(n)
        for i in range(1, n - 50):
            rets[i] = 0.05 * rets[i - 1] + rng.normal(0, 0.003)
        # Giant vol spike in last 50 bars
        rets[-50:] = rng.normal(0, 0.05, 50)
        close = 100 * np.exp(np.cumsum(rets))
        r = detect_regime(pd.DataFrame({"close": close}))
        # vol_z unclipped can be massive; clipped must stay bounded
        assert abs(r.vol_z_clipped) < 20  # sanity bound (normally ≤ 3)

    def test_high_vol_trend_classification(self):
        """AR1 positive phi + vol spike recently → high_vol_trend."""
        rng = np.random.default_rng(19)
        n = 3000
        rets = np.zeros(n)
        for i in range(1, n):
            vol = 0.003 if i < n - 300 else 0.012
            rets[i] = 0.08 * rets[i - 1] + 0.002 + rng.normal(0, vol)
        close = 100 * np.exp(np.cumsum(rets))
        r = detect_regime(pd.DataFrame({"close": close}))
        assert r.mode in ("high_vol_trend", "trend")
        assert r.vol_z_clipped > 0.5

    def test_realized_vol_annualized(self):
        """Realized vol must be positive and roughly in annualized range."""
        df = _ar1_closes(7000, phi=0.02, drift=0.0001, sigma=0.004, seed=20)
        r = detect_regime(df)
        # sigma=0.004 per 1h bar ≈ sqrt(24*365)*0.004 ≈ 37% ann
        assert 0.15 < r.realized_vol_annualized < 0.80
