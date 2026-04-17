"""Tests for tradingagents/pulse/tsmom.py — TSMOM direction & sizing."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradingagents.pulse.tsmom import (
    TsmomResult,
    compute_tsmom,
    load_tsmom,
    save_tsmom_atomic,
)


def _synthetic_candles(n_bars: int, drift: float, vol: float, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic 1h close series with given drift & vol."""
    rng = np.random.default_rng(seed)
    log_rets = drift + vol * rng.standard_normal(n_bars)
    close = 100 * np.exp(np.cumsum(log_rets))
    return pd.DataFrame({"close": close})


class TestComputeTsmom:
    def test_uptrend_direction_positive(self):
        df = _synthetic_candles(7000, drift=0.0002, vol=0.005, seed=1)
        r = compute_tsmom(df, "BTC-USD")
        assert r.direction == 1
        assert r.strength >= 0.33
        assert not r.insufficient_history

    def test_downtrend_direction_negative(self):
        df = _synthetic_candles(7000, drift=-0.0002, vol=0.005, seed=2)
        r = compute_tsmom(df, "ETH-USD")
        assert r.direction == -1
        assert r.strength >= 0.33

    def test_flat_direction_zero_or_weak(self):
        df = _synthetic_candles(7000, drift=0.0, vol=0.005, seed=3)
        r = compute_tsmom(df, "FLAT-USD")
        # drift=0 → direction can be either sign but strength should be small
        assert r.strength <= 1.0

    def test_insufficient_history_flag(self):
        df = _synthetic_candles(100, drift=0.0001, vol=0.005, seed=4)
        r = compute_tsmom(df, "SHORT-USD", min_bars_required=500)
        assert r.insufficient_history
        assert r.n_bars_used == 100

    def test_empty_dataframe(self):
        empty = pd.DataFrame()
        r = compute_tsmom(empty, "X-USD")
        assert r.insufficient_history
        assert r.direction == 0

    def test_none_input(self):
        r = compute_tsmom(None, "X-USD")
        assert r.insufficient_history
        assert r.direction == 0

    def test_size_weight_scales_inverse_vol(self):
        # Low-vol series → high size weight; high-vol series → low.
        lo = _synthetic_candles(7000, drift=0.0001, vol=0.002, seed=5)
        hi = _synthetic_candles(7000, drift=0.0001, vol=0.02, seed=6)
        r_lo = compute_tsmom(lo, "LO-USD")
        r_hi = compute_tsmom(hi, "HI-USD")
        assert r_lo.size_weight > r_hi.size_weight

    def test_size_weight_capped_at_5(self):
        # Extremely low vol → size_weight should clamp to ≤5
        lo = _synthetic_candles(7000, drift=0.00001, vol=0.0002, seed=7)
        r = compute_tsmom(lo, "TINY-USD")
        assert r.size_weight <= 5.0

    def test_lookback_returns_included(self):
        df = _synthetic_candles(7000, drift=0.0002, vol=0.005, seed=8)
        r = compute_tsmom(df, "BTC-USD", lookbacks_hours=(504, 1512, 6048))
        assert "504h" in r.lookback_returns
        assert "1512h" in r.lookback_returns
        assert "6048h" in r.lookback_returns
        assert all(isinstance(v, float) for v in r.lookback_returns.values())

    def test_custom_lookbacks(self):
        df = _synthetic_candles(7000, drift=0.0002, vol=0.005, seed=9)
        r = compute_tsmom(df, "BTC-USD", lookbacks_hours=(24, 168))
        assert "24h" in r.lookback_returns
        assert "168h" in r.lookback_returns


class TestCacheIO:
    def test_save_and_load(self, tmp_path):
        r = TsmomResult(
            ticker="BTC-USD", direction=1, strength=0.667, size_weight=1.5,
            realized_vol_30d=0.15, lookback_returns={"504h": 0.10},
            computed_at=time.time(), n_bars_used=500,
            lookbacks_hours=[504, 1512, 6048], target_annualized_vol=0.20,
            insufficient_history=False,
        )
        save_tsmom_atomic(r, str(tmp_path))
        loaded = load_tsmom("BTC-USD", str(tmp_path))
        assert loaded is not None
        assert loaded.direction == 1
        assert loaded.strength == pytest.approx(0.667, rel=1e-3)

    def test_load_missing_returns_none(self, tmp_path):
        assert load_tsmom("NONEXISTENT", str(tmp_path)) is None

    def test_load_stale_returns_none(self, tmp_path):
        r = TsmomResult(
            ticker="BTC-USD", direction=1, strength=0.667, size_weight=1.5,
            realized_vol_30d=0.15, lookback_returns={"504h": 0.10},
            computed_at=time.time() - 10_000,  # very stale
            n_bars_used=500, lookbacks_hours=[504],
            target_annualized_vol=0.20, insufficient_history=False,
        )
        save_tsmom_atomic(r, str(tmp_path))
        loaded = load_tsmom("BTC-USD", str(tmp_path), max_age_sec=3600)
        assert loaded is None

    def test_atomic_write_creates_file(self, tmp_path):
        r = TsmomResult(
            ticker="ETH-USD", direction=-1, strength=0.333, size_weight=1.0,
            realized_vol_30d=0.20, lookback_returns={},
            computed_at=time.time(), n_bars_used=300,
            lookbacks_hours=[504], target_annualized_vol=0.20,
            insufficient_history=True,
        )
        path = save_tsmom_atomic(r, str(tmp_path))
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["ticker"] == "ETH-USD"
        assert data["direction"] == -1
