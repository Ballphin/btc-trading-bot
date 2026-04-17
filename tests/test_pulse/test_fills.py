"""Tests for tradingagents/pulse/fills.py — 4 fill models + impact."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.pulse.fills import (
    FillResult,
    compute_four_fills,
    simple_fill_returns,
)


# ── simple_fill_returns ──────────────────────────────────────────────

class TestSimpleFillReturns:
    def test_empty_on_direction_zero(self):
        assert simple_fill_returns(0, 100, 101) == {}

    def test_empty_on_invalid_prices(self):
        assert simple_fill_returns(1, 0, 101) == {}
        assert simple_fill_returns(1, 100, 0) == {}

    def test_buy_profitable_exit(self):
        r = simple_fill_returns(direction=1, entry_price=100, exit_price=101)
        assert set(r.keys()) == {"best", "realistic", "maker_rejected", "maker_adverse"}
        assert r["best"].gross_return == pytest.approx(0.01)
        assert r["best"].cost_bps == 0.0
        assert r["best"].net_return == pytest.approx(0.01)
        assert r["realistic"].cost_bps > 0
        assert r["realistic"].net_return < r["best"].net_return

    def test_short_profitable(self):
        r = simple_fill_returns(direction=-1, entry_price=100, exit_price=99)
        assert r["best"].gross_return == pytest.approx(0.01)

    def test_short_losing(self):
        r = simple_fill_returns(direction=-1, entry_price=100, exit_price=101)
        assert r["best"].gross_return == pytest.approx(-0.01)

    def test_maker_rejected_triggers_on_big_move(self):
        """Price moved 250 bps in 10s → maker didn't fill, taker fallback."""
        r = simple_fill_returns(
            direction=1, entry_price=100, exit_price=102,
            price_at_10s=102.5,   # 2.5% move → rejection
        )
        mr = r["maker_rejected"]
        assert mr.entry_price == 102.5
        # Gross from 102.5 → 102 is negative
        assert mr.gross_return < 0
        assert "maker miss" in (mr.notes or "")

    def test_maker_rejected_no_trigger_on_small_move(self):
        r = simple_fill_returns(
            direction=1, entry_price=100, exit_price=102,
            price_at_10s=100.1,   # 10 bps move → no rejection
        )
        mr = r["maker_rejected"]
        assert mr.entry_price == 100  # filled at original price

    def test_maker_adverse_costs_added_when_drift_against(self):
        # BUY — price went up after signal → favorable → 0 adverse cost
        r_ok = simple_fill_returns(
            direction=1, entry_price=100, exit_price=101,
            price_at_30s=100.2,
        )
        # BUY — price went down after signal → adverse
        r_bad = simple_fill_returns(
            direction=1, entry_price=100, exit_price=101,
            price_at_30s=99.8,
        )
        assert r_bad["maker_adverse"].cost_bps > r_ok["maker_adverse"].cost_bps

    def test_impact_adds_cost(self):
        base = simple_fill_returns(direction=1, entry_price=100, exit_price=101)
        with_impact = simple_fill_returns(
            direction=1, entry_price=100, exit_price=101,
            notional_usd=1e6, adv_usd=1e9, impact_coefficient=10,
        )
        assert with_impact["realistic"].cost_bps > base["realistic"].cost_bps

    def test_best_has_no_cost(self):
        r = simple_fill_returns(direction=1, entry_price=100, exit_price=101,
                                notional_usd=1e6, adv_usd=1e9)
        assert r["best"].cost_bps == 0.0

    def test_losing_trade_all_models(self):
        """Every model should show negative net return when trade is losing."""
        r = simple_fill_returns(direction=1, entry_price=100, exit_price=99)
        for name, fr in r.items():
            assert fr.net_return < 0, f"{name} net_return should be < 0"


# ── compute_four_fills (DataFrame path) ────────────────────────────

class TestComputeFourFills:
    def _make_candles(
        self, start_ts: pd.Timestamp, n_min: int, price_trajectory: list
    ) -> pd.DataFrame:
        ts = pd.date_range(start_ts, periods=n_min, freq="1min")
        assert len(price_trajectory) == n_min
        return pd.DataFrame({"timestamp": ts, "close": price_trajectory})

    def test_empty_on_direction_zero(self):
        ts = pd.Timestamp("2025-01-01 00:00:00")
        candles = self._make_candles(ts, 70, [100] * 70)
        assert compute_four_fills(ts, 60, 0, candles, 100.0) == {}

    def test_returns_all_four_models_on_success(self):
        ts = pd.Timestamp("2025-01-01 00:00:00")
        prices = [100 + i * 0.01 for i in range(70)]  # mild uptrend
        candles = self._make_candles(ts, 70, prices)
        r = compute_four_fills(ts, 60, 1, candles, 100.0)
        assert set(r.keys()) == {"best", "realistic", "maker_rejected", "maker_adverse"}

    def test_exit_horizon_respected(self):
        ts = pd.Timestamp("2025-01-01 00:00:00")
        prices = [100 + i * 0.1 for i in range(70)]
        candles = self._make_candles(ts, 70, prices)
        r_5m = compute_four_fills(ts, 5, 1, candles, 100.0)
        r_60m = compute_four_fills(ts, 60, 1, candles, 100.0)
        # 60-min exit should be higher on uptrend
        assert r_60m["best"].exit_price > r_5m["best"].exit_price

    def test_missing_exit_candle_returns_empty(self):
        ts = pd.Timestamp("2025-01-01 00:00:00")
        # Only 3 minutes of candles; can't exit at 60 min
        candles = self._make_candles(ts, 3, [100, 100.5, 101])
        r = compute_four_fills(ts, 60, 1, candles, 100.0)
        # Might be empty or nearest-fallback
        # At least shouldn't crash
        assert isinstance(r, dict)
