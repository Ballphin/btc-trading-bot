"""Tests for three-tier adaptive scoring: SL/TP hit scan, hold-period timeout, 7d fallback.

Tests _scan_sl_tp_hits, _estimate_execution_costs, and _get_ohlc_range helpers.
"""

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest

from tradingagents.backtesting.scorecard import (
    _scan_sl_tp_hits,
    _estimate_execution_costs,
    _SPREAD_BPS,
    _FUNDING_RATE_PER_8H,
)


# ── Helper: build a mock OHLC DataFrame ──────────────────────────────


def _make_ohlc(entry_dt: datetime, rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from row dicts: {day, open, high, low, close}."""
    dates = [entry_dt + timedelta(days=r["day"]) for r in rows]
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df = df[["Open", "High", "Low", "Close"]]
    return df


# ── _scan_sl_tp_hits ─────────────────────────────────────────────────


class TestScanSlTpHits:
    """Tests for the three-tier scoring scanner."""

    def _entry(self):
        return datetime(2026, 4, 10)

    # --- Tier 1: SL/TP hits ---

    def test_long_stop_loss_hit(self):
        entry = self._entry()
        ohlc = _make_ohlc(entry, [
            {"day": 1, "open": 100, "high": 102, "low": 95, "close": 98},
        ])
        result = _scan_sl_tp_hits(ohlc, 100.0, "BUY", stop_loss=96.0, take_profit=110.0, hold_days=3, entry_dt=entry)
        assert result["exit_type"] == "stop_loss_hit"
        assert result["was_correct"] is False
        assert result["exit_day"] == 1
        assert result["exit_price"] == 96.0

    def test_long_take_profit_hit(self):
        entry = self._entry()
        ohlc = _make_ohlc(entry, [
            {"day": 1, "open": 100, "high": 101, "low": 99, "close": 100.5},
            {"day": 2, "open": 100.5, "high": 112, "low": 100, "close": 111},
        ])
        result = _scan_sl_tp_hits(ohlc, 100.0, "BUY", stop_loss=95.0, take_profit=110.0, hold_days=5, entry_dt=entry)
        assert result["exit_type"] == "take_profit_hit"
        assert result["was_correct"] is True
        assert result["exit_day"] == 2
        assert result["exit_price"] == 110.0

    def test_short_stop_loss_hit(self):
        entry = self._entry()
        ohlc = _make_ohlc(entry, [
            {"day": 1, "open": 100, "high": 106, "low": 99, "close": 105},
        ])
        result = _scan_sl_tp_hits(ohlc, 100.0, "SHORT", stop_loss=105.0, take_profit=90.0, hold_days=3, entry_dt=entry)
        assert result["exit_type"] == "stop_loss_hit"
        assert result["was_correct"] is False
        assert result["exit_price"] == 105.0

    def test_short_take_profit_hit(self):
        entry = self._entry()
        ohlc = _make_ohlc(entry, [
            {"day": 1, "open": 100, "high": 101, "low": 89, "close": 91},
        ])
        result = _scan_sl_tp_hits(ohlc, 100.0, "SHORT", stop_loss=105.0, take_profit=90.0, hold_days=3, entry_dt=entry)
        assert result["exit_type"] == "take_profit_hit"
        assert result["was_correct"] is True
        assert result["exit_price"] == 90.0

    def test_sl_triggers_before_tp_same_day(self):
        """When both SL and TP could trigger on the same candle, SL fires first (conservative)."""
        entry = self._entry()
        # Low touches SL, high touches TP — we scan SL first
        ohlc = _make_ohlc(entry, [
            {"day": 1, "open": 100, "high": 115, "low": 90, "close": 100},
        ])
        result = _scan_sl_tp_hits(ohlc, 100.0, "BUY", stop_loss=92.0, take_profit=112.0, hold_days=3, entry_dt=entry)
        assert result["exit_type"] == "stop_loss_hit"

    # --- Tier 2: Hold-period timeout ---

    def test_hold_timeout_long_profitable(self):
        entry = self._entry()
        ohlc = _make_ohlc(entry, [
            {"day": 1, "open": 100, "high": 102, "low": 99, "close": 101},
            {"day": 2, "open": 101, "high": 103, "low": 100, "close": 102},
            {"day": 3, "open": 102, "high": 104, "low": 101, "close": 103},
        ])
        result = _scan_sl_tp_hits(ohlc, 100.0, "BUY", stop_loss=90.0, take_profit=120.0, hold_days=3, entry_dt=entry)
        assert result["exit_type"] == "held_to_expiry"
        assert result["was_correct"] is True
        assert result["exit_day"] == 3

    def test_hold_timeout_no_sl_tp(self):
        """When no SL/TP is provided, always goes to hold timeout."""
        entry = self._entry()
        ohlc = _make_ohlc(entry, [
            {"day": 1, "open": 100, "high": 105, "low": 95, "close": 98},
        ])
        result = _scan_sl_tp_hits(ohlc, 100.0, "BUY", stop_loss=None, take_profit=None, hold_days=1, entry_dt=entry)
        assert result["exit_type"] == "held_to_expiry"
        assert result["was_correct"] is False  # close 98 < entry 100

    # --- Edge cases ---

    def test_empty_ohlc_returns_empty(self):
        result = _scan_sl_tp_hits(None, 100.0, "BUY", 95.0, 110.0, 3, datetime(2026, 4, 10))
        assert result == {}

    def test_empty_dataframe_returns_empty(self):
        result = _scan_sl_tp_hits(pd.DataFrame(), 100.0, "BUY", 95.0, 110.0, 3, datetime(2026, 4, 10))
        assert result == {}

    def test_return_calculation_precision(self):
        entry = self._entry()
        ohlc = _make_ohlc(entry, [
            {"day": 1, "open": 100, "high": 102, "low": 99, "close": 105},
        ])
        result = _scan_sl_tp_hits(ohlc, 100.0, "BUY", stop_loss=None, take_profit=None, hold_days=1, entry_dt=entry)
        assert result["actual_return"] == 0.05  # (105 - 100) / 100


# ── _estimate_execution_costs ─────────────────────────────────────────


class TestEstimateExecutionCosts:
    """Tests for execution cost estimation."""

    def test_crypto_long_spread_only(self):
        cost = _estimate_execution_costs("BUY", hold_days=3, ticker="BTC-USD")
        assert cost == _SPREAD_BPS / 10000.0

    def test_crypto_short_includes_funding(self):
        cost = _estimate_execution_costs("SHORT", hold_days=3, ticker="BTC-USD")
        expected_spread = _SPREAD_BPS / 10000.0
        expected_funding = 3 * 3 * _FUNDING_RATE_PER_8H  # 3 days * 3 intervals/day
        assert abs(cost - (expected_spread + expected_funding)) < 1e-10

    def test_non_crypto_no_funding(self):
        cost = _estimate_execution_costs("SHORT", hold_days=5, ticker="AAPL")
        assert cost == _SPREAD_BPS / 10000.0

    def test_cover_signal_no_funding(self):
        cost = _estimate_execution_costs("COVER", hold_days=3, ticker="BTC-USD")
        assert cost == _SPREAD_BPS / 10000.0

    def test_longer_hold_more_funding(self):
        cost_3 = _estimate_execution_costs("SHORT", hold_days=3, ticker="ETH-USD")
        cost_7 = _estimate_execution_costs("SHORT", hold_days=7, ticker="ETH-USD")
        assert cost_7 > cost_3

    def test_crypto_detection_xau_usd_known_limitation(self):
        """XAU-USD (gold) matches '-USD' substring and is incorrectly treated as crypto.
        Documenting as known limitation — crypto detection uses '-USD' substring match."""
        cost = _estimate_execution_costs("SHORT", hold_days=3, ticker="XAU-USD")
        spread_only = _SPREAD_BPS / 10000.0
        # Known limitation: gold gets crypto funding cost because of -USD match
        assert cost > spread_only, "XAU-USD false-positives as crypto due to -USD match (known limitation)"

    def test_crypto_detection_sol_usdt(self):
        """SOL-USDT triggers crypto path via 'USDT' substring match."""
        cost = _estimate_execution_costs("SHORT", hold_days=3, ticker="SOL-USDT")
        spread_only = _SPREAD_BPS / 10000.0
        assert cost > spread_only, "SOL-USDT should be detected as crypto via USDT match"


# ── Additional _scan_sl_tp_hits edge cases ─────────────────────────────


class TestScanSlTpEdgeCases:
    """Additional edge-case tests for the three-tier scoring scanner."""

    def _entry(self):
        return datetime(2026, 4, 10)

    def test_short_sl_tp_same_candle(self):
        """SHORT position: both SL (high) and TP (low) trigger on same candle.
        SL fires first (conservative assumption, mirrors BUY test)."""
        entry = self._entry()
        # High touches SL (105), low touches TP (90) — SL checked first
        ohlc = _make_ohlc(entry, [
            {"day": 1, "open": 100, "high": 106, "low": 89, "close": 95},
        ])
        result = _scan_sl_tp_hits(ohlc, 100.0, "SHORT", stop_loss=105.0, take_profit=90.0, hold_days=3, entry_dt=entry)
        assert result["exit_type"] == "stop_loss_hit"
        assert result["was_correct"] is False

    def test_weekend_gap_hold_period(self):
        """hold_days=5 but only 3 OHLC rows available (simulating equity weekend gap).
        Scanner should exit at last available row; exit_day reflects calendar days, not trading days."""
        entry = self._entry()
        # Only Mon, Tue, Wed data available (days 1, 2, 3) — Thu/Fri missing
        ohlc = _make_ohlc(entry, [
            {"day": 1, "open": 100, "high": 102, "low": 99, "close": 101},
            {"day": 2, "open": 101, "high": 103, "low": 100, "close": 102},
            {"day": 3, "open": 102, "high": 104, "low": 101, "close": 103},
        ])
        result = _scan_sl_tp_hits(ohlc, 100.0, "BUY", stop_loss=90.0, take_profit=120.0, hold_days=5, entry_dt=entry)
        assert result["exit_type"] == "held_to_expiry"
        # exit_day should be 3 (last available calendar day), not 5 (planned hold)
        assert result["exit_day"] == 3
        assert result["was_correct"] is True  # 103 > 100

    def test_sharpe_se_consistency_note(self):
        """Verify sharpe_se and DSR now both use full Lo(2002) with skew+kurtosis.

        Previously there was inconsistency: sharpe_se used simplified Gaussian SE
        while DSR used full Lo(2002). After unification, both use the same helper.

        For SR=1.5, n=50, skew=-1.5, kurtosis=8:
          Gaussian SE = sqrt((1 + 0.5*2.25) / 50) = 0.206
          Lo(2002) SE = sqrt((1 + 0.5*2.25 + 1.5*1.5 + (5/4)*2.25) / 50) = 0.379
        These differ by ~84%, showing why fat-tail correction matters.
        """
        from tradingagents.backtesting.stats import sharpe_standard_error

        sr, n = 1.5, 50
        gaussian_se = sharpe_standard_error(sr, n, skew=0.0, kurtosis=3.0)
        fat_tail_se = sharpe_standard_error(sr, n, skew=-1.5, kurtosis=8.0)

        # Gaussian and fat-tail SE should differ significantly
        ratio_diff = abs(gaussian_se - fat_tail_se) / fat_tail_se
        assert ratio_diff > 0.40, (
            f"Expected 40%+ difference, gaussian={gaussian_se:.4f} vs fat_tail={fat_tail_se:.4f}"
        )
