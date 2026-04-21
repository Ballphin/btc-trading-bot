"""R.3 — Exhaustive ``_resolve_intrabar_fill`` matrix (BLOCKER #1 + #3).

Every branch of the fill resolver has a named test. The matrix is
{long, short} × {sl-only, tp-only, both-touch, neither} plus the
flash-crash slippage-cap and degenerate-ATR edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.pulse_verifier import _resolve_intrabar_fill


@dataclass
class _Bar:
    high: float
    low: float
    close: float = 0.0


# ── long-side matrix ────────────────────────────────────────────────

def test_sl_only_long():
    # Low wicks SL but high never reaches TP.
    bar = _Bar(high=101.0, low=97.5)
    fr = _resolve_intrabar_fill(bar, entry_price=100.0, sl=98.0, tp=104.0,
                                side="BUY", atr_5m=1.0)
    assert fr.exit_type == "sl_hit"
    # With ATR=1 the slip cap is 2 → worst allowed fill is 98.0 (=sl).
    assert fr.fill_price == 98.0
    assert fr.clamped is False


def test_tp_only_long():
    bar = _Bar(high=104.5, low=99.0)
    fr = _resolve_intrabar_fill(bar, entry_price=100.0, sl=98.0, tp=104.0,
                                side="BUY", atr_5m=1.0)
    assert fr.exit_type == "tp_hit"
    # TP 104 exceeds slip cap 100+2=102 → fill clamped down.
    assert fr.fill_price == 102.0
    assert fr.clamped is True


def test_both_touch_long_sl_wins():
    bar = _Bar(high=104.5, low=97.5)
    fr = _resolve_intrabar_fill(bar, entry_price=100.0, sl=98.0, tp=104.0,
                                side="BUY", atr_5m=1.0)
    assert fr.exit_type == "sl_hit", "SL wins ties on same-bar straddle"
    assert fr.fill_price == 98.0


def test_neither_touch_long():
    bar = _Bar(high=101.0, low=99.0)
    fr = _resolve_intrabar_fill(bar, entry_price=100.0, sl=98.0, tp=104.0,
                                side="BUY", atr_5m=1.0)
    assert fr.exit_type is None
    assert fr.fill_price is None


def test_flash_crash_clamp_long():
    # Low prints far below entry−2×ATR — stop fill capped at entry − 2×ATR.
    bar = _Bar(high=101.0, low=50.0)
    fr = _resolve_intrabar_fill(bar, entry_price=100.0, sl=98.0, tp=104.0,
                                side="BUY", atr_5m=1.0)
    # slip cap = 2.0 → worst allowed fill = 98.0 (= sl), so no clamp here
    # because sl is not below the cap. Verify SL wins.
    assert fr.exit_type == "sl_hit"
    assert fr.fill_price == 98.0

    # Push SL deeper to force the clamp.
    fr2 = _resolve_intrabar_fill(bar, entry_price=100.0, sl=90.0, tp=104.0,
                                 side="BUY", atr_5m=1.0)
    assert fr2.exit_type == "sl_hit"
    # Worst allowed = 100 − 2 = 98; fill = max(90, 98) = 98.
    assert fr2.fill_price == 98.0
    assert fr2.clamped is True


# ── short-side matrix ────────────────────────────────────────────────

def test_sl_only_short():
    # Short SL above entry, TP below. High wicks SL but low doesn't
    # reach TP.
    bar = _Bar(high=102.5, low=99.5)
    fr = _resolve_intrabar_fill(bar, entry_price=100.0, sl=102.0, tp=96.0,
                                side="SHORT", atr_5m=1.0)
    assert fr.exit_type == "sl_hit"
    # Worst allowed = entry + 2×ATR = 102; fill = min(sl, 102) = 102.
    assert fr.fill_price == 102.0


def test_tp_only_short():
    bar = _Bar(high=100.5, low=95.5)
    fr = _resolve_intrabar_fill(bar, entry_price=100.0, sl=102.0, tp=96.0,
                                side="SHORT", atr_5m=1.0)
    assert fr.exit_type == "tp_hit"
    # Best allowed = entry − 2×ATR = 98; TP=96 → clamp to 98.
    assert fr.fill_price == 98.0
    assert fr.clamped is True


def test_both_touch_short_sl_wins():
    bar = _Bar(high=102.5, low=95.5)
    fr = _resolve_intrabar_fill(bar, entry_price=100.0, sl=102.0, tp=96.0,
                                side="SHORT", atr_5m=1.0)
    assert fr.exit_type == "sl_hit"
    assert fr.fill_price == 102.0


def test_neither_touch_short():
    bar = _Bar(high=101.0, low=99.0)
    fr = _resolve_intrabar_fill(bar, entry_price=100.0, sl=102.0, tp=96.0,
                                side="SHORT", atr_5m=1.0)
    assert fr.exit_type is None
    assert fr.fill_price is None


def test_flash_crash_clamp_short():
    bar = _Bar(high=150.0, low=99.0)  # massive wick above
    # Push SL well beyond the clamp to force clamping.
    fr = _resolve_intrabar_fill(bar, entry_price=100.0, sl=110.0, tp=96.0,
                                side="SHORT", atr_5m=1.0)
    # Worst allowed = 100+2 = 102; fill = min(110, 102) = 102.
    assert fr.exit_type == "sl_hit"
    assert fr.fill_price == 102.0
    assert fr.clamped is True


# ── degenerate ATR ──────────────────────────────────────────────────

def test_degenerate_atr_uses_raw_fill_without_clamp():
    """ATR=0 or None must not crash. Raw SL/TP levels are used and
    ``clamped`` is False so the outcome can flag the missing clamp."""
    bar = _Bar(high=101.0, low=50.0)
    for atr in (0.0, None):
        fr = _resolve_intrabar_fill(bar, entry_price=100.0, sl=98.0, tp=104.0,
                                    side="BUY", atr_5m=atr)
        assert fr.exit_type == "sl_hit"
        assert fr.fill_price == 98.0
        assert fr.clamped is False
