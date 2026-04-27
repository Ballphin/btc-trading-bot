"""Tests for the 1m liquidity-sweep detector."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.pulse.liquidity_sweep import detect_liquidity_sweep


def _mk_candles(n: int, base: float = 100.0, vol: float = 100.0) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        rows.append({"open": base, "high": base + 0.5, "low": base - 0.5,
                     "close": base, "volume": vol})
    return pd.DataFrame(rows)


def test_clean_long_trap_returns_short_signal():
    # 60 bars of quiet range with prior_high ~= 100.5, then breach up then reclaim down.
    df = _mk_candles(60)
    # breach bar
    df = pd.concat([df, pd.DataFrame([{"open": 100, "high": 102, "low": 99,
                                        "close": 101.5, "volume": 100}])], ignore_index=True)
    # reclaim bar: close below 100.5 with 3× volume
    df = pd.concat([df, pd.DataFrame([{"open": 101, "high": 101.5, "low": 99,
                                        "close": 100.0, "volume": 300}])], ignore_index=True)
    # pad for min bar requirement
    tail = _mk_candles(8)
    df = pd.concat([df, tail], ignore_index=True)
    res = detect_liquidity_sweep(df, extreme_lookback_bars=60, reclaim_within_bars=10,
                                 reclaim_volume_mul=2.0)
    assert res.direction == -1


def test_no_reclaim_returns_zero():
    df = _mk_candles(60)
    df = pd.concat([df, pd.DataFrame([{"open": 100, "high": 103, "low": 100,
                                        "close": 102.5, "volume": 300}])], ignore_index=True)
    # tail stays above prior_high — no reclaim
    for _ in range(10):
        df = pd.concat([df, pd.DataFrame([{"open": 102, "high": 103, "low": 101.5,
                                            "close": 102.8, "volume": 100}])], ignore_index=True)
    res = detect_liquidity_sweep(df)
    assert res.direction == 0


def test_low_volume_reclaim_rejected():
    df = _mk_candles(60)
    df = pd.concat([df, pd.DataFrame([{"open": 100, "high": 102, "low": 99,
                                        "close": 101.5, "volume": 100}])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame([{"open": 101, "high": 101.5, "low": 99.5,
                                        "close": 100.0, "volume": 110}])], ignore_index=True)  # only 1.1×
    for _ in range(8):
        df = pd.concat([df, _mk_candles(1)], ignore_index=True)
    res = detect_liquidity_sweep(df, reclaim_volume_mul=2.0)
    assert res.direction == 0


def test_long_wick_sweep_not_filtered():
    """Wick/body = 5 must still be detected — legacy _wick_filter would reject this."""
    df = _mk_candles(60)
    # breach bar: wick-heavy
    df = pd.concat([df, pd.DataFrame([{"open": 100.5, "high": 103, "low": 100,
                                        "close": 101, "volume": 300}])], ignore_index=True)
    # reclaim
    df = pd.concat([df, pd.DataFrame([{"open": 101, "high": 101, "low": 99,
                                        "close": 100, "volume": 350}])], ignore_index=True)
    for _ in range(8):
        df = pd.concat([df, _mk_candles(1)], ignore_index=True)
    res = detect_liquidity_sweep(df)
    assert res.direction == -1


def test_aligned_funding_rejects():
    df = _mk_candles(60)
    df = pd.concat([df, pd.DataFrame([{"open": 100, "high": 102, "low": 99,
                                        "close": 101.5, "volume": 100}])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame([{"open": 101, "high": 101.5, "low": 99.5,
                                        "close": 100.0, "volume": 300}])], ignore_index=True)
    for _ in range(8):
        df = pd.concat([df, _mk_candles(1)], ignore_index=True)
    # funding positive → long-trap short signal (-1) is aligned; reject.
    res = detect_liquidity_sweep(df, funding_rate=0.0001, reject_aligned_funding=True)
    assert res.direction == 0
    assert res.reason == "aligned_funding_reject"
    # Same data without the funding filter → the sweep fires.
    res2 = detect_liquidity_sweep(df, funding_rate=0.0001, reject_aligned_funding=False)
    assert res2.direction == -1


def test_insufficient_bars():
    df = _mk_candles(5)
    res = detect_liquidity_sweep(df)
    assert res.direction == 0
