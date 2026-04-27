"""Strict candlestick pattern detection (Pulse v4).

Replaces the legacy ``_PATTERN_DETECTORS`` list in
``tradingagents/agents/quant_pulse_data.py``.

All rules use closed-bar OHLCV only (rule evaluated on ``df.iloc[-1]``);
no look-ahead. ``detect_all(df)`` returns a list of pattern-name strings,
matching the contract previously exposed by ``detect_patterns()``.

Rule changes vs legacy:
    * ``hammer``         — wick ratio ≥ ``HAMMER_WICK_RATIO`` (default 0.70)
                           instead of ``lw ≥ 2*body`` heuristic.
    * ``bullish_engulfing`` / ``bearish_engulfing``
                         — volume gate ``vol_curr ≥ 1.2 × vol_prev``.
    * ``dark_cloud_cover`` (NEW).
    * ``inverted_harami`` (NEW).
    * Legacy ``_wick_filter`` is **not** applied — long-wick candles are
      part of the signal, not noise (see ``liquidity_sweep.py``).
"""

from __future__ import annotations

from typing import List

import pandas as pd

# Default thresholds (overridable via PulseConfig at the call site).
HAMMER_WICK_RATIO = 0.70
ENGULFING_VOLUME_MUL = 1.2
DOJI_BODY_RATIO = 0.10
HARAMI_BODY_RATIO = 0.5
SMALL_BODY_RATIO = 0.3   # for morning/evening star middle bar


# ── Helpers ──────────────────────────────────────────────────────────

def _body(row) -> float:
    return abs(float(row["close"]) - float(row["open"]))


def _upper_wick(row) -> float:
    return float(row["high"]) - max(float(row["close"]), float(row["open"]))


def _lower_wick(row) -> float:
    return min(float(row["close"]), float(row["open"])) - float(row["low"])


def _total(row) -> float:
    return float(row["high"]) - float(row["low"])


def _is_bullish(row) -> bool:
    return float(row["close"]) > float(row["open"])


def _is_bearish(row) -> bool:
    return float(row["close"]) < float(row["open"])


# ── Single-bar patterns ─────────────────────────────────────────────

def is_doji(row) -> bool:
    total = _total(row)
    if total < 1e-10:
        return False
    return _body(row) / total < DOJI_BODY_RATIO


def is_hammer(row, wick_ratio: float = HAMMER_WICK_RATIO) -> bool:
    """Hammer: lower wick is at least ``wick_ratio`` of the total range.

    Mathematically: ``(min(O,C) − L) / (H − L) ≥ wick_ratio``.
    """
    total = _total(row)
    if total < 1e-10:
        return False
    return (_lower_wick(row) / total) >= wick_ratio


def is_shooting_star(row, wick_ratio: float = HAMMER_WICK_RATIO) -> bool:
    """Inverted hammer / shooting star: upper wick ≥ wick_ratio of total."""
    total = _total(row)
    if total < 1e-10:
        return False
    return (_upper_wick(row) / total) >= wick_ratio


# ── Two-bar patterns ─────────────────────────────────────────────────

def is_bullish_engulfing(prev, curr, volume_mul: float = ENGULFING_VOLUME_MUL) -> bool:
    """Bullish engulfing with strict containment + volume confirmation.

    Rules:
        * prev red, curr green
        * curr.open ≤ prev.close
        * curr.close ≥ prev.open
        * curr.volume ≥ volume_mul × prev.volume
    """
    if not (_is_bearish(prev) and _is_bullish(curr)):
        return False
    if not (float(curr["open"]) <= float(prev["close"])):
        return False
    if not (float(curr["close"]) >= float(prev["open"])):
        return False
    pv = float(prev.get("volume", 0) or 0)
    cv = float(curr.get("volume", 0) or 0)
    if pv > 0 and cv < volume_mul * pv:
        return False
    return True


def is_bearish_engulfing(prev, curr, volume_mul: float = ENGULFING_VOLUME_MUL) -> bool:
    if not (_is_bullish(prev) and _is_bearish(curr)):
        return False
    if not (float(curr["open"]) >= float(prev["close"])):
        return False
    if not (float(curr["close"]) <= float(prev["open"])):
        return False
    pv = float(prev.get("volume", 0) or 0)
    cv = float(curr.get("volume", 0) or 0)
    if pv > 0 and cv < volume_mul * pv:
        return False
    return True


def is_dark_cloud_cover(prev, curr) -> bool:
    """Dark cloud cover: green prev, red curr, opens above prev.high,
    closes below prev midpoint."""
    if not (_is_bullish(prev) and _is_bearish(curr)):
        return False
    if float(curr["open"]) <= float(prev["high"]):
        return False
    midpoint = (float(prev["open"]) + float(prev["close"])) / 2.0
    return float(curr["close"]) < midpoint


def is_bullish_harami(prev, curr) -> bool:
    """Small bullish body fully contained inside a larger bearish prev body."""
    if not (_is_bearish(prev) and _is_bullish(curr)):
        return False
    if not (float(curr["open"]) >= float(prev["close"])):
        return False
    if not (float(curr["close"]) <= float(prev["open"])):
        return False
    if _body(prev) < 1e-10:
        return False
    return _body(curr) < HARAMI_BODY_RATIO * _body(prev)


def is_bearish_harami(prev, curr) -> bool:
    if not (_is_bullish(prev) and _is_bearish(curr)):
        return False
    if not (float(curr["open"]) <= float(prev["close"])):
        return False
    if not (float(curr["close"]) >= float(prev["open"])):
        return False
    if _body(prev) < 1e-10:
        return False
    return _body(curr) < HARAMI_BODY_RATIO * _body(prev)


def is_inverted_harami(prev, curr) -> bool:
    """Inverted harami: small prev body inside a larger curr body in the
    opposite direction. Bullish variant: small bearish prev, large bullish
    curr that engulfs prev's body. Distinct from regular harami in that
    the *prev* body is the small one of the pair."""
    if _body(curr) < 1e-10:
        return False
    if _body(prev) >= HARAMI_BODY_RATIO * _body(curr):
        return False
    # Bullish inverted harami: bearish prev contained in bullish curr.
    if _is_bearish(prev) and _is_bullish(curr):
        return (
            float(prev["open"]) <= float(curr["close"])
            and float(prev["close"]) >= float(curr["open"])
        )
    # Bearish variant: bullish prev contained in bearish curr.
    if _is_bullish(prev) and _is_bearish(curr):
        return (
            float(prev["close"]) <= float(curr["open"])
            and float(prev["open"]) >= float(curr["close"])
        )
    return False


# ── Three-bar patterns (kept for legacy parity) ─────────────────────

def is_morning_star(b0, b1, b2) -> bool:
    if not (_is_bearish(b0) and _is_bullish(b2)):
        return False
    if _body(b1) >= SMALL_BODY_RATIO * _body(b0):
        return False
    midpoint = (float(b0["open"]) + float(b0["close"])) / 2.0
    return float(b2["close"]) > midpoint


def is_evening_star(b0, b1, b2) -> bool:
    if not (_is_bullish(b0) and _is_bearish(b2)):
        return False
    if _body(b1) >= SMALL_BODY_RATIO * _body(b0):
        return False
    midpoint = (float(b0["open"]) + float(b0["close"])) / 2.0
    return float(b2["close"]) < midpoint


def is_three_white_soldiers(b0, b1, b2) -> bool:
    if not (_is_bullish(b0) and _is_bullish(b1) and _is_bullish(b2)):
        return False
    return float(b1["close"]) > float(b0["close"]) and float(b2["close"]) > float(b1["close"])


def is_three_black_crows(b0, b1, b2) -> bool:
    if not (_is_bearish(b0) and _is_bearish(b1) and _is_bearish(b2)):
        return False
    return float(b1["close"]) < float(b0["close"]) and float(b2["close"]) < float(b1["close"])


# ── Public entry point ──────────────────────────────────────────────

def detect_all(df: pd.DataFrame) -> List[str]:
    """Run all candlestick detectors on the closed bar at ``df.iloc[-1]``.

    Returns a list of pattern-name strings (legacy-compatible). Empty
    list when ``df`` is too small or evaluation fails. Does **not** apply
    the legacy ``_wick_filter`` — sweep candles must reach
    ``liquidity_sweep.py`` un-truncated.
    """
    if df is None or df.empty:
        return []
    n = len(df)
    out: List[str] = []
    try:
        last = df.iloc[-1]
        # Single-bar
        if is_doji(last):
            out.append("doji")
        if is_hammer(last):
            out.append("hammer")
        if is_shooting_star(last):
            out.append("shooting_star")

        # Two-bar
        if n >= 2:
            prev = df.iloc[-2]
            if is_bullish_engulfing(prev, last):
                out.append("bullish_engulfing")
            if is_bearish_engulfing(prev, last):
                out.append("bearish_engulfing")
            if is_dark_cloud_cover(prev, last):
                out.append("dark_cloud_cover")
            if is_bullish_harami(prev, last):
                out.append("bullish_harami")
            if is_bearish_harami(prev, last):
                out.append("bearish_harami")
            if is_inverted_harami(prev, last):
                out.append("inverted_harami")

        # Three-bar
        if n >= 3:
            b0 = df.iloc[-3]
            b1 = df.iloc[-2]
            if is_morning_star(b0, b1, last):
                out.append("morning_star")
            if is_evening_star(b0, b1, last):
                out.append("evening_star")
            if is_three_white_soldiers(b0, b1, last):
                out.append("three_white_soldiers")
            if is_three_black_crows(b0, b1, last):
                out.append("three_black_crows")
    except Exception:
        return out
    return out
