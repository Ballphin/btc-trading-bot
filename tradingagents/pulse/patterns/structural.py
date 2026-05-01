"""Multi-bar structural patterns (Pulse v4).

Detects H&S, Inverse H&S, Double Bottom/Top, Ascending Triangle,
Channel Up/Down, Rectangle on smoothed extrema.

Detector contract (BLOCKER #1 from review): every detector returns
``Optional[PatternHit]`` with both ``fired_at_idx`` (the bar where the
geometric rule first becomes true) and ``confirmation_idx`` (the bar at
which the last extremum's bandwidth has fully passed, i.e., the rule is
prefix-stable). Backtests must score from ``confirmation_idx``.

Crypto-derived rules (per WCT review):
    * Double Bottom: equity-style "V1 > V2 volume" rule is dropped.
      Required instead: ``upper_wick(E2) ≤ wick_ratio × upper_wick(E1)``
      AND ``E2-low is reclaimed within reclaim_minutes`` on 1m data.
    * Channels: R² thresholds dropped — all extrema must lie within
      ``±channel_atr_band_mul × ATR`` of the regression line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from tradingagents.pulse.patterns.extrema import Extremum, find_extrema

PatternDirection = Literal[1, -1]


@dataclass(frozen=True)
class PatternHit:
    name: str
    direction: PatternDirection      # +1 bullish, -1 bearish
    fired_at_idx: int                # first bar where rule is true
    confirmation_idx: int            # ≥ fired_at_idx; rule is prefix-stable here
    extrema_indices: tuple           # raw idx of each defining extremum
    metadata: Dict[str, float] = field(default_factory=dict)
    quality: float = 0.0             # composite quality 0-1 (geometry + volume)
    invalidation_price: Optional[float] = None  # price that invalidates the pattern


# ── Helpers ──────────────────────────────────────────────────────────

def _within_pct(a: float, b: float, tol: float) -> bool:
    if a <= 0 or b <= 0:
        return False
    avg = (a + b) / 2.0
    return abs(a - b) / avg <= tol


def _pct_diff(a: float, b: float) -> float:
    """Absolute percentage difference between two positive values."""
    if a <= 0 or b <= 0:
        return 1.0
    avg = (a + b) / 2.0
    return abs(a - b) / avg


def _last_n_extrema(extrema: List[Extremum], n: int) -> Optional[List[Extremum]]:
    return extrema[-n:] if len(extrema) >= n else None


# ── Quality sub-scores ──────────────────────────────────────────────

_GEO_WEIGHT = 0.6
_VOL_WEIGHT = 0.4


def _volume_breakout_score(
    candles: Optional[pd.DataFrame],
    breakout_idx: int,
    lookback: int = 20,
    threshold: float = 0.25,
) -> float:
    """Breakout volume relative to recent average.  Returns 0–1.

    ``threshold`` is the fractional lift required for a full score
    (e.g. 0.25 → breakout bar ≥ 1.25× trailing mean).
    """
    if candles is None or candles.empty or "volume" not in candles.columns:
        return 0.0
    if breakout_idx >= len(candles) or breakout_idx < 0:
        return 0.0
    start = max(0, breakout_idx - lookback)
    if breakout_idx - start < 5:
        return 0.0
    avg_vol = float(candles.iloc[start:breakout_idx]["volume"].astype(float).mean())
    if avg_vol <= 0:
        return 0.0
    breakout_vol = float(candles.iloc[breakout_idx]["volume"])
    ratio = breakout_vol / avg_vol
    if ratio >= 1.0 + threshold:
        return 1.0
    if ratio >= 1.0:
        return 0.5 + 0.5 * (ratio - 1.0) / max(threshold, 1e-9)
    return max(0.0, 0.5 * ratio)


def _composite_quality(geo: float, vol: float) -> float:
    """Weighted composite of geometry and volume sub-scores."""
    return round(max(0.0, min(1.0, _GEO_WEIGHT * geo + _VOL_WEIGHT * vol)), 4)


# Hard ceiling for double-top/bottom peak-equality tolerance.
# Even if the caller passes match_pct=0.05, we reject anything above this.
_DOUBLE_MATCH_PCT_HARD_CAP = 0.03

# Minimum breakout volume ratio (vs trailing average) for a pattern
# to receive full quality.  Below this, quality is penalised.
_MIN_VOL_RATIO = 0.8
_VOL_PENALTY = 0.3   # subtracted from composite when vol < _MIN_VOL_RATIO × avg


def _apply_vol_floor(
    quality: float,
    candles: Optional[pd.DataFrame],
    breakout_idx: int,
    lookback: int = 20,
) -> float:
    """Penalise quality when breakout volume is below the floor."""
    if candles is None or candles.empty or "volume" not in candles.columns:
        return quality  # no data → no penalty
    if breakout_idx >= len(candles) or breakout_idx < 0:
        return quality
    start = max(0, breakout_idx - lookback)
    if breakout_idx - start < 5:
        return quality
    avg = float(candles.iloc[start:breakout_idx]["volume"].astype(float).mean())
    if avg <= 0:
        return quality
    ratio = float(candles.iloc[breakout_idx]["volume"]) / avg
    if ratio < _MIN_VOL_RATIO:
        return max(0.0, quality - _VOL_PENALTY)
    return quality


# ── 5-extrema reversal patterns ──────────────────────────────────────

def detect_inverse_head_shoulders(
    extrema: List[Extremum],
    *,
    symmetry_pct: float = 0.015,
    candles_higher_tf: Optional[pd.DataFrame] = None,
) -> Optional[PatternHit]:
    """Inverse H&S: minima E1, E3, E5 with E3 < E1 and E3 < E5,
    maxima E2, E4 within ``symmetry_pct`` of their average.
    """
    seq = _last_n_extrema(extrema, 5)
    if seq is None:
        return None
    e1, e2, e3, e4, e5 = seq
    expected = ("min", "max", "min", "max", "min")
    if tuple(e.kind for e in seq) != expected:
        return None
    if not (e3.price < e1.price and e3.price < e5.price):
        return None
    if not _within_pct(e1.price, e5.price, symmetry_pct):
        return None
    if not _within_pct(e2.price, e4.price, symmetry_pct):
        return None
    # Quality: shoulder symmetry + head prominence + neckline symmetry
    shoulder_diff = _pct_diff(e1.price, e5.price)
    neckline_diff = _pct_diff(e2.price, e4.price)
    head_depth = (((e1.price + e5.price) / 2.0) - e3.price) / max(e3.price, 1e-9)
    geo = 1.0 - (shoulder_diff / max(symmetry_pct, 1e-9)) * 0.4 \
        - (neckline_diff / max(symmetry_pct, 1e-9)) * 0.3
    geo = max(0.0, min(1.0, geo + min(head_depth * 2.0, 0.3)))  # deeper head → better
    vol = _volume_breakout_score(candles_higher_tf, e5.idx, threshold=0.25)
    neckline = (e2.price + e4.price) / 2.0
    raw_q = _composite_quality(geo, vol)
    final_q = _apply_vol_floor(raw_q, candles_higher_tf, e5.idx)
    return PatternHit(
        name="inverse_head_shoulders",
        direction=1,
        fired_at_idx=e5.idx,
        confirmation_idx=e5.confirmation_idx,
        extrema_indices=tuple(e.idx for e in seq),
        metadata={"head_price": e3.price, "neckline_price": neckline,
                  "geo_score": round(geo, 4), "vol_score": round(vol, 4)},
        quality=final_q,
        invalidation_price=e3.price,  # head re-test invalidates
    )


def detect_head_shoulders(
    extrema: List[Extremum],
    *,
    symmetry_pct: float = 0.015,
    candles_higher_tf: Optional[pd.DataFrame] = None,
) -> Optional[PatternHit]:
    """Head & Shoulders: maxima E1, E3, E5 with E3 > E1 and E3 > E5,
    minima E2, E4 within ``symmetry_pct``.
    """
    seq = _last_n_extrema(extrema, 5)
    if seq is None:
        return None
    e1, e2, e3, e4, e5 = seq
    expected = ("max", "min", "max", "min", "max")
    if tuple(e.kind for e in seq) != expected:
        return None
    if not (e3.price > e1.price and e3.price > e5.price):
        return None
    if not _within_pct(e1.price, e5.price, symmetry_pct):
        return None
    if not _within_pct(e2.price, e4.price, symmetry_pct):
        return None
    shoulder_diff = _pct_diff(e1.price, e5.price)
    neckline_diff = _pct_diff(e2.price, e4.price)
    head_height = (e3.price - (e1.price + e5.price) / 2.0) / max(e3.price, 1e-9)
    geo = 1.0 - (shoulder_diff / max(symmetry_pct, 1e-9)) * 0.4 \
        - (neckline_diff / max(symmetry_pct, 1e-9)) * 0.3
    geo = max(0.0, min(1.0, geo + min(head_height * 2.0, 0.3)))
    vol = _volume_breakout_score(candles_higher_tf, e5.idx, threshold=0.25)
    neckline = (e2.price + e4.price) / 2.0
    raw_q = _composite_quality(geo, vol)
    final_q = _apply_vol_floor(raw_q, candles_higher_tf, e5.idx)
    return PatternHit(
        name="head_shoulders",
        direction=-1,
        fired_at_idx=e5.idx,
        confirmation_idx=e5.confirmation_idx,
        extrema_indices=tuple(e.idx for e in seq),
        metadata={"head_price": e3.price, "neckline_price": neckline,
                  "geo_score": round(geo, 4), "vol_score": round(vol, 4)},
        quality=final_q,
        invalidation_price=e3.price,  # head re-test invalidates
    )


# ── Double bottom / top (crypto-derived rule) ────────────────────────

def detect_double_bottom(
    extrema: List[Extremum],
    *,
    match_pct: float = 0.02,
    upper_wick_ratio: float = 0.5,
    reclaim_minutes: int = 15,
    candles_higher_tf: Optional[pd.DataFrame] = None,
    candles_1m: Optional[pd.DataFrame] = None,
    timeframe_seconds: Optional[int] = None,
) -> Optional[PatternHit]:
    """Two minima within ``match_pct`` of each other.

    Crypto-derived confirmation (WCT review): instead of equity-style
    ``V1 > V2`` volume rule, require:

        * ``upper_wick(bar_at_E2) ≤ upper_wick_ratio × upper_wick(bar_at_E1)``
          on the higher-tf candles (``candles_higher_tf``); AND
        * the E2-low is reclaimed (price returns above E2.price) within
          ``reclaim_minutes`` minutes on 1m data.

    Returns None if any required candle data is missing.
    """
    # find the two most recent minima
    minima = [e for e in extrema if e.kind == "min"]
    if len(minima) < 2:
        return None
    e1, e2 = minima[-2], minima[-1]
    # Hard cap: reject if peaks differ by more than _DOUBLE_MATCH_PCT_HARD_CAP
    effective_pct = min(match_pct, _DOUBLE_MATCH_PCT_HARD_CAP)
    if not _within_pct(e1.price, e2.price, effective_pct):
        return None

    # Crypto-rule: upper-wick contraction at E2.
    if candles_higher_tf is None or candles_higher_tf.empty:
        return None
    if e2.idx >= len(candles_higher_tf) or e1.idx >= len(candles_higher_tf):
        return None
    bar1 = candles_higher_tf.iloc[e1.idx]
    bar2 = candles_higher_tf.iloc[e2.idx]
    uw1 = float(bar1["high"]) - max(float(bar1["close"]), float(bar1["open"]))
    uw2 = float(bar2["high"]) - max(float(bar2["close"]), float(bar2["open"]))
    if uw1 <= 0:
        return None
    if uw2 > upper_wick_ratio * uw1:
        return None

    # Reclaim within reclaim_minutes on 1m data.
    if candles_1m is None or candles_1m.empty:
        return None
    e2_ts = bar2["timestamp"]
    if not isinstance(e2_ts, pd.Timestamp):
        e2_ts = pd.Timestamp(e2_ts)
    deadline = e2_ts + pd.Timedelta(minutes=reclaim_minutes)
    window = candles_1m[(candles_1m["timestamp"] >= e2_ts)
                        & (candles_1m["timestamp"] <= deadline)]
    if window.empty:
        return None
    reclaimed = bool((window["high"] >= e2.price).any())
    if not reclaimed:
        return None

    # Quality: peak match closeness + wick contraction + volume divergence
    price_diff = _pct_diff(e1.price, e2.price)
    geo = max(0.0, 1.0 - (price_diff / max(effective_pct, 1e-9)))
    wick_factor = 1.0 - (uw2 / max(uw1, 1e-9))  # better contraction → higher
    # Volume divergence bonus: declining volume at E2 signals absorption
    vol_div_bonus = 0.0
    if (candles_higher_tf is not None and "volume" in candles_higher_tf.columns
            and e1.idx < len(candles_higher_tf) and e2.idx < len(candles_higher_tf)):
        v1 = float(candles_higher_tf.iloc[e1.idx]["volume"])
        v2 = float(candles_higher_tf.iloc[e2.idx]["volume"])
        if v1 > 0 and v2 < v1:
            vol_div_bonus = min(0.15, (v1 - v2) / v1 * 0.3)
    geo = max(0.0, min(1.0, geo * 0.6 + max(0.0, wick_factor) * 0.25 + vol_div_bonus))
    vol = _volume_breakout_score(candles_higher_tf, e2.idx, threshold=0.25)
    # Invalidation: below the lower of the two troughs
    inval = min(e1.price, e2.price) * 0.995  # small buffer below
    raw_q = _composite_quality(geo, vol)
    final_q = _apply_vol_floor(raw_q, candles_higher_tf, e2.idx)
    return PatternHit(
        name="double_bottom",
        direction=1,
        fired_at_idx=e2.idx,
        confirmation_idx=e2.confirmation_idx,
        extrema_indices=(e1.idx, e2.idx),
        metadata={"e1_price": e1.price, "e2_price": e2.price,
                  "uw1": uw1, "uw2": uw2, "vol_div_bonus": round(vol_div_bonus, 4),
                  "geo_score": round(geo, 4), "vol_score": round(vol, 4)},
        quality=final_q,
        invalidation_price=inval,
    )


def detect_double_top(
    extrema: List[Extremum],
    *,
    match_pct: float = 0.02,
    lower_wick_ratio: float = 0.5,
    reclaim_minutes: int = 15,
    candles_higher_tf: Optional[pd.DataFrame] = None,
    candles_1m: Optional[pd.DataFrame] = None,
) -> Optional[PatternHit]:
    """Mirror of double bottom: two maxima with lower-wick contraction
    at E2 and the E2-high rejected (price falls below E2.price) within
    ``reclaim_minutes``.
    """
    maxima = [e for e in extrema if e.kind == "max"]
    if len(maxima) < 2:
        return None
    e1, e2 = maxima[-2], maxima[-1]
    effective_pct = min(match_pct, _DOUBLE_MATCH_PCT_HARD_CAP)
    if not _within_pct(e1.price, e2.price, effective_pct):
        return None
    if candles_higher_tf is None or candles_higher_tf.empty:
        return None
    if e2.idx >= len(candles_higher_tf) or e1.idx >= len(candles_higher_tf):
        return None
    bar1 = candles_higher_tf.iloc[e1.idx]
    bar2 = candles_higher_tf.iloc[e2.idx]
    lw1 = min(float(bar1["close"]), float(bar1["open"])) - float(bar1["low"])
    lw2 = min(float(bar2["close"]), float(bar2["open"])) - float(bar2["low"])
    if lw1 <= 0:
        return None
    if lw2 > lower_wick_ratio * lw1:
        return None
    if candles_1m is None or candles_1m.empty:
        return None
    e2_ts = bar2["timestamp"]
    if not isinstance(e2_ts, pd.Timestamp):
        e2_ts = pd.Timestamp(e2_ts)
    deadline = e2_ts + pd.Timedelta(minutes=reclaim_minutes)
    window = candles_1m[(candles_1m["timestamp"] >= e2_ts)
                        & (candles_1m["timestamp"] <= deadline)]
    if window.empty:
        return None
    rejected = bool((window["low"] <= e2.price).any())
    if not rejected:
        return None
    price_diff = _pct_diff(e1.price, e2.price)
    geo = max(0.0, 1.0 - (price_diff / max(effective_pct, 1e-9)))
    wick_factor = 1.0 - (lw2 / max(lw1, 1e-9))
    # Volume divergence bonus: declining volume at E2 signals exhaustion
    vol_div_bonus = 0.0
    if (candles_higher_tf is not None and "volume" in candles_higher_tf.columns
            and e1.idx < len(candles_higher_tf) and e2.idx < len(candles_higher_tf)):
        v1 = float(candles_higher_tf.iloc[e1.idx]["volume"])
        v2 = float(candles_higher_tf.iloc[e2.idx]["volume"])
        if v1 > 0 and v2 < v1:
            vol_div_bonus = min(0.15, (v1 - v2) / v1 * 0.3)
    geo = max(0.0, min(1.0, geo * 0.6 + max(0.0, wick_factor) * 0.25 + vol_div_bonus))
    vol = _volume_breakout_score(candles_higher_tf, e2.idx, threshold=0.25)
    inval = max(e1.price, e2.price) * 1.005
    raw_q = _composite_quality(geo, vol)
    final_q = _apply_vol_floor(raw_q, candles_higher_tf, e2.idx)
    return PatternHit(
        name="double_top",
        direction=-1,
        fired_at_idx=e2.idx,
        confirmation_idx=e2.confirmation_idx,
        extrema_indices=(e1.idx, e2.idx),
        metadata={"e1_price": e1.price, "e2_price": e2.price,
                  "lw1": lw1, "lw2": lw2, "vol_div_bonus": round(vol_div_bonus, 4),
                  "geo_score": round(geo, 4), "vol_score": round(vol, 4)},
        quality=final_q,
        invalidation_price=inval,
    )


# ── Channels (ATR-band rule, no R²) ──────────────────────────────────

def _fit_line(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """OLS slope/intercept (no scipy)."""
    n = len(xs)
    mx = xs.mean()
    my = ys.mean()
    denom = float(np.sum((xs - mx) ** 2))
    if denom < 1e-12:
        return 0.0, my
    slope = float(np.sum((xs - mx) * (ys - my)) / denom)
    intercept = float(my - slope * mx)
    return slope, intercept


def _detect_channel(
    extrema: List[Extremum],
    *,
    direction: PatternDirection,
    atr: float,
    atr_band_mul: float = 0.5,
    min_extrema: int = 6,
    min_bars: int = 18,
    max_volume_ratio: float = 0.7,
    candles_higher_tf: Optional[pd.DataFrame] = None,
) -> Optional[PatternHit]:
    if atr is None or atr <= 0:
        return None
    if len(extrema) < min_extrema:
        return None
    seq = extrema[-min_extrema:]
    span_bars = seq[-1].idx - seq[0].idx
    if span_bars < min_bars:
        return None
    xs = np.array([e.idx for e in seq], dtype=float)
    ys = np.array([e.price for e in seq], dtype=float)
    slope, intercept = _fit_line(xs, ys)
    if direction == 1 and slope <= 0:
        return None
    if direction == -1 and slope >= 0:
        return None
    fitted = slope * xs + intercept
    band = atr_band_mul * atr
    if not bool(np.all(np.abs(ys - fitted) <= band)):
        return None
    # Volume gate: latest higher-tf bar volume ≤ max_volume_ratio × channel-median.
    if candles_higher_tf is not None and not candles_higher_tf.empty:
        lo = max(0, seq[0].idx)
        hi = min(len(candles_higher_tf), seq[-1].idx + 1)
        window = candles_higher_tf.iloc[lo:hi]
        if not window.empty and "volume" in window.columns:
            med = float(window["volume"].median())
            latest = float(candles_higher_tf["volume"].iloc[-1])
            if med > 0 and latest > max_volume_ratio * med:
                return None
    # Quality: tightness of fit (max residual relative to band)
    max_residual = float(np.max(np.abs(ys - fitted)))
    geo = max(0.0, 1.0 - max_residual / max(band, 1e-9))
    vol = _volume_breakout_score(candles_higher_tf, seq[-1].idx, threshold=0.25)
    # Invalidation: price crossing the regression line in the opposite direction
    last_fitted = slope * seq[-1].idx + intercept
    inval = last_fitted - band if direction == 1 else last_fitted + band
    raw_q = _composite_quality(geo, vol)
    final_q = _apply_vol_floor(raw_q, candles_higher_tf, seq[-1].idx)
    return PatternHit(
        name="channel_up" if direction == 1 else "channel_down",
        direction=direction,
        fired_at_idx=seq[-1].idx,
        confirmation_idx=seq[-1].confirmation_idx,
        extrema_indices=tuple(e.idx for e in seq),
        metadata={"slope": slope, "intercept": intercept, "atr": atr,
                  "geo_score": round(geo, 4), "vol_score": round(vol, 4)},
        quality=final_q,
        invalidation_price=round(inval, 2),
    )


def detect_channel_up(extrema, **kw) -> Optional[PatternHit]:
    return _detect_channel(extrema, direction=1, **kw)


def detect_channel_down(extrema, **kw) -> Optional[PatternHit]:
    return _detect_channel(extrema, direction=-1, **kw)


# ── Triangles & Rectangles ───────────────────────────────────────────

def detect_ascending_triangle(
    extrema: List[Extremum],
    *,
    atr: float,
    atr_band_mul: float = 0.5,
    flat_pct: float = 0.01,
    min_touches: int = 4,
) -> Optional[PatternHit]:
    """Flat resistance (≥ 2 maxima within ``flat_pct``) + rising support
    (≥ 2 minima with positive slope, all within ``atr_band_mul × ATR``)."""
    maxima = [e for e in extrema if e.kind == "max"][-3:]
    minima = [e for e in extrema if e.kind == "min"][-3:]
    if len(maxima) < 2 or len(minima) < 2:
        return None
    if (len(maxima) + len(minima)) < min_touches:
        return None
    top = float(np.mean([e.price for e in maxima]))
    if not all(_within_pct(e.price, top, flat_pct) for e in maxima):
        return None
    xs = np.array([e.idx for e in minima], dtype=float)
    ys = np.array([e.price for e in minima], dtype=float)
    slope, intercept = _fit_line(xs, ys)
    if slope <= 0:
        return None
    if atr is None or atr <= 0:
        return None
    fitted = slope * xs + intercept
    if not bool(np.all(np.abs(ys - fitted) <= atr_band_mul * atr)):
        return None
    last_idx = max(maxima[-1].idx, minima[-1].idx)
    confirm = max(maxima[-1].confirmation_idx, minima[-1].confirmation_idx)
    # Quality: flatness of resistance + tightness of support fit
    resistance_spread = max(_pct_diff(m.price, top) for m in maxima)
    fitted = slope * xs + intercept
    support_residual = float(np.max(np.abs(ys - fitted)))
    geo = max(0.0, 1.0 - resistance_spread * 10.0) * 0.5 \
        + max(0.0, 1.0 - support_residual / max(atr_band_mul * atr, 1e-9)) * 0.5
    geo = max(0.0, min(1.0, geo))
    # Invalidation: below the support trendline
    last_support = slope * last_idx + intercept
    inval = last_support - 0.5 * atr
    return PatternHit(
        name="ascending_triangle",
        direction=1,
        fired_at_idx=last_idx,
        confirmation_idx=confirm,
        extrema_indices=tuple(e.idx for e in (maxima + minima)),
        metadata={"resistance": top, "support_slope": slope,
                  "geo_score": round(geo, 4)},
        quality=max(0.0, min(1.0, geo)),  # no volume gate on triangles yet
        invalidation_price=round(inval, 2),
    )


def detect_rectangle(
    extrema: List[Extremum],
    *,
    atr: float,
    atr_band_mul: float = 0.5,
    flat_pct: float = 0.01,
    min_touches: int = 4,
) -> Optional[PatternHit]:
    """Horizontal channel: maxima all within ``flat_pct`` and minima
    all within ``flat_pct``, separated by > 2 × ATR (otherwise it's a
    micro-range)."""
    maxima = [e for e in extrema if e.kind == "max"][-3:]
    minima = [e for e in extrema if e.kind == "min"][-3:]
    if len(maxima) < 2 or len(minima) < 2:
        return None
    if (len(maxima) + len(minima)) < min_touches:
        return None
    top = float(np.mean([e.price for e in maxima]))
    bot = float(np.mean([e.price for e in minima]))
    if not all(_within_pct(e.price, top, flat_pct) for e in maxima):
        return None
    if not all(_within_pct(e.price, bot, flat_pct) for e in minima):
        return None
    if atr is None or atr <= 0:
        return None
    if (top - bot) < 2.0 * atr:
        return None
    last_idx = max(maxima[-1].idx, minima[-1].idx)
    confirm = max(maxima[-1].confirmation_idx, minima[-1].confirmation_idx)
    top_spread = max(_pct_diff(m.price, top) for m in maxima)
    bot_spread = max(_pct_diff(m.price, bot) for m in minima)
    geo = max(0.0, 1.0 - (top_spread + bot_spread) * 5.0)
    return PatternHit(
        name="rectangle",
        direction=0,                    # neutral / range-bound
        fired_at_idx=last_idx,
        confirmation_idx=confirm,
        extrema_indices=tuple(e.idx for e in (maxima + minima)),
        metadata={"top": top, "bottom": bot,
                  "geo_score": round(geo, 4)},
        quality=max(0.0, min(1.0, geo)),
        invalidation_price=None,  # range — no directional invalidation
    )


# ── Single-call helper ──────────────────────────────────────────────

def detect_structural_all(
    candles: pd.DataFrame,
    *,
    bandwidth: Union[int, str] = 8,
    atr: Optional[float] = None,
    candles_1m: Optional[pd.DataFrame] = None,
    symmetry_pct: float = 0.015,
    channel_atr_band_mul: float = 0.5,
    channel_min_extrema: int = 6,
    channel_min_bars: int = 18,
    channel_max_volume_ratio: float = 0.7,
    double_bottom_match_pct: float = 0.02,
    double_bottom_wick_ratio: float = 0.5,
    double_bottom_reclaim_minutes: int = 15,
) -> List[PatternHit]:
    """Run every structural detector on ``candles`` and return all hits."""
    if candles is None or candles.empty:
        return []
    # For auto bandwidth, use minimum candidate (3) for length guard;
    # find_extrema will pick the actual optimal bandwidth.
    min_bw = 3 if bandwidth == "auto" else int(bandwidth)
    if len(candles) < 2 * min_bw + 5:
        return []
    closes = candles["close"].astype(float).values
    extrema = find_extrema(closes, bandwidth=bandwidth)
    out: List[PatternHit] = []
    for fn in (detect_inverse_head_shoulders, detect_head_shoulders):
        hit = fn(extrema, symmetry_pct=symmetry_pct,
                 candles_higher_tf=candles)
        if hit is not None:
            out.append(hit)
    db = detect_double_bottom(
        extrema,
        match_pct=double_bottom_match_pct,
        upper_wick_ratio=double_bottom_wick_ratio,
        reclaim_minutes=double_bottom_reclaim_minutes,
        candles_higher_tf=candles,
        candles_1m=candles_1m,
    )
    if db is not None:
        out.append(db)
    dt = detect_double_top(
        extrema,
        match_pct=double_bottom_match_pct,
        lower_wick_ratio=double_bottom_wick_ratio,
        reclaim_minutes=double_bottom_reclaim_minutes,
        candles_higher_tf=candles,
        candles_1m=candles_1m,
    )
    if dt is not None:
        out.append(dt)
    if atr is not None and atr > 0:
        for fn in (detect_channel_up, detect_channel_down):
            hit = fn(
                extrema,
                atr=atr,
                atr_band_mul=channel_atr_band_mul,
                min_extrema=channel_min_extrema,
                min_bars=channel_min_bars,
                max_volume_ratio=channel_max_volume_ratio,
                candles_higher_tf=candles,
            )
            if hit is not None:
                out.append(hit)
        at = detect_ascending_triangle(extrema, atr=atr, atr_band_mul=channel_atr_band_mul)
        if at is not None:
            out.append(at)
        rect = detect_rectangle(extrema, atr=atr, atr_band_mul=channel_atr_band_mul)
        if rect is not None:
            out.append(rect)
    return out


def _to_chartable(hit: PatternHit, df: pd.DataFrame) -> Dict[str, Any]:
    """Resolve a PatternHit's raw indices to absolute timestamps/prices.

    Returns a JSON-serializable dict with ``extrema`` as a list of
    ``{timestamp, open, high, low, close, kind}`` so the frontend can
    draw trend lines and markers without knowing the original DataFrame.
    """
    extrema_out = []
    for idx in hit.extrema_indices:
        try:
            row = df.iloc[idx]
            ts = row["timestamp"]
            # Handle both pd.Timestamp and datetime
            if hasattr(ts, "isoformat"):
                ts_str = ts.isoformat()
            elif hasattr(ts, "strftime"):
                ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S%z")
            else:
                ts_str = str(ts)
            extrema_out.append({
                "timestamp": ts_str,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "kind": "max" if "max" in hit.name or hit.direction == -1 else "min",
            })
        except (IndexError, KeyError):
            continue
    return {
        "name": hit.name,
        "direction": hit.direction,
        "fired_at_idx": hit.fired_at_idx,
        "confirmation_idx": hit.confirmation_idx,
        "extrema": extrema_out,
        "metadata": dict(hit.metadata),
    }
