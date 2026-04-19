"""Continuation pattern detectors: Triangles, Flags, Pennants, Wedges."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .atr import atr_tol
from .config import pattern_cfg, global_cfg
from .pivots import find_pivots, hybrid_series
from .schemas import (
    AnchorPoint,
    PatternLine,
    PatternMatch,
    PatternState,
    VOLUME_UNKNOWN,
)
from .scoring import combined_score, duration_score, fit_score_from_violations
from .state_machine import StateParams, classify_state


def _ts_at(df: pd.DataFrame, idx: int) -> str:
    try:
        t = df["timestamp"].iloc[idx]
        if isinstance(t, pd.Timestamp):
            return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).isoformat()
        return str(t)
    except Exception:
        return str(idx)


def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return (slope, intercept, r_squared) for a linear fit."""
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


# ── Triangles (ascending / descending / symmetrical) ──────────────────

def detect_triangles(df: pd.DataFrame, timeframe: str, atr_ref: float) -> List[PatternMatch]:
    cfg = pattern_cfg("triangle")
    gcfg = global_cfg()
    if len(df) < cfg.get("min_bars", 20):
        return []

    peaks, troughs = find_pivots(
        df, timeframe,
        prominence_atr=gcfg.get("pivot_prominence_atr", 0.5),
        atr_ref=atr_ref,
        k_tf_minutes=gcfg.get("k_tf_minutes", 60),
    )
    if len(peaks) < cfg.get("min_pivots_per_side", 2) or len(troughs) < cfg.get("min_pivots_per_side", 2):
        return []

    # Use last N pivots of each kind (most recent window)
    n_pivots = 4
    recent_peaks = peaks[-n_pivots:]
    recent_troughs = troughs[-n_pivots:]
    peak_src, trough_src = hybrid_series(df)

    pk_x = np.array(recent_peaks, dtype=float)
    pk_y = np.array([peak_src[i] for i in recent_peaks])
    tr_x = np.array(recent_troughs, dtype=float)
    tr_y = np.array([trough_src[i] for i in recent_troughs])

    up_slope, up_int, up_r2 = _fit_line(pk_x, pk_y)       # resistance line (peaks)
    dn_slope, dn_int, dn_r2 = _fit_line(tr_x, tr_y)       # support line (troughs)
    r2_gate = cfg.get("r2_threshold", 0.80)
    if up_r2 < r2_gate or dn_r2 < r2_gate:
        return []

    # Slopes normalized to ATR/bar
    if atr_ref <= 0:
        return []
    up_norm = up_slope / atr_ref
    dn_norm = dn_slope / atr_ref
    horiz_tol = 0.05  # < 0.05 ATR/bar => horizontal

    # Classify
    name, display, bias, color = None, None, "neutral", "ok_sky"
    if abs(up_norm) < horiz_tol and dn_norm > horiz_tol:
        name, display, bias = "ascending_triangle", "Ascending Triangle", "bullish"
    elif abs(dn_norm) < horiz_tol and up_norm < -horiz_tol:
        name, display, bias = "descending_triangle", "Descending Triangle", "bearish"
    elif up_norm < -horiz_tol and dn_norm > horiz_tol:
        name, display, bias = "symmetrical_triangle", "Symmetrical Triangle", "neutral"
    else:
        return []

    # Apex (where lines meet)
    if abs(up_slope - dn_slope) < 1e-9:
        return []
    apex_x = (dn_int - up_int) / (up_slope - dn_slope)
    current_x = len(df) - 1
    if apex_x <= current_x - 5:
        return []  # Apex already passed, triangle is done
    if apex_x - current_x > cfg.get("apex_max_bars_ahead", 30):
        return []

    # Build anchors
    anchors = []
    for i, idx in enumerate(recent_peaks):
        anchors.append(AnchorPoint(f"R{i+1}", _ts_at(df, int(idx)), float(peak_src[int(idx)]), "peak", int(idx)))
    for i, idx in enumerate(recent_troughs):
        anchors.append(AnchorPoint(f"S{i+1}", _ts_at(df, int(idx)), float(trough_src[int(idx)]), "trough", int(idx)))
    anchors.sort(key=lambda a: a.idx)

    lines = [
        PatternLine(int(recent_peaks[0]), int(recent_peaks[-1]), "resistance", "solid", 2, color),
        PatternLine(int(recent_troughs[0]), int(recent_troughs[-1]), "support", "solid", 2, color),
    ]

    # Scoring
    fit = fit_score_from_violations(
        [1.0 - up_r2, 1.0 - dn_r2],
        [1.0 - r2_gate, 1.0 - r2_gate],
    )
    bars = int(max(recent_peaks[-1], recent_troughs[-1]) - min(recent_peaks[0], recent_troughs[0]) + 1)
    dur = duration_score(bars, target_bars=40)
    vol = VOLUME_UNKNOWN

    # Key line = whichever line breaks first (use avg of last pivots for state)
    last_pivot_idx = int(max(recent_peaks[-1], recent_troughs[-1]))
    key_line = None
    inv_price = None
    if bias == "bullish":
        key_line = float(up_slope * last_pivot_idx + up_int)
    elif bias == "bearish":
        key_line = float(dn_slope * last_pivot_idx + dn_int)
    st = classify_state(df, StateParams(
        key_line_price=key_line,
        bias=bias,
        invalidation_price=inv_price,
        final_anchor_idx=last_pivot_idx,
        min_post_bars=2,
    ))

    return [PatternMatch(
        name=name,
        display_name=display,
        bias=bias,
        state=st,
        fit_score=fit,
        duration_score=dur,
        volume_score=vol,
        combined_score=combined_score(fit, dur, vol),
        timeframe=timeframe,
        anchors=anchors,
        lines=lines,
        bars_in_pattern=bars,
        description=f"{display} converging over {bars} bars.",
        color_token=color,
    )]


# ── Wedges (rising / falling) ─────────────────────────────────────────

def detect_wedges(df: pd.DataFrame, timeframe: str, atr_ref: float) -> List[PatternMatch]:
    cfg = pattern_cfg("wedge")
    gcfg = global_cfg()
    if len(df) < cfg.get("min_bars", 20) or atr_ref <= 0:
        return []

    peaks, troughs = find_pivots(
        df, timeframe,
        prominence_atr=gcfg.get("pivot_prominence_atr", 0.5),
        atr_ref=atr_ref,
        k_tf_minutes=gcfg.get("k_tf_minutes", 60),
    )
    if len(peaks) < 2 or len(troughs) < 2:
        return []

    peak_src, trough_src = hybrid_series(df)
    recent_peaks = peaks[-3:]
    recent_troughs = troughs[-3:]

    pk_x = np.array(recent_peaks, dtype=float)
    pk_y = np.array([peak_src[i] for i in recent_peaks])
    tr_x = np.array(recent_troughs, dtype=float)
    tr_y = np.array([trough_src[i] for i in recent_troughs])

    up_slope, _, up_r2 = _fit_line(pk_x, pk_y)
    dn_slope, _, dn_r2 = _fit_line(tr_x, tr_y)
    r2_gate = cfg.get("r2_threshold", 0.80)
    if up_r2 < r2_gate or dn_r2 < r2_gate:
        return []

    up_norm = up_slope / atr_ref
    dn_norm = dn_slope / atr_ref

    # Both slopes same direction + converging
    if up_norm > 0 and dn_norm > 0 and up_norm < dn_norm:
        name, display, bias, color = "rising_wedge", "Rising Wedge", "bearish", "ok_pink"
    elif up_norm < 0 and dn_norm < 0 and up_norm < dn_norm:
        name, display, bias, color = "falling_wedge", "Falling Wedge", "bullish", "ok_pink"
    else:
        return []

    min_diff = cfg.get("min_slope_diff_atr_per_bar", 0.05)
    if abs(up_norm - dn_norm) < min_diff:
        return []

    anchors = []
    for i, idx in enumerate(recent_peaks):
        anchors.append(AnchorPoint(f"R{i+1}", _ts_at(df, int(idx)), float(peak_src[int(idx)]), "peak", int(idx)))
    for i, idx in enumerate(recent_troughs):
        anchors.append(AnchorPoint(f"S{i+1}", _ts_at(df, int(idx)), float(trough_src[int(idx)]), "trough", int(idx)))
    anchors.sort(key=lambda a: a.idx)

    lines = [
        PatternLine(int(recent_peaks[0]), int(recent_peaks[-1]), "upper", "solid", 2, color),
        PatternLine(int(recent_troughs[0]), int(recent_troughs[-1]), "lower", "solid", 2, color),
    ]

    fit = fit_score_from_violations(
        [1.0 - up_r2, 1.0 - dn_r2, max(0.0, min_diff - abs(up_norm - dn_norm))],
        [1.0 - r2_gate, 1.0 - r2_gate, min_diff],
    )
    bars = int(max(recent_peaks[-1], recent_troughs[-1]) - min(recent_peaks[0], recent_troughs[0]) + 1)
    dur = duration_score(bars, target_bars=40)
    vol = VOLUME_UNKNOWN

    last_idx = int(max(recent_peaks[-1], recent_troughs[-1]))
    # Key line = the line whose break confirms; rising wedge breaks the lower line down.
    if bias == "bearish":
        slope_l, int_l, _ = _fit_line(tr_x, tr_y)
        key_line = float(slope_l * last_idx + int_l)
    else:
        slope_u, int_u, _ = _fit_line(pk_x, pk_y)
        key_line = float(slope_u * last_idx + int_u)

    st = classify_state(df, StateParams(
        key_line_price=key_line,
        bias=bias,
        invalidation_price=None,
        final_anchor_idx=last_idx,
        min_post_bars=2,
    ))

    return [PatternMatch(
        name=name,
        display_name=display,
        bias=bias,
        state=st,
        fit_score=fit,
        duration_score=dur,
        volume_score=vol,
        combined_score=combined_score(fit, dur, vol),
        timeframe=timeframe,
        anchors=anchors,
        lines=lines,
        bars_in_pattern=bars,
        description=f"{display} over {bars} bars.",
        color_token=color,
    )]


# ── Flags and Pennants ────────────────────────────────────────────────

def detect_flags_pennants(df: pd.DataFrame, timeframe: str, atr_ref: float) -> List[PatternMatch]:
    """Detect pole + consolidation shape. Channel = flag, converging = pennant."""
    flag_cfg = pattern_cfg("flag")
    pen_cfg = pattern_cfg("pennant")
    if len(df) < flag_cfg.get("min_bars", 10) or atr_ref <= 0:
        return []

    closes = df["close"].to_numpy(dtype=float)
    # Split: first 40% = potential pole, last 60% = consolidation
    n = len(closes)
    pole_end = int(n * 0.4)
    if pole_end < 3 or n - pole_end < 5:
        return []

    pole = closes[:pole_end]
    cons = closes[pole_end:]
    pole_range = float(pole[-1] - pole[0])
    min_pole = flag_cfg.get("pole_height_atr", 3.0) * atr_ref
    if abs(pole_range) < min_pole:
        return []

    # Determine pole direction
    pole_bull = pole_range > 0

    # Consolidation pivots (small window, use simple highs/lows)
    highs = df["high"].to_numpy(dtype=float)[pole_end:]
    lows = df["low"].to_numpy(dtype=float)[pole_end:]
    x = np.arange(len(cons), dtype=float)

    hi_slope, hi_int, hi_r2 = _fit_line(x, highs)
    lo_slope, lo_int, lo_r2 = _fit_line(x, lows)
    if hi_r2 < 0.4 or lo_r2 < 0.4:
        return []

    hi_norm = hi_slope / atr_ref
    lo_norm = lo_slope / atr_ref

    # Flag: parallel channel sloping AGAINST the pole direction.
    # Pennant: converging (hi_slope < 0, lo_slope > 0 for bull; mirror).
    name = None
    display = None
    bias = "bullish" if pole_bull else "bearish"
    color = "ok_vermil"
    parallel_tol = 0.04
    is_parallel = abs(hi_norm - lo_norm) < parallel_tol
    is_converging = (hi_slope < 0 and lo_slope > 0) or (hi_slope > 0 and lo_slope < 0)

    if is_parallel:
        # Flag — slopes should be opposite to pole
        if pole_bull and hi_norm < -0.01 and lo_norm < -0.01:
            name, display = "bull_flag", "Bull Flag"
        elif not pole_bull and hi_norm > 0.01 and lo_norm > 0.01:
            name, display = "bear_flag", "Bear Flag"
    elif is_converging:
        if pole_bull:
            name, display = "bull_pennant", "Bull Pennant"
        else:
            name, display = "bear_pennant", "Bear Pennant"

    if not name:
        return []

    cons_end_global = n - 1
    cons_start_global = pole_end
    pole_top = float(np.max(pole)) if pole_bull else float(np.min(pole))
    pole_bottom = float(np.min(pole)) if pole_bull else float(np.max(pole))

    anchors = [
        AnchorPoint("P0", _ts_at(df, 0), float(pole[0]), "peak" if not pole_bull else "trough", 0),
        AnchorPoint("P1", _ts_at(df, pole_end - 1), float(pole[-1]), "peak" if pole_bull else "trough", pole_end - 1),
        AnchorPoint("C0", _ts_at(df, cons_start_global), float(closes[cons_start_global]), "break", cons_start_global),
        AnchorPoint("C1", _ts_at(df, cons_end_global), float(closes[cons_end_global]), "break", cons_end_global),
    ]
    lines = [
        PatternLine(cons_start_global, cons_end_global, "upper", "solid", 2, color),
        PatternLine(cons_start_global, cons_end_global, "lower", "solid", 2, color),
    ]

    fit = fit_score_from_violations(
        [max(0.0, 0.6 - hi_r2), max(0.0, 0.6 - lo_r2)],
        [0.4, 0.4],
    )
    bars = n
    dur = duration_score(bars, target_bars=20)
    vol = VOLUME_UNKNOWN

    # Key line: upper for bull, lower for bear
    key_line = (
        float(hi_slope * (len(cons) - 1) + hi_int)
        if pole_bull
        else float(lo_slope * (len(cons) - 1) + lo_int)
    )

    st = classify_state(df, StateParams(
        key_line_price=key_line,
        bias=bias,
        invalidation_price=pole_bottom if pole_bull else pole_top,
        final_anchor_idx=cons_end_global,
        min_post_bars=1,
    ))

    return [PatternMatch(
        name=name,
        display_name=display,
        bias=bias,
        state=st,
        fit_score=fit,
        duration_score=dur,
        volume_score=vol,
        combined_score=combined_score(fit, dur, vol),
        timeframe=timeframe,
        anchors=anchors,
        lines=lines,
        bars_in_pattern=bars,
        description=f"{display}: pole ${abs(pole_range):.2f} ({abs(pole_range)/atr_ref:.1f} ATR).",
        color_token=color,
    )]
