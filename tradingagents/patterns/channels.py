"""Channel detectors + auto-drawn support/resistance trendlines."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .config import pattern_cfg, global_cfg
from .continuation import _fit_line
from .pivots import find_pivots, hybrid_series
from .schemas import (
    AnchorPoint,
    PatternLine,
    PatternMatch,
    PatternState,
    VOLUME_UNKNOWN,
)
from .scoring import combined_score, duration_score, fit_score_from_violations


def _ts_at(df: pd.DataFrame, idx: int) -> str:
    try:
        t = df["timestamp"].iloc[idx]
        if isinstance(t, pd.Timestamp):
            return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).isoformat()
        return str(t)
    except Exception:
        return str(idx)


def detect_channels(df: pd.DataFrame, timeframe: str, atr_ref: float) -> List[PatternMatch]:
    """Parallel channel (horizontal / ascending / descending)."""
    cfg = pattern_cfg("channel")
    gcfg = global_cfg()
    if len(df) < cfg.get("min_bars", 20) or atr_ref <= 0:
        return []

    peaks, troughs = find_pivots(
        df, timeframe,
        prominence_atr=gcfg.get("pivot_prominence_atr", 0.5),
        atr_ref=atr_ref,
        k_tf_minutes=gcfg.get("k_tf_minutes", 60),
    )
    if len(peaks) < cfg.get("min_pivots_per_side", 2) or len(troughs) < cfg.get("min_pivots_per_side", 2):
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
    r2_gate = cfg.get("r2_threshold", 0.85)
    if up_r2 < r2_gate or dn_r2 < r2_gate:
        return []

    # Must be parallel (same sign, similar magnitude)
    up_norm = up_slope / atr_ref
    dn_norm = dn_slope / atr_ref
    if abs(up_norm - dn_norm) > 0.03:
        return []

    if abs(up_norm) < 0.02:
        name, display, bias = "horizontal_channel", "Range", "neutral"
    elif up_norm > 0:
        name, display, bias = "ascending_channel", "Ascending Channel", "bullish"
    else:
        name, display, bias = "descending_channel", "Descending Channel", "bearish"

    anchors = []
    for i, idx in enumerate(recent_peaks):
        anchors.append(AnchorPoint(f"R{i+1}", _ts_at(df, int(idx)), float(peak_src[int(idx)]), "peak", int(idx)))
    for i, idx in enumerate(recent_troughs):
        anchors.append(AnchorPoint(f"S{i+1}", _ts_at(df, int(idx)), float(trough_src[int(idx)]), "trough", int(idx)))
    anchors.sort(key=lambda a: a.idx)

    lines = [
        PatternLine(int(recent_peaks[0]), int(recent_peaks[-1]), "upper_channel", "solid", 1, "ok_gray"),
        PatternLine(int(recent_troughs[0]), int(recent_troughs[-1]), "lower_channel", "solid", 1, "ok_gray"),
    ]

    fit = fit_score_from_violations(
        [1.0 - up_r2, 1.0 - dn_r2, abs(up_norm - dn_norm)],
        [1.0 - r2_gate, 1.0 - r2_gate, 0.03],
    )
    bars = int(max(recent_peaks[-1], recent_troughs[-1]) - min(recent_peaks[0], recent_troughs[0]) + 1)
    dur = duration_score(bars, target_bars=40)

    return [PatternMatch(
        name=name,
        display_name=display,
        bias=bias,
        state=PatternState.COMPLETED,  # Channels don't have break semantics here
        fit_score=fit,
        duration_score=dur,
        volume_score=VOLUME_UNKNOWN,
        combined_score=combined_score(fit, dur, VOLUME_UNKNOWN),
        timeframe=timeframe,
        anchors=anchors,
        lines=lines,
        bars_in_pattern=bars,
        description=f"{display} over {bars} bars.",
        color_token="ok_gray",
    )]


def detect_auto_trendlines(df: pd.DataFrame, timeframe: str, atr_ref: float) -> List[PatternMatch]:
    """Auto-drawn support and resistance diagonals from the last 2-3 pivots each."""
    gcfg = global_cfg()
    if len(df) < 15 or atr_ref <= 0:
        return []

    peaks, troughs = find_pivots(
        df, timeframe,
        prominence_atr=gcfg.get("pivot_prominence_atr", 0.5),
        atr_ref=atr_ref,
        k_tf_minutes=gcfg.get("k_tf_minutes", 60),
    )

    out: List[PatternMatch] = []
    peak_src, trough_src = hybrid_series(df)

    for kind, idxs, src, label, color, role in (
        ("resistance_trendline", peaks, peak_src, "Resistance Line", "ok_gray", "peak"),
        ("support_trendline", troughs, trough_src, "Support Line", "ok_gray", "trough"),
    ):
        if len(idxs) < 2:
            continue
        recent = idxs[-3:] if len(idxs) >= 3 else idxs[-2:]
        x = np.array(recent, dtype=float)
        y = np.array([src[i] for i in recent])
        slope, intercept, r2 = _fit_line(x, y)
        if r2 < 0.80:
            continue

        anchors = [
            AnchorPoint(f"{role.upper()[:1]}{i+1}", _ts_at(df, int(idx)), float(src[int(idx)]), role, int(idx))
            for i, idx in enumerate(recent)
        ]
        lines = [
            PatternLine(int(recent[0]), int(recent[-1]), kind, "dotted", 1, color),
        ]
        fit = max(0.0, (r2 - 0.8) / 0.2)
        bars = int(recent[-1] - recent[0] + 1)

        out.append(PatternMatch(
            name=kind,
            display_name=label,
            bias="neutral",
            state=PatternState.COMPLETED,
            fit_score=fit,
            duration_score=duration_score(bars, target_bars=40),
            volume_score=VOLUME_UNKNOWN,
            combined_score=combined_score(fit, duration_score(bars, target_bars=40), VOLUME_UNKNOWN),
            timeframe=timeframe,
            anchors=anchors,
            lines=lines,
            bars_in_pattern=bars,
            description=f"{label} through {len(recent)} pivots (R²={r2:.2f}).",
            color_token=color,
        ))
    return out
