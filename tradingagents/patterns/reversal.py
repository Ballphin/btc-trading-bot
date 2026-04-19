"""Reversal pattern detectors: H&S, Inverse H&S, Double/Triple Top/Bottom, Cup & Handle.

All detectors follow the same signature:

    def detect_*(df: pd.DataFrame, timeframe: str, atr_ref: float) -> list[PatternMatch]

Returns 0-or-more matches; caller aggregates via registry.detect_all.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .atr import atr_tol
from .config import pattern_cfg, global_cfg
from .pivots import find_pivots, hybrid_series, pivot_price
from .schemas import (
    AnchorPoint,
    PatternLine,
    PatternMatch,
    PatternState,
    VOLUME_UNKNOWN,
)
from .scoring import (
    combined_score,
    duration_score,
    fit_score_from_violations,
    volume_score_monotonic,
)
from .state_machine import StateParams, classify_state


# ── helpers ───────────────────────────────────────────────────────────

def _iso(ts) -> str:
    if isinstance(ts, (pd.Timestamp,)):
        return ts.tz_localize("UTC").isoformat() if ts.tzinfo is None else ts.tz_convert("UTC").isoformat()
    return str(ts)


def _ts_at(df: pd.DataFrame, idx: int) -> str:
    try:
        t = df["timestamp"].iloc[idx]
        if isinstance(t, pd.Timestamp):
            return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).isoformat()
        return str(t)
    except Exception:
        return str(idx)


def _vol_at(df: pd.DataFrame, idx: int) -> float:
    try:
        return float(df["volume"].iloc[idx])
    except Exception:
        return 0.0


# ── Head and Shoulders ────────────────────────────────────────────────

def detect_head_and_shoulders(df: pd.DataFrame, timeframe: str, atr_ref: float) -> List[PatternMatch]:
    cfg = pattern_cfg("head_and_shoulders")
    gcfg = global_cfg()
    if len(df) < cfg.get("min_bars", 25):
        return []

    peaks, troughs = find_pivots(
        df, timeframe,
        prominence_atr=gcfg.get("pivot_prominence_atr", 0.5),
        atr_ref=atr_ref,
        k_tf_minutes=gcfg.get("k_tf_minutes", 60),
    )
    if len(peaks) < 3 or len(troughs) < 2:
        return []

    peak_src, trough_src = hybrid_series(df)
    out: List[PatternMatch] = []
    price_ref = float(df["close"].iloc[-1])
    min_gap = cfg.get("min_bars_between_shoulders", 6)

    # Iterate over triplets of peaks: (left_shoulder, head, right_shoulder)
    for i in range(len(peaks) - 2):
        for j in range(i + 1, len(peaks) - 1):
            for k in range(j + 1, len(peaks)):
                ls, hd, rs = int(peaks[i]), int(peaks[j]), int(peaks[k])
                if rs - ls < min_gap:
                    continue

                ls_p, hd_p, rs_p = peak_src[ls], peak_src[hd], peak_src[rs]

                # Head must exceed both shoulders
                head_excess = hd_p - max(ls_p, rs_p)
                excess_tol = atr_tol(cfg["head_excess"]["k_atr"], atr_ref,
                                     cfg["head_excess"]["k_pct"], price_ref)
                if head_excess < excess_tol * 0.5:
                    continue

                # Shoulders roughly symmetric
                shoulder_diff = abs(ls_p - rs_p)
                sh_tol = atr_tol(cfg["shoulder_diff"]["k_atr"], atr_ref,
                                 cfg["shoulder_diff"]["k_pct"], price_ref)
                if shoulder_diff > sh_tol * 3.0:
                    continue

                # Find the two troughs between (ls, hd) and (hd, rs)
                left_neck = _nearest_trough_between(troughs, ls, hd)
                right_neck = _nearest_trough_between(troughs, hd, rs)
                if left_neck is None or right_neck is None:
                    continue

                ln_p = float(trough_src[left_neck])
                rn_p = float(trough_src[right_neck])

                # Neckline slope tolerance (roughly horizontal)
                bar_span = max(right_neck - left_neck, 1)
                slope_per_bar = abs(rn_p - ln_p) / bar_span
                slope_tol = atr_tol(cfg["neckline_slope_per_bar"]["k_atr"], atr_ref,
                                    cfg["neckline_slope_per_bar"]["k_pct"], price_ref)
                if slope_per_bar > slope_tol * 3.0:
                    continue

                # Scoring
                fit = fit_score_from_violations(
                    violations=[shoulder_diff, max(0.0, excess_tol - head_excess), slope_per_bar],
                    max_violations=[sh_tol, excess_tol, slope_tol],
                )
                bars = rs - ls + 1
                dur = duration_score(bars)
                vols = [_vol_at(df, ls), _vol_at(df, hd), _vol_at(df, rs)]
                vol = volume_score_monotonic(vols, direction="declining") if sum(vols) > 0 else VOLUME_UNKNOWN

                # Key line = neckline interpolated to RS bar, projected right
                neckline_at_rs = ln_p + (rn_p - ln_p) * ((rs - left_neck) / bar_span)

                # State
                st = classify_state(df, StateParams(
                    key_line_price=neckline_at_rs,
                    bias="bearish",
                    invalidation_price=hd_p,  # close above head invalidates
                    final_anchor_idx=rs,
                    min_post_bars=2,
                ))

                # Build anchors A/B/C/D/E (+ F break if confirmed)
                anchors = [
                    AnchorPoint("A", _ts_at(df, ls), float(ls_p), "peak", ls),
                    AnchorPoint("B", _ts_at(df, left_neck), ln_p, "trough", left_neck),
                    AnchorPoint("C", _ts_at(df, hd), float(hd_p), "peak", hd),
                    AnchorPoint("D", _ts_at(df, right_neck), rn_p, "trough", right_neck),
                    AnchorPoint("E", _ts_at(df, rs), float(rs_p), "peak", rs),
                ]
                if st in (PatternState.CONFIRMED, PatternState.RETESTED):
                    # find first bar after rs closing below neckline
                    post_close = df["close"].to_numpy()[rs + 1:]
                    below = np.where(post_close < neckline_at_rs)[0]
                    if len(below):
                        break_idx = rs + 1 + int(below[0])
                        anchors.append(AnchorPoint(
                            "F", _ts_at(df, break_idx),
                            float(df["close"].iloc[break_idx]), "break", break_idx
                        ))

                lines = [
                    PatternLine(left_neck, right_neck, "neckline",
                                "solid" if st != PatternState.FORMING else "dashed",
                                2, "ok_blue"),
                ]

                match = PatternMatch(
                    name="head_and_shoulders",
                    display_name="Head and Shoulders",
                    bias="bearish",
                    state=st,
                    fit_score=fit,
                    duration_score=dur,
                    volume_score=vol,
                    combined_score=combined_score(fit, dur, vol),
                    timeframe=timeframe,
                    anchors=anchors,
                    lines=lines,
                    bars_in_pattern=bars,
                    description=(
                        f"Left shoulder ${ls_p:.2f}, head ${hd_p:.2f}, "
                        f"right shoulder ${rs_p:.2f}. Neckline ~${neckline_at_rs:.2f}."
                    ),
                    color_token="ok_blue",
                )
                out.append(match)

    return _dedupe_overlapping(out)


def detect_inverse_head_and_shoulders(df: pd.DataFrame, timeframe: str, atr_ref: float) -> List[PatternMatch]:
    """Mirror of H&S — pivots on troughs, bias bullish, invalidation BELOW head."""
    # Flip the series and delegate: cleanest way to avoid code duplication.
    inverted = df.copy()
    inverted["high"], inverted["low"] = -df["low"], -df["high"]
    inverted["open"] = -df["open"]
    inverted["close"] = -df["close"]
    mirrored = detect_head_and_shoulders(inverted, timeframe, atr_ref)

    out: List[PatternMatch] = []
    for m in mirrored:
        # Flip prices back positive; prices read off the INVERTED hybrid series,
        # so anchor.price should be re-read from original df.
        new_anchors = []
        for a in m.anchors:
            # Re-resolve price from original df at the anchor index.
            idx = a.idx
            if a.role == "peak":
                # In the inverted frame, our "peaks" are troughs of original.
                _, trough_src = hybrid_series(df)
                price = float(trough_src[idx])
                role = "trough"
            elif a.role == "trough":
                peak_src, _ = hybrid_series(df)
                price = float(peak_src[idx])
                role = "peak"
            else:
                price = float(df["close"].iloc[idx])
                role = a.role
            new_anchors.append(AnchorPoint(a.label, _ts_at(df, idx), price, role, idx))

        # Re-classify state on the ORIGINAL df (bullish bias, bounce up through neckline).
        key_line = (new_anchors[1].price + new_anchors[3].price) / 2.0
        st = classify_state(df, StateParams(
            key_line_price=key_line,
            bias="bullish",
            invalidation_price=new_anchors[2].price,  # close below head invalidates
            final_anchor_idx=new_anchors[-1].idx if new_anchors[-1].role == "peak" else m.anchors[-1].idx,
            min_post_bars=2,
        ))

        new_lines = [
            PatternLine(m.lines[0].from_idx, m.lines[0].to_idx, "neckline",
                        "solid" if st != PatternState.FORMING else "dashed", 2, "ok_blue")
        ]
        out.append(PatternMatch(
            name="inverse_head_and_shoulders",
            display_name="Inverse Head and Shoulders",
            bias="bullish",
            state=st,
            fit_score=m.fit_score,
            duration_score=m.duration_score,
            volume_score=m.volume_score,
            combined_score=m.combined_score,
            timeframe=timeframe,
            anchors=new_anchors,
            lines=new_lines,
            bars_in_pattern=m.bars_in_pattern,
            description=m.description.replace("Head and Shoulders", "Inverse H&S"),
            color_token="ok_blue",
        ))
    return out


def _nearest_trough_between(troughs: np.ndarray, left: int, right: int) -> Optional[int]:
    """Return lowest-indexed trough strictly between left and right, else None."""
    mask = (troughs > left) & (troughs < right)
    if not mask.any():
        return None
    return int(troughs[mask][0])


def _dedupe_overlapping(matches: List[PatternMatch]) -> List[PatternMatch]:
    """If two matches overlap >50% bar-wise, keep the higher combined_score."""
    if len(matches) <= 1:
        return matches
    matches.sort(key=lambda m: m.combined_score, reverse=True)
    kept: List[PatternMatch] = []
    for m in matches:
        span = (m.anchors[0].idx, m.anchors[-1].idx)
        overlap = False
        for k in kept:
            k_span = (k.anchors[0].idx, k.anchors[-1].idx)
            inter = min(span[1], k_span[1]) - max(span[0], k_span[0])
            union = max(span[1], k_span[1]) - min(span[0], k_span[0])
            if union > 0 and inter / union > 0.5:
                overlap = True
                break
        if not overlap:
            kept.append(m)
    return kept


# ── Double Top / Bottom ───────────────────────────────────────────────

def detect_double_top(df: pd.DataFrame, timeframe: str, atr_ref: float) -> List[PatternMatch]:
    cfg = pattern_cfg("double_top")
    gcfg = global_cfg()
    if len(df) < cfg.get("min_bars", 15):
        return []

    peaks, troughs = find_pivots(
        df, timeframe,
        prominence_atr=gcfg.get("pivot_prominence_atr", 0.5),
        atr_ref=atr_ref,
        k_tf_minutes=gcfg.get("k_tf_minutes", 60),
    )
    if len(peaks) < 2 or len(troughs) < 1:
        return []

    peak_src, trough_src = hybrid_series(df)
    price_ref = float(df["close"].iloc[-1])
    out: List[PatternMatch] = []

    for i in range(len(peaks) - 1):
        for j in range(i + 1, len(peaks)):
            p1, p2 = int(peaks[i]), int(peaks[j])
            p1_p, p2_p = float(peak_src[p1]), float(peak_src[p2])
            diff_tol = atr_tol(cfg["peak_diff"]["k_atr"], atr_ref,
                               cfg["peak_diff"]["k_pct"], price_ref)
            if abs(p1_p - p2_p) > diff_tol * 2.0:
                continue

            # Trough between the two peaks
            mid_t = _nearest_trough_between(troughs, p1, p2)
            if mid_t is None:
                continue
            trough_p = float(trough_src[mid_t])
            depth = min(p1_p, p2_p) - trough_p
            depth_tol = atr_tol(cfg["trough_depth"]["k_atr"], atr_ref,
                                cfg["trough_depth"]["k_pct"], price_ref)
            if depth < depth_tol * 0.5:
                continue

            fit = fit_score_from_violations(
                [abs(p1_p - p2_p), max(0.0, depth_tol - depth)],
                [diff_tol, depth_tol],
            )
            bars = p2 - p1 + 1
            dur = duration_score(bars, target_bars=30)
            vols = [_vol_at(df, p1), _vol_at(df, p2)]
            vol = volume_score_monotonic(vols, direction="declining") if sum(vols) > 0 else VOLUME_UNKNOWN

            st = classify_state(df, StateParams(
                key_line_price=trough_p,
                bias="bearish",
                invalidation_price=max(p1_p, p2_p),
                final_anchor_idx=p2,
                min_post_bars=2,
            ))

            out.append(PatternMatch(
                name="double_top",
                display_name="Double Top",
                bias="bearish",
                state=st,
                fit_score=fit,
                duration_score=dur,
                volume_score=vol,
                combined_score=combined_score(fit, dur, vol),
                timeframe=timeframe,
                anchors=[
                    AnchorPoint("1", _ts_at(df, p1), p1_p, "peak", p1),
                    AnchorPoint("V", _ts_at(df, mid_t), trough_p, "trough", mid_t),
                    AnchorPoint("2", _ts_at(df, p2), p2_p, "peak", p2),
                ],
                lines=[
                    PatternLine(mid_t, p2, "neckline",
                                "solid" if st != PatternState.FORMING else "dashed",
                                2, "ok_orange"),
                ],
                bars_in_pattern=bars,
                description=f"Peaks ~${(p1_p + p2_p) / 2:.2f}, neckline ${trough_p:.2f}.",
                color_token="ok_orange",
            ))
    return _dedupe_overlapping(out)


def detect_double_bottom(df: pd.DataFrame, timeframe: str, atr_ref: float) -> List[PatternMatch]:
    """Mirror of Double Top."""
    inverted = df.copy()
    inverted["high"], inverted["low"] = -df["low"], -df["high"]
    inverted["open"] = -df["open"]
    inverted["close"] = -df["close"]
    mirrored = detect_double_top(inverted, timeframe, atr_ref)

    out: List[PatternMatch] = []
    peak_src, trough_src = hybrid_series(df)
    for m in mirrored:
        new_anchors = []
        for a in m.anchors:
            idx = a.idx
            if a.role == "peak":
                price = float(trough_src[idx]); role = "trough"
            else:
                price = float(peak_src[idx]); role = "peak"
            new_anchors.append(AnchorPoint(a.label, _ts_at(df, idx), price, role, idx))

        # Key line = neckline (peak between the two troughs), bias bullish
        key_line = new_anchors[1].price
        inv_price = min(new_anchors[0].price, new_anchors[2].price)
        st = classify_state(df, StateParams(
            key_line_price=key_line,
            bias="bullish",
            invalidation_price=inv_price,
            final_anchor_idx=new_anchors[-1].idx,
            min_post_bars=2,
        ))
        out.append(PatternMatch(
            name="double_bottom",
            display_name="Double Bottom",
            bias="bullish",
            state=st,
            fit_score=m.fit_score,
            duration_score=m.duration_score,
            volume_score=m.volume_score,
            combined_score=m.combined_score,
            timeframe=timeframe,
            anchors=new_anchors,
            lines=[PatternLine(m.lines[0].from_idx, m.lines[0].to_idx, "neckline",
                               "solid" if st != PatternState.FORMING else "dashed", 2, "ok_orange")],
            bars_in_pattern=m.bars_in_pattern,
            description=f"Troughs ~${(new_anchors[0].price + new_anchors[2].price) / 2:.2f}, neckline ${key_line:.2f}.",
            color_token="ok_orange",
        ))
    return out


# ── Triple Top / Bottom ───────────────────────────────────────────────

def detect_triple_top(df: pd.DataFrame, timeframe: str, atr_ref: float) -> List[PatternMatch]:
    cfg = pattern_cfg("triple_top")
    gcfg = global_cfg()
    if len(df) < cfg.get("min_bars", 25):
        return []

    peaks, troughs = find_pivots(
        df, timeframe,
        prominence_atr=gcfg.get("pivot_prominence_atr", 0.5),
        atr_ref=atr_ref,
        k_tf_minutes=gcfg.get("k_tf_minutes", 60),
    )
    if len(peaks) < 3 or len(troughs) < 2:
        return []

    peak_src, trough_src = hybrid_series(df)
    price_ref = float(df["close"].iloc[-1])
    out: List[PatternMatch] = []

    for i in range(len(peaks) - 2):
        for j in range(i + 1, len(peaks) - 1):
            for k in range(j + 1, len(peaks)):
                p1, p2, p3 = int(peaks[i]), int(peaks[j]), int(peaks[k])
                prices = [peak_src[p1], peak_src[p2], peak_src[p3]]
                diff_tol = atr_tol(cfg["peak_diff"]["k_atr"], atr_ref,
                                   cfg["peak_diff"]["k_pct"], price_ref)
                if max(prices) - min(prices) > diff_tol * 2.5:
                    continue

                t1 = _nearest_trough_between(troughs, p1, p2)
                t2 = _nearest_trough_between(troughs, p2, p3)
                if t1 is None or t2 is None:
                    continue

                trough_line = (trough_src[t1] + trough_src[t2]) / 2.0
                depth_tol = atr_tol(cfg["trough_depth"]["k_atr"], atr_ref,
                                    cfg["trough_depth"]["k_pct"], price_ref)
                depth = min(prices) - trough_line
                if depth < depth_tol * 0.5:
                    continue

                fit = fit_score_from_violations(
                    [max(prices) - min(prices), max(0.0, depth_tol - depth)],
                    [diff_tol, depth_tol],
                )
                bars = p3 - p1 + 1
                dur = duration_score(bars, target_bars=40)
                vols = [_vol_at(df, p1), _vol_at(df, p2), _vol_at(df, p3)]
                vol = volume_score_monotonic(vols, direction="declining") if sum(vols) > 0 else VOLUME_UNKNOWN

                st = classify_state(df, StateParams(
                    key_line_price=float(trough_line),
                    bias="bearish",
                    invalidation_price=float(max(prices)),
                    final_anchor_idx=p3,
                    min_post_bars=2,
                ))

                out.append(PatternMatch(
                    name="triple_top",
                    display_name="Triple Top",
                    bias="bearish",
                    state=st,
                    fit_score=fit,
                    duration_score=dur,
                    volume_score=vol,
                    combined_score=combined_score(fit, dur, vol),
                    timeframe=timeframe,
                    anchors=[
                        AnchorPoint("1", _ts_at(df, p1), float(prices[0]), "peak", p1),
                        AnchorPoint("V1", _ts_at(df, t1), float(trough_src[t1]), "trough", t1),
                        AnchorPoint("2", _ts_at(df, p2), float(prices[1]), "peak", p2),
                        AnchorPoint("V2", _ts_at(df, t2), float(trough_src[t2]), "trough", t2),
                        AnchorPoint("3", _ts_at(df, p3), float(prices[2]), "peak", p3),
                    ],
                    lines=[
                        PatternLine(t1, t2, "neckline",
                                    "solid" if st != PatternState.FORMING else "dashed",
                                    2, "ok_green"),
                    ],
                    bars_in_pattern=bars,
                    description=f"Three peaks ~${np.mean(prices):.2f}, support ~${trough_line:.2f}.",
                    color_token="ok_green",
                ))
    return _dedupe_overlapping(out)


def detect_triple_bottom(df: pd.DataFrame, timeframe: str, atr_ref: float) -> List[PatternMatch]:
    inverted = df.copy()
    inverted["high"], inverted["low"] = -df["low"], -df["high"]
    inverted["open"] = -df["open"]
    inverted["close"] = -df["close"]
    mirrored = detect_triple_top(inverted, timeframe, atr_ref)

    out: List[PatternMatch] = []
    peak_src, trough_src = hybrid_series(df)
    for m in mirrored:
        new_anchors = []
        for a in m.anchors:
            idx = a.idx
            if a.role == "peak":
                price = float(trough_src[idx]); role = "trough"
            else:
                price = float(peak_src[idx]); role = "peak"
            new_anchors.append(AnchorPoint(a.label, _ts_at(df, idx), price, role, idx))

        key_line = (new_anchors[1].price + new_anchors[3].price) / 2.0
        st = classify_state(df, StateParams(
            key_line_price=key_line,
            bias="bullish",
            invalidation_price=min(new_anchors[0].price, new_anchors[2].price, new_anchors[4].price),
            final_anchor_idx=new_anchors[-1].idx,
            min_post_bars=2,
        ))
        out.append(PatternMatch(
            name="triple_bottom",
            display_name="Triple Bottom",
            bias="bullish",
            state=st,
            fit_score=m.fit_score,
            duration_score=m.duration_score,
            volume_score=m.volume_score,
            combined_score=m.combined_score,
            timeframe=timeframe,
            anchors=new_anchors,
            lines=[PatternLine(m.lines[0].from_idx, m.lines[0].to_idx, "neckline",
                               "solid" if st != PatternState.FORMING else "dashed", 2, "ok_green")],
            bars_in_pattern=m.bars_in_pattern,
            description=f"Three troughs, resistance ${key_line:.2f}.",
            color_token="ok_green",
        ))
    return out


# ── Cup and Handle ────────────────────────────────────────────────────

def detect_cup_and_handle(df: pd.DataFrame, timeframe: str, atr_ref: float) -> List[PatternMatch]:
    """Rounded-bottom cup + short downward handle.

    Simple heuristic: split window into cup (first 70%) and handle (last 30%).
    Cup requires rounded bottom (quadratic fit R² > 0.7) with depth >= cup_depth_atr.
    Handle requires a modest dip not exceeding handle_max_depth_frac * cup_depth.
    """
    cfg = pattern_cfg("cup_and_handle")
    if len(df) < cfg.get("min_bars", 40) or atr_ref <= 0:
        return []

    closes = df["close"].to_numpy(dtype=float)
    n = len(closes)
    split = int(n * 0.7)
    cup = closes[:split]
    handle = closes[split:]
    if len(cup) < 15 or len(handle) < 5:
        return []

    # Quadratic fit on cup
    x = np.arange(len(cup))
    try:
        coefs = np.polyfit(x, cup, 2)
    except Exception:
        return []
    if coefs[0] <= 0:
        return []  # Needs to open upward

    fit_vals = np.polyval(coefs, x)
    ss_res = float(np.sum((cup - fit_vals) ** 2))
    ss_tot = float(np.sum((cup - cup.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if r2 < 0.55:
        return []

    cup_top = max(cup[0], cup[-1])
    cup_bottom = float(np.min(cup))
    cup_depth = cup_top - cup_bottom
    if cup_depth < cfg.get("cup_depth_atr", 2.0) * atr_ref:
        return []

    # Cup rims roughly equal
    sym = abs(cup[0] - cup[-1]) / max(cup_depth, 1e-9)
    if sym > cfg.get("cup_symmetry_tol", 0.35):
        return []

    # Handle: max drop should be <= handle_max_depth_frac * cup_depth
    handle_bottom = float(np.min(handle))
    handle_drop = cup_top - handle_bottom
    max_handle_drop = cfg.get("handle_max_depth_frac", 0.5) * cup_depth
    if handle_drop > max_handle_drop * 1.5:
        return []

    # Anchors: cup-left-rim, cup-bottom, cup-right-rim, handle-bottom, (now)
    cup_bot_idx = int(np.argmin(cup))
    handle_bot_idx = split + int(np.argmin(handle))

    fit = fit_score_from_violations(
        [1.0 - r2, sym, max(0.0, handle_drop - max_handle_drop)],
        [0.35, cfg.get("cup_symmetry_tol", 0.35), max_handle_drop],
    )
    bars = n
    dur = duration_score(bars, target_bars=60)
    vols = [_vol_at(df, 0), _vol_at(df, cup_bot_idx), _vol_at(df, split - 1), _vol_at(df, handle_bot_idx)]
    vol = volume_score_monotonic(vols, direction="declining") if sum(vols) > 0 else VOLUME_UNKNOWN

    st = classify_state(df, StateParams(
        key_line_price=float(cup_top),
        bias="bullish",
        invalidation_price=float(cup_bottom),
        final_anchor_idx=handle_bot_idx,
        min_post_bars=2,
    ))

    return [PatternMatch(
        name="cup_and_handle",
        display_name="Cup and Handle",
        bias="bullish",
        state=st,
        fit_score=fit,
        duration_score=dur,
        volume_score=vol,
        combined_score=combined_score(fit, dur, vol),
        timeframe=timeframe,
        anchors=[
            AnchorPoint("L", _ts_at(df, 0), float(cup[0]), "peak", 0),
            AnchorPoint("B", _ts_at(df, cup_bot_idx), float(cup_bottom), "trough", cup_bot_idx),
            AnchorPoint("R", _ts_at(df, split - 1), float(cup[-1]), "peak", split - 1),
            AnchorPoint("H", _ts_at(df, handle_bot_idx), float(handle_bottom), "trough", handle_bot_idx),
        ],
        lines=[
            PatternLine(0, split - 1, "rim",
                        "solid" if st != PatternState.FORMING else "dashed", 2, "ok_pink"),
        ],
        bars_in_pattern=bars,
        description=f"Cup depth ${cup_depth:.2f} ({cup_depth/atr_ref:.1f} ATR), handle dip ${handle_drop:.2f}.",
        color_token="ok_pink",
    )]
