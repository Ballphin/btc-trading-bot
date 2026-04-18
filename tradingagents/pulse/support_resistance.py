"""Support/Resistance — swing-pivot + (optional) L2 book clusters.

Design (from the aggressive-defaults plan):
    * Pivot S/R: swing highs/lows on 1h + 4h with left/right windows.
        Pivots clustered within cluster_atr_mul × ATR.
        Weighted by touch-count (1 + 0.5 × (touches-1)), capped at 2.0.
        Recency decay half-life 24h.
    * Book S/R (Hyperliquid L2): largest contiguous size cluster within ±2%
        of spot, on each side. Liquidity-sweep filter: discard a level if
        it was pierced within the current 5m bar (anti-spoof).
    * Merge: pick the CLOSER level per side (pivot vs book). Tag source.

S/R never *creates* signals in the engine — it only amplifies confluence
that already agrees with the level (see `_score_sr_proximity` in the
scoring engine).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────


@dataclass
class Pivot:
    """A single swing pivot."""
    price: float
    timestamp: datetime
    direction: int          # +1 = resistance (swing high); -1 = support (swing low)
    touches: int = 1        # # of retests within tolerance


@dataclass
class SRLevels:
    """Merged S/R output."""
    support: Optional[float]
    resistance: Optional[float]
    source: str             # "pivot" | "book" | "both" | "none"
    support_touches: int = 0
    resistance_touches: int = 0
    # Diagnostic
    pivot_support: Optional[float] = None
    pivot_resistance: Optional[float] = None
    book_support: Optional[float] = None
    book_resistance: Optional[float] = None


# ── Pivot detection ───────────────────────────────────────────────────


def find_swing_pivots(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
) -> List[Pivot]:
    """Detect strict swing highs/lows on a candle DataFrame.

    A bar is a swing high if its `high` is strictly greater than the `high`
    of the `left` preceding bars AND the `right` following bars. Symmetric
    for swing lows. Needs at least left+right+1 bars.

    Args:
        df: DataFrame with `timestamp`, `high`, `low` columns (ascending).
        left, right: lookback/lookforward bar counts.

    Returns:
        List[Pivot] in chronological order.
    """
    if df is None or df.empty or len(df) < left + right + 1:
        return []

    highs = df["high"].values
    lows = df["low"].values
    timestamps = df["timestamp"].values

    pivots: List[Pivot] = []
    for i in range(left, len(df) - right):
        ph = highs[i]
        pl = lows[i]
        left_highs = highs[i - left:i]
        right_highs = highs[i + 1:i + 1 + right]
        left_lows = lows[i - left:i]
        right_lows = lows[i + 1:i + 1 + right]

        ts = pd.Timestamp(timestamps[i])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ts_py = ts.to_pydatetime()

        if ph > left_highs.max() and ph > right_highs.max():
            pivots.append(Pivot(price=float(ph), timestamp=ts_py, direction=+1))
        if pl < left_lows.min() and pl < right_lows.min():
            pivots.append(Pivot(price=float(pl), timestamp=ts_py, direction=-1))

    return pivots


def cluster_pivots(
    pivots: List[Pivot],
    atr: Optional[float],
    cluster_atr_mul: float = 0.15,
    retest_atr_mul: float = 0.1,
) -> List[Pivot]:
    """Merge pivots within cluster_atr_mul × ATR into single levels.

    Touch count accumulates. Representative price = weighted mean. Timestamp
    = most recent. Direction preserved (separate clusters for supports vs
    resistances).
    """
    if not pivots:
        return []
    if atr is None or atr <= 0:
        # Fallback: 0.15% relative tolerance
        tol_abs = lambda price: price * 0.0015
    else:
        tol_abs = lambda _price: cluster_atr_mul * atr

    # Separate by direction
    result: List[Pivot] = []
    for direction in (+1, -1):
        side = sorted(
            [p for p in pivots if p.direction == direction],
            key=lambda p: p.price,
        )
        if not side:
            continue
        clusters: List[List[Pivot]] = [[side[0]]]
        for p in side[1:]:
            last_cluster = clusters[-1]
            anchor = last_cluster[0].price
            if abs(p.price - anchor) <= tol_abs(anchor):
                last_cluster.append(p)
            else:
                clusters.append([p])

        for cl in clusters:
            prices = [p.price for p in cl]
            ts = max(p.timestamp for p in cl)
            rep_price = float(np.mean(prices))
            touches = len(cl)
            result.append(
                Pivot(
                    price=rep_price,
                    timestamp=ts,
                    direction=direction,
                    touches=touches,
                )
            )

    # Also re-measure retest count: any pivot within retest_atr_mul × ATR of
    # the cluster price (using ALL raw pivots) counts. This catches touches
    # that happened on adjacent bars but fell just outside the cluster
    # threshold.
    retest_tol = (retest_atr_mul * atr) if (atr and atr > 0) else None
    if retest_tol:
        raw_by_dir = {+1: [p.price for p in pivots if p.direction == +1],
                      -1: [p.price for p in pivots if p.direction == -1]}
        for p in result:
            retests = sum(1 for rp in raw_by_dir[p.direction]
                          if abs(rp - p.price) <= retest_tol)
            p.touches = max(p.touches, retests)

    return result


def _pivot_weight(
    pivot: Pivot,
    now: datetime,
    half_life_hours: float = 24.0,
) -> float:
    """Recency-decayed touch-weighted pivot weight (for nearest-level pick).

    weight = (1 + 0.5 × (touches-1), capped at 2.0) × exp(-age_h / half_life)
    """
    touch_w = min(1.0 + 0.5 * max(pivot.touches - 1, 0), 2.0)
    age_h = max((now - pivot.timestamp).total_seconds() / 3600.0, 0.0)
    decay = math.exp(-age_h / max(half_life_hours, 1e-6))
    return touch_w * decay


def pick_nearest_levels(
    pivots: List[Pivot],
    spot: float,
    now: datetime,
    half_life_hours: float = 24.0,
) -> Tuple[Optional[Pivot], Optional[Pivot]]:
    """Return (nearest_support, nearest_resistance) as Pivots.

    Among pivots of the right side, picks the closest to spot whose weight
    (recency × touches) passes a minimum threshold (0.1). Returns None per
    side if no candidate exists.
    """
    support_candidates = [p for p in pivots if p.direction == -1 and p.price < spot]
    resistance_candidates = [p for p in pivots if p.direction == +1 and p.price > spot]

    def _pick(cands: List[Pivot]) -> Optional[Pivot]:
        scored = [(p, _pivot_weight(p, now, half_life_hours)) for p in cands]
        # Filter on minimum weight
        scored = [(p, w) for p, w in scored if w >= 0.1]
        if not scored:
            return None
        # Sort by proximity to spot
        scored.sort(key=lambda pw: abs(pw[0].price - spot))
        return scored[0][0]

    return _pick(support_candidates), _pick(resistance_candidates)


# ── Book-cluster S/R (Hyperliquid L2) ─────────────────────────────────


def detect_book_cluster(
    levels: List[Tuple[float, float]],
    spot: float,
    side: str,
    band_pct: float = 0.02,
    min_notional_usd: float = 500_000,
) -> Optional[float]:
    """Find the largest contiguous size cluster within ±band_pct of spot.

    Args:
        levels: list of (price, size) tuples, sorted by price-distance from spot.
        spot: current spot price.
        side: "bid" (support) or "ask" (resistance).
        band_pct: fraction of spot as search band (0.02 = ±2%).
        min_notional_usd: minimum cluster notional to be considered significant.

    Returns:
        Representative price of the largest cluster, or None if no cluster
        meets the notional floor.
    """
    if not levels or spot <= 0:
        return None

    band = band_pct * spot
    if side == "bid":
        in_band = [(p, s) for p, s in levels if 0 < (spot - p) <= band and s > 0]
    else:
        in_band = [(p, s) for p, s in levels if 0 < (p - spot) <= band and s > 0]
    if not in_band:
        return None

    # Cluster levels whose prices are within 0.1% of each other
    in_band.sort(key=lambda x: x[0])
    clusters: List[List[Tuple[float, float]]] = [[in_band[0]]]
    for lvl in in_band[1:]:
        anchor_price = clusters[-1][0][0]
        if abs(lvl[0] - anchor_price) / anchor_price <= 0.001:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    # Notional per cluster (price × size)
    best = None
    best_notional = 0.0
    for cl in clusters:
        notional = sum(p * s for p, s in cl)
        if notional < min_notional_usd:
            continue
        if notional > best_notional:
            best_notional = notional
            # Size-weighted representative price
            total_size = sum(s for _, s in cl)
            best = sum(p * s for p, s in cl) / total_size if total_size > 0 else cl[0][0]

    return best


def liquidity_sweep_pierced(
    level: float,
    df_5m: pd.DataFrame,
    side: str,
) -> bool:
    """Return True if the most recent 5m bar already pierced `level`.

    For a BID support: pierced if the bar's LOW dropped below `level` but
    closed back above (spoof-pulled wall). For ASK resistance: pierced if
    HIGH pushed above `level` but closed back below.
    """
    if df_5m is None or df_5m.empty:
        return False
    last = df_5m.iloc[-1]
    low = float(last.get("low", last.get("close", 0)))
    high = float(last.get("high", last.get("close", 0)))
    close = float(last.get("close", 0))
    if side == "bid":
        return low < level < close  # wicked below, closed above → pulled
    return high > level > close     # wicked above, closed below → pulled


# ── Top-level compute ─────────────────────────────────────────────────


def compute_support_resistance(
    spot_price: Optional[float],
    df_1h: Optional[pd.DataFrame],
    df_4h: Optional[pd.DataFrame],
    atr_1h: Optional[float] = None,
    l2_snapshot: Optional[dict] = None,
    df_5m: Optional[pd.DataFrame] = None,
    cluster_atr_mul: float = 0.15,
    pivot_left: int = 3,
    pivot_right: int = 3,
    band_pct: float = 0.02,
    min_book_notional_usd: float = 500_000,
    recency_half_life_hours: float = 24.0,
    now: Optional[datetime] = None,
) -> SRLevels:
    """Compute merged support/resistance from pivots + (optional) L2 book.

    Returns SRLevels with source tag. On total failure returns
    SRLevels(None, None, "none").
    """
    now = now or datetime.now(timezone.utc)

    # --- Pivot S/R ---
    pivots: List[Pivot] = []
    if df_1h is not None and not df_1h.empty:
        pivots.extend(find_swing_pivots(df_1h, left=pivot_left, right=pivot_right))
    if df_4h is not None and not df_4h.empty:
        pivots.extend(find_swing_pivots(df_4h, left=pivot_left, right=pivot_right))

    clustered = cluster_pivots(pivots, atr_1h, cluster_atr_mul=cluster_atr_mul)

    pivot_sup_p: Optional[Pivot] = None
    pivot_res_p: Optional[Pivot] = None
    if spot_price is not None and clustered:
        pivot_sup_p, pivot_res_p = pick_nearest_levels(
            clustered, spot_price, now, recency_half_life_hours
        )

    pivot_sup = pivot_sup_p.price if pivot_sup_p else None
    pivot_res = pivot_res_p.price if pivot_res_p else None
    sup_touches = pivot_sup_p.touches if pivot_sup_p else 0
    res_touches = pivot_res_p.touches if pivot_res_p else 0

    # --- Book S/R (optional) ---
    book_sup: Optional[float] = None
    book_res: Optional[float] = None
    if l2_snapshot and spot_price:
        bids = l2_snapshot.get("bids") or []
        asks = l2_snapshot.get("asks") or []
        # Normalize to list of (price, size) tuples of floats
        def _norm(rows):
            out = []
            for r in rows:
                try:
                    if isinstance(r, dict):
                        out.append((float(r.get("px", r.get("price"))), float(r.get("sz", r.get("size")))))
                    else:
                        out.append((float(r[0]), float(r[1])))
                except Exception:
                    continue
            return out

        bid_levels = _norm(bids)
        ask_levels = _norm(asks)

        book_sup = detect_book_cluster(
            bid_levels, spot_price, side="bid",
            band_pct=band_pct, min_notional_usd=min_book_notional_usd,
        )
        book_res = detect_book_cluster(
            ask_levels, spot_price, side="ask",
            band_pct=band_pct, min_notional_usd=min_book_notional_usd,
        )

        # Liquidity-sweep spoof filter
        if book_sup is not None and df_5m is not None:
            if liquidity_sweep_pierced(book_sup, df_5m, "bid"):
                book_sup = None
        if book_res is not None and df_5m is not None:
            if liquidity_sweep_pierced(book_res, df_5m, "ask"):
                book_res = None

    # --- Merge: pick the CLOSER level per side ---
    def _pick_closer(spot: Optional[float], piv: Optional[float], bk: Optional[float]):
        if spot is None:
            return piv or bk, "pivot" if piv else ("book" if bk else "none")
        if piv is None and bk is None:
            return None, "none"
        if piv is None:
            return bk, "book"
        if bk is None:
            return piv, "pivot"
        # Both present: closer wins; if within 0.5 × ATR of each other → "both"
        if atr_1h and abs(piv - bk) <= 0.5 * atr_1h:
            # Average them; tag "both"
            return (piv + bk) / 2.0, "both"
        return (piv if abs(piv - spot) < abs(bk - spot) else bk,
                "pivot" if abs(piv - spot) < abs(bk - spot) else "book")

    sup_price, sup_src = _pick_closer(spot_price, pivot_sup, book_sup)
    res_price, res_src = _pick_closer(spot_price, pivot_res, book_res)

    # Aggregate source tag — both sides must agree for "both"
    if sup_src == res_src:
        source = sup_src
    elif "none" in (sup_src, res_src):
        source = sup_src if res_src == "none" else res_src
    else:
        source = "both"

    return SRLevels(
        support=sup_price,
        resistance=res_price,
        source=source,
        support_touches=sup_touches,
        resistance_touches=res_touches,
        pivot_support=pivot_sup,
        pivot_resistance=pivot_res,
        book_support=book_sup,
        book_resistance=book_res,
    )
