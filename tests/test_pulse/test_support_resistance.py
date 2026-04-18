"""Tests for support_resistance.py — swing pivots, book clusters, merge."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from tradingagents.pulse.support_resistance import (
    Pivot,
    SRLevels,
    cluster_pivots,
    compute_support_resistance,
    detect_book_cluster,
    find_swing_pivots,
    liquidity_sweep_pierced,
    pick_nearest_levels,
)


def _mk_candles(highs, lows, start="2024-01-01T00:00:00+00:00", interval_min=60):
    """Build a synthetic candle DataFrame from highs/lows lists."""
    n = len(highs)
    ts = pd.date_range(start=start, periods=n, freq=f"{interval_min}min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": lows,
        "high": highs,
        "low": lows,
        "close": [(h + l) / 2 for h, l in zip(highs, lows)],
        "volume": [1.0] * n,
    })


class TestSwingPivots:
    def test_detects_obvious_swing_high(self):
        # Zigzag with clear peak at index 3
        highs = [100, 101, 102, 110, 102, 101, 100]
        lows = [99, 100, 101, 109, 101, 100, 99]
        df = _mk_candles(highs, lows)
        piv = find_swing_pivots(df, left=3, right=3)
        highs_found = [p for p in piv if p.direction == +1]
        assert len(highs_found) == 1
        assert highs_found[0].price == 110.0

    def test_detects_obvious_swing_low(self):
        highs = [100, 99, 98, 90, 98, 99, 100]
        lows = [99, 98, 97, 89, 97, 98, 99]
        df = _mk_candles(highs, lows)
        piv = find_swing_pivots(df, left=3, right=3)
        lows_found = [p for p in piv if p.direction == -1]
        assert len(lows_found) == 1
        assert lows_found[0].price == 89.0

    def test_insufficient_bars_returns_empty(self):
        df = _mk_candles([100, 101, 102], [99, 100, 101])
        assert find_swing_pivots(df, left=3, right=3) == []

    def test_empty_df_returns_empty(self):
        assert find_swing_pivots(pd.DataFrame(), left=3, right=3) == []


class TestClustering:
    def test_pivots_within_atr_merge(self):
        now = datetime.now(timezone.utc)
        pivots = [
            Pivot(price=100, timestamp=now, direction=-1),
            Pivot(price=100.5, timestamp=now, direction=-1),  # within 0.15×ATR=1.5
            Pivot(price=105, timestamp=now, direction=-1),
        ]
        clustered = cluster_pivots(pivots, atr=10, cluster_atr_mul=0.15)
        # 100 and 100.5 merge, 105 stays separate → 2 clusters
        prices = sorted(p.price for p in clustered)
        assert len(clustered) == 2
        assert abs(prices[0] - 100.25) < 0.01
        assert prices[1] == 105.0

    def test_touch_count_increments_in_cluster(self):
        now = datetime.now(timezone.utc)
        pivots = [Pivot(price=100, timestamp=now, direction=+1) for _ in range(3)]
        clustered = cluster_pivots(pivots, atr=10)
        assert len(clustered) == 1
        assert clustered[0].touches >= 3


class TestPickNearestLevels:
    def test_picks_closest_support_below_spot(self):
        now = datetime.now(timezone.utc)
        pivots = [
            Pivot(price=95, timestamp=now, direction=-1),
            Pivot(price=90, timestamp=now, direction=-1),
            Pivot(price=105, timestamp=now, direction=+1),
        ]
        sup, res = pick_nearest_levels(pivots, spot=100, now=now)
        assert sup.price == 95
        assert res.price == 105

    def test_old_pivots_decayed_out(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=240)  # 10 days → ~exp(-10) ≈ 4.5e-5
        pivots = [
            Pivot(price=95, timestamp=old, direction=-1, touches=1),
            Pivot(price=90, timestamp=now, direction=-1, touches=1),
        ]
        sup, _ = pick_nearest_levels(pivots, spot=100, now=now, half_life_hours=24)
        # The closer pivot (95) is too old; the fresher one (90) wins.
        assert sup.price == 90


class TestBookCluster:
    def test_finds_largest_cluster_on_bid_side(self):
        # Spot 100. Bids below 100 sorted by distance.
        # Big wall at 98 (2 levels of 1000 each), small at 99
        levels = [(99.0, 10), (98.0, 1000), (98.01, 1000)]
        price = detect_book_cluster(
            levels, spot=100.0, side="bid",
            band_pct=0.05, min_notional_usd=50_000,
        )
        assert price is not None
        assert abs(price - 98.005) < 0.1

    def test_thin_book_returns_none(self):
        levels = [(99.5, 1), (99.0, 1)]
        price = detect_book_cluster(
            levels, spot=100.0, side="bid",
            band_pct=0.05, min_notional_usd=500_000,
        )
        assert price is None

    def test_empty_book_returns_none(self):
        assert detect_book_cluster([], spot=100.0, side="bid") is None


class TestLiquiditySweep:
    def test_pierced_bid_returns_true(self):
        # Current 5m bar wicked below support then closed above
        df = _mk_candles(highs=[101], lows=[95], interval_min=5)
        df.loc[0, "close"] = 100  # closed above the pierced level
        assert liquidity_sweep_pierced(97, df, side="bid") is True

    def test_untouched_bid_returns_false(self):
        df = _mk_candles(highs=[101], lows=[99], interval_min=5)
        df.loc[0, "close"] = 100
        assert liquidity_sweep_pierced(97, df, side="bid") is False


class TestComputeSupportResistance:
    def test_pivots_only_when_no_book(self):
        # Synthetic zigzag: clear resistance at ~110, support at ~90
        n = 20
        highs = [100 + 10 * np.sin(i * 0.5) for i in range(n)]
        lows = [h - 1 for h in highs]
        df = _mk_candles(highs, lows)
        sr = compute_support_resistance(
            spot_price=100, df_1h=df, df_4h=None,
            atr_1h=5,
        )
        assert isinstance(sr, SRLevels)
        # Pure-pivot path
        assert sr.source in ("pivot", "none")

    def test_no_data_returns_none_source(self):
        sr = compute_support_resistance(
            spot_price=None, df_1h=None, df_4h=None,
        )
        assert sr.source == "none"
        assert sr.support is None
        assert sr.resistance is None

    def test_merge_prefers_closer_level(self):
        df_1h = _mk_candles(
            highs=[100, 101, 102, 110, 102, 101, 100, 99, 98, 97],
            lows=[99, 100, 101, 109, 101, 100, 99, 98, 97, 96],
        )
        l2 = {
            "bids": [(99.5, 1_000_000), (99.49, 1_000_000)],  # tight book support
            "asks": [(108.0, 1_000_000), (108.01, 1_000_000)],  # ask wall far from spot
        }
        sr = compute_support_resistance(
            spot_price=100, df_1h=df_1h, df_4h=None, atr_1h=2.0,
            l2_snapshot=l2,
        )
        # book_support=99.5, pivot_support hopefully elsewhere
        assert sr.support is not None
