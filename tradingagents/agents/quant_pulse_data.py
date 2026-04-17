"""Quant Pulse Data Aggregator — multi-timeframe candle fetch, indicator
computation, candlestick pattern detection, VWAP, and order flow context.

All indicator logic lives in _compute_tf_indicators() which is shared between
the live build_pulse_report() path and the historical backtest path.
"""

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.hyperliquid_client import (
    HyperliquidClient,
    PULSE_CACHE_TTL,
    _INTERVAL_SECONDS,
)

logger = logging.getLogger(__name__)

# ── Timeframe configs ─────────────────────────────────────────────────

PULSE_TIMEFRAMES = {
    "1m":  {"candles": 60,  "pattern": False},
    "5m":  {"candles": 60,  "pattern": False},
    "15m": {"candles": 48,  "pattern": False},
    "1h":  {"candles": 48,  "pattern": True},
    "4h":  {"candles": 24,  "pattern": True},
}

# Minimum bars needed for each indicator via stockstats
_MIN_BARS = {
    "rsi_14": 15,
    "macdh": 35,        # MACD(12,26,9) needs 26+9
    "boll_ub": 20,      # BB(20,2)
    "close_9_ema": 9,
    "close_21_ema": 21,
    "atr_14": 15,
}

# ── 10 candlestick patterns (pure Python, 1h/4h only) ─────────────────

_PATTERN_DETECTORS = {}  # populated below


def _body(row):
    return abs(row["close"] - row["open"])


def _upper_wick(row):
    return row["high"] - max(row["close"], row["open"])


def _lower_wick(row):
    return min(row["close"], row["open"]) - row["low"]


def _is_bullish(row):
    return row["close"] > row["open"]


def _is_bearish(row):
    return row["close"] < row["open"]


def _wick_filter(row, max_wick_body_ratio=3.0):
    """Return True if wick is suspiciously large (liquidation artifact)."""
    b = _body(row)
    if b < 1e-10:
        return True  # doji-like, no body to compare
    return max(_upper_wick(row), _lower_wick(row)) / b > max_wick_body_ratio


def _detect_doji(df):
    if len(df) < 1:
        return []
    row = df.iloc[-1]
    b = _body(row)
    total = row["high"] - row["low"]
    if total < 1e-10:
        return []
    if b / total < 0.1:
        return [("doji", 0)]  # neutral direction
    return []


def _detect_bullish_engulfing(df):
    if len(df) < 2:
        return []
    prev, curr = df.iloc[-2], df.iloc[-1]
    if _is_bearish(prev) and _is_bullish(curr):
        if curr["open"] <= prev["close"] and curr["close"] >= prev["open"]:
            if not _wick_filter(curr):
                return [("bullish_engulfing", 1)]
    return []


def _detect_bearish_engulfing(df):
    if len(df) < 2:
        return []
    prev, curr = df.iloc[-2], df.iloc[-1]
    if _is_bullish(prev) and _is_bearish(curr):
        if curr["open"] >= prev["close"] and curr["close"] <= prev["open"]:
            if not _wick_filter(curr):
                return [("bearish_engulfing", -1)]
    return []


def _detect_hammer(df):
    if len(df) < 1:
        return []
    row = df.iloc[-1]
    b = _body(row)
    lw = _lower_wick(row)
    uw = _upper_wick(row)
    if b < 1e-10:
        return []
    if lw >= 2 * b and uw <= b * 0.5 and not _wick_filter(row):
        return [("hammer", 1)]
    return []


def _detect_shooting_star(df):
    if len(df) < 1:
        return []
    row = df.iloc[-1]
    b = _body(row)
    uw = _upper_wick(row)
    lw = _lower_wick(row)
    if b < 1e-10:
        return []
    if uw >= 2 * b and lw <= b * 0.5 and not _wick_filter(row):
        return [("shooting_star", -1)]
    return []


def _detect_morning_star(df):
    if len(df) < 3:
        return []
    first, mid, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if (_is_bearish(first) and _body(mid) < _body(first) * 0.3
            and _is_bullish(third) and third["close"] > (first["open"] + first["close"]) / 2):
        if not _wick_filter(third):
            return [("morning_star", 1)]
    return []


def _detect_evening_star(df):
    if len(df) < 3:
        return []
    first, mid, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if (_is_bullish(first) and _body(mid) < _body(first) * 0.3
            and _is_bearish(third) and third["close"] < (first["open"] + first["close"]) / 2):
        if not _wick_filter(third):
            return [("evening_star", -1)]
    return []


def _detect_three_white_soldiers(df):
    if len(df) < 3:
        return []
    bars = [df.iloc[-3], df.iloc[-2], df.iloc[-1]]
    if all(_is_bullish(b) for b in bars):
        if bars[1]["close"] > bars[0]["close"] and bars[2]["close"] > bars[1]["close"]:
            if all(not _wick_filter(b) for b in bars):
                return [("three_white_soldiers", 1)]
    return []


def _detect_three_black_crows(df):
    if len(df) < 3:
        return []
    bars = [df.iloc[-3], df.iloc[-2], df.iloc[-1]]
    if all(_is_bearish(b) for b in bars):
        if bars[1]["close"] < bars[0]["close"] and bars[2]["close"] < bars[1]["close"]:
            if all(not _wick_filter(b) for b in bars):
                return [("three_black_crows", -1)]
    return []


def _detect_harami(df):
    if len(df) < 2:
        return []
    prev, curr = df.iloc[-2], df.iloc[-1]
    results = []
    if _is_bearish(prev) and _is_bullish(curr):
        if curr["open"] >= prev["close"] and curr["close"] <= prev["open"]:
            if _body(curr) < _body(prev) * 0.5 and not _wick_filter(curr):
                results.append(("bullish_harami", 1))
    elif _is_bullish(prev) and _is_bearish(curr):
        if curr["open"] <= prev["close"] and curr["close"] >= prev["open"]:
            if _body(curr) < _body(prev) * 0.5 and not _wick_filter(curr):
                results.append(("bearish_harami", -1))
    return results


_PATTERN_DETECTORS = [
    _detect_doji, _detect_bullish_engulfing, _detect_bearish_engulfing,
    _detect_hammer, _detect_shooting_star, _detect_morning_star,
    _detect_evening_star, _detect_three_white_soldiers,
    _detect_three_black_crows, _detect_harami,
]


def detect_patterns(df: pd.DataFrame) -> List[str]:
    """Run all pattern detectors on a candle DataFrame, return pattern names."""
    patterns = []
    for detector in _PATTERN_DETECTORS:
        for name, _ in detector(df):
            patterns.append(name)
    return patterns


# ── Shared indicator computation ──────────────────────────────────────

def _compute_tf_indicators(
    df: pd.DataFrame,
    timeframe: str,
    detect_patterns_flag: bool = False,
) -> Tuple[dict, float]:
    """Compute RSI, MACD, BB, EMA, ATR, rel_volume from a candle DataFrame.

    Shared between live build_pulse_report() and backtest
    build_pulse_report_from_historical().

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        timeframe: Interval string (e.g. "1m", "1h")
        detect_patterns_flag: If True, run candlestick pattern detection

    Returns:
        (indicators_dict, coverage_pct) where coverage_pct is 0.0-1.0
    """
    n_total = 6  # rsi, macdh, bb_pct, ema_cross, atr, rel_volume
    n_valid = 0
    result = {
        "rsi": None,
        "macd_hist": None,
        "bb_pct": None,
        "ema_cross": None,
        "rel_volume": None,
        "atr": None,
        "patterns": [],
    }

    if df.empty or len(df) < 2:
        return result, 0.0

    try:
        ss = wrap(df[["open", "high", "low", "close", "volume"]].copy())

        # RSI
        if len(df) >= _MIN_BARS["rsi_14"]:
            rsi_val = ss["rsi_14"].iloc[-1]
            if pd.notna(rsi_val):
                result["rsi"] = float(rsi_val)
                n_valid += 1

        # MACD histogram
        if len(df) >= _MIN_BARS["macdh"]:
            macdh_val = ss["macdh"].iloc[-1]
            if pd.notna(macdh_val):
                result["macd_hist"] = float(macdh_val)
                n_valid += 1

        # Bollinger Bands → bb_pct
        if len(df) >= _MIN_BARS["boll_ub"]:
            ub = ss["boll_ub"].iloc[-1]
            lb = ss["boll_lb"].iloc[-1]
            close = df["close"].iloc[-1]
            if pd.notna(ub) and pd.notna(lb) and (ub - lb) > 1e-10:
                result["bb_pct"] = float((close - lb) / (ub - lb))
                n_valid += 1

        # EMA cross
        if len(df) >= _MIN_BARS["close_21_ema"]:
            ema9 = ss["close_9_ema"].iloc[-1]
            ema21 = ss["close_21_ema"].iloc[-1]
            if pd.notna(ema9) and pd.notna(ema21):
                result["ema_cross"] = "bullish" if ema9 > ema21 else "bearish"
                result["_ema9"] = float(ema9)
                result["_ema21"] = float(ema21)
                n_valid += 1

        # ATR
        if len(df) >= _MIN_BARS["atr_14"]:
            atr_val = ss["atr_14"].iloc[-1]
            if pd.notna(atr_val):
                result["atr"] = float(atr_val)
                n_valid += 1

        # Relative volume (manual)
        if len(df) >= 20:
            vol_sma20 = df["volume"].iloc[-20:].mean()
            if vol_sma20 > 0:
                result["rel_volume"] = float(df["volume"].iloc[-1] / vol_sma20)
                n_valid += 1
        elif len(df) >= 2:
            vol_mean = df["volume"].mean()
            if vol_mean > 0:
                result["rel_volume"] = float(df["volume"].iloc[-1] / vol_mean)
                n_valid += 1

        # MACD direction (3 bars ago, with contiguity check)
        if result["macd_hist"] is not None and len(df) >= 4 and result["atr"] is not None:
            interval_sec = _INTERVAL_SECONDS.get(timeframe, 60)
            ts_last = df["timestamp"].iloc[-1]
            ts_4ago = df["timestamp"].iloc[-4]
            gap_sec = (ts_last - ts_4ago).total_seconds()
            if gap_sec <= 3.3 * interval_sec:
                macdh_3ago = ss["macdh"].iloc[-4]
                if pd.notna(macdh_3ago) and result["atr"] > 1e-10:
                    delta_norm = (result["macd_hist"] - float(macdh_3ago)) / result["atr"]
                    result["_macd_direction"] = (
                        "rising" if delta_norm > 0.1
                        else "falling" if delta_norm < -0.1
                        else "flat"
                    )
                else:
                    result["_macd_direction"] = None
            else:
                result["_macd_direction"] = None
        else:
            result["_macd_direction"] = None

    except Exception as e:
        logger.warning(f"stockstats indicator computation failed for {timeframe}: {e}")

    # Candlestick patterns (1h/4h only)
    if detect_patterns_flag and len(df) >= 3:
        try:
            result["patterns"] = detect_patterns(df)
        except Exception as e:
            logger.warning(f"Pattern detection failed for {timeframe}: {e}")

    coverage = n_valid / n_total if n_total > 0 else 0.0
    return result, coverage


# ── VWAP computation ──────────────────────────────────────────────────

def compute_vwap(candles_1m: pd.DataFrame) -> Optional[float]:
    """Compute daily-anchored VWAP from 1m candles since midnight UTC.

    Args:
        candles_1m: 1m candle DataFrame with [timestamp, open, high, low, close, volume]

    Returns:
        VWAP value or None if insufficient data.
    """
    if candles_1m.empty:
        return None

    now_utc = datetime.now(timezone.utc)
    midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    today = candles_1m[candles_1m["timestamp"] >= pd.Timestamp(midnight.replace(tzinfo=None))].copy()
    if today.empty or today["volume"].sum() < 1e-10:
        return None

    typical_price = (today["high"] + today["low"] + today["close"]) / 3
    vwap = (typical_price * today["volume"]).sum() / today["volume"].sum()
    return float(vwap)


def compute_vwap_from_slice(candles_1m: pd.DataFrame, ts: datetime) -> Optional[float]:
    """Compute VWAP anchored at midnight UTC of the given timestamp (for backtest)."""
    if candles_1m.empty:
        return None
    midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    today = candles_1m[
        (candles_1m["timestamp"] >= pd.Timestamp(midnight.replace(tzinfo=None)))
        & (candles_1m["timestamp"] <= pd.Timestamp(ts.replace(tzinfo=None) if hasattr(ts, 'replace') else ts))
    ]
    if today.empty or today["volume"].sum() < 1e-10:
        return None
    typical_price = (today["high"] + today["low"] + today["close"]) / 3
    return float((typical_price * today["volume"]).sum() / today["volume"].sum())


# ── Volatility flag ───────────────────────────────────────────────────

def compute_volatility_flag(candles_1m: pd.DataFrame, lookback: int = 5) -> float:
    """Max absolute % move in the last N one-minute candles."""
    if candles_1m.empty or len(candles_1m) < 2:
        return 0.0
    recent = candles_1m.tail(lookback + 1)
    if len(recent) < 2:
        return 0.0
    closes = recent["close"].values
    pct_moves = [abs((closes[i] - closes[i - 1]) / closes[i - 1]) * 100
                 for i in range(1, len(closes)) if closes[i - 1] > 0]
    return max(pct_moves) if pct_moves else 0.0


# ── Live pulse report builder ─────────────────────────────────────────

def _read_last_pulse(jsonl_path: Path) -> Optional[dict]:
    """Read the last line of a pulse JSONL file for cold-start funding delta."""
    if not jsonl_path.exists():
        return None
    try:
        last_line = None
        with open(jsonl_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        if last_line:
            return json.loads(last_line)
    except Exception:
        pass
    return None


def build_pulse_report(
    ticker: str,
    results_dir: str = "./eval_results",
) -> dict:
    """Build a multi-timeframe pulse report from live Hyperliquid data.

    Args:
        ticker: e.g. "BTC-USD"
        results_dir: Directory for pulse JSONL files (cold-start reads)

    Returns:
        Pulse report dict matching the plan's output shape.
    """
    base_asset = ticker.replace("-USD", "").replace("USDT", "").upper()
    hl = HyperliquidClient()
    now_utc = datetime.now(timezone.utc)

    # Fetch candles for each timeframe with appropriate cache TTL
    # Per-TF closed-bar rule: strict (drop in-progress) for 1h/4h,
    # partial-allowed (keep last bar, flag it) for 1m/5m/15m.
    STRICT_CLOSED_BAR_TFS = {"1h", "4h"}
    candles: Dict[str, pd.DataFrame] = {}
    partial_bar_flags: Dict[str, bool] = {tf: False for tf in PULSE_TIMEFRAMES}
    for tf, cfg in PULSE_TIMEFRAMES.items():
        n_candles = cfg["candles"]
        interval_sec = _INTERVAL_SECONDS.get(tf, 60)
        lookback_sec = n_candles * interval_sec
        start_dt = now_utc - timedelta(seconds=lookback_sec + interval_sec * 2)
        try:
            df = hl.get_ohlcv(
                base_asset, tf,
                start=start_dt.strftime("%Y-%m-%d"),
                end=(now_utc + timedelta(days=1)).strftime("%Y-%m-%d"),
                max_age_override=PULSE_CACHE_TTL.get(tf, 600),
            )
            if df.empty:
                candles[tf] = df
                continue

            # Closed-bar check: is the last bar's close ≤ now?
            ts_last = df["timestamp"].iloc[-1]
            if hasattr(ts_last, "to_pydatetime"):
                ts_last_utc = ts_last.to_pydatetime()
                if ts_last_utc.tzinfo is None:
                    ts_last_utc = ts_last_utc.replace(tzinfo=timezone.utc)
            else:
                ts_last_utc = now_utc

            bar_end = ts_last_utc + timedelta(seconds=interval_sec)
            if bar_end > now_utc:
                # Last bar not yet closed
                if tf in STRICT_CLOSED_BAR_TFS:
                    # Drop in-progress bar
                    df = df.iloc[:-1]
                else:
                    # Keep partial bar but flag it
                    partial_bar_flags[tf] = True
            candles[tf] = df.tail(n_candles) if not df.empty else df
        except Exception as e:
            logger.warning(f"Failed to fetch {tf} candles for {ticker}: {e}")
            candles[tf] = pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

    # Compute indicators per timeframe
    timeframes = {}
    coverages = {}
    for tf, cfg in PULSE_TIMEFRAMES.items():
        indicators, cov = _compute_tf_indicators(
            candles[tf], tf, detect_patterns_flag=cfg["pattern"]
        )
        timeframes[tf] = indicators
        coverages[tf] = cov

    # Spot price
    spot_price = hl.get_spot_price(base_asset, max_age_override=60)
    if spot_price is None and not candles["1m"].empty:
        spot_price = float(candles["1m"]["close"].iloc[-1])

    # VWAP
    vwap_daily = compute_vwap(candles["1m"])
    vwap_position = 0
    if vwap_daily is not None and spot_price is not None:
        vwap_position = 1 if spot_price > vwap_daily else -1

    # Order flow / derivatives context
    premium_pct = 0.0
    funding_rate = None
    funding_delta = None
    funding_acceleration = None
    oi_notional = None
    day_volume = None

    ctx = hl.get_asset_context(base_asset, max_age_override=120)
    if ctx is not None:
        premium_pct = ctx.get("premium", 0) * 100
        funding_rate = ctx.get("funding", 0)
        oi_notional = ctx.get("openInterest", 0) * ctx.get("markPx", 0)
        day_volume = ctx.get("dayNtlVlm", 0)

    # Funding delta (hourly — from funding history)
    try:
        end_str = now_utc.strftime("%Y-%m-%d")
        start_str = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
        funding_df = hl.get_funding_history(
            base_asset, start=start_str, end=end_str,
            max_age_override=300,
        )
        if not funding_df.empty and len(funding_df) >= 2:
            rates = funding_df["funding_rate"].values
            funding_delta = float(rates[-1] - rates[-2])
            if len(rates) >= 3:
                prev_delta = float(rates[-2] - rates[-3])
                funding_acceleration = float(funding_delta - prev_delta)
            if funding_rate is None:
                funding_rate = float(rates[-1])
    except Exception as e:
        logger.warning(f"Funding history fetch failed for {ticker}: {e}")

    # Cold-start: if no funding_delta, try reading from last pulse
    if funding_delta is None:
        pulse_path = Path(results_dir) / "pulse" / ticker / "pulse.jsonl"
        last_pulse = _read_last_pulse(pulse_path)
        if last_pulse and funding_rate is not None:
            prev_rate = last_pulse.get("funding_rate")
            if prev_rate is not None:
                funding_delta = funding_rate - prev_rate

    # Volatility flag
    max_1m_move_pct = compute_volatility_flag(candles["1m"])

    # Overall coverage
    total_cov = sum(coverages.values()) / len(coverages) if coverages else 0.0

    return {
        "ticker": ticker,
        "timestamp": now_utc.isoformat(),
        "spot_price": spot_price,
        "vwap_daily": vwap_daily,
        "vwap_position": vwap_position,
        "premium_pct": premium_pct,
        "funding_rate": funding_rate,
        "funding_delta": funding_delta,
        "funding_acceleration": funding_acceleration,
        "oi_notional": oi_notional,
        "day_volume": day_volume,
        "max_1m_move_pct": max_1m_move_pct,
        "timeframes": timeframes,
        "partial_bar_flags": partial_bar_flags,
        "_coverages": coverages,
        "_overall_coverage": total_cov,
    }
