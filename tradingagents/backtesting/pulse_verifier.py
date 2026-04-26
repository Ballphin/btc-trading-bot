"""Pulse Verifier — deterministic verification of pulse outcomes using
Hyperliquid candle data.

Centralizes forward-hit threshold computation and per-pulse verification
so that both the live scorer (server.py) and the backtest engine
(pulse_backtest.py) produce identical results.

Candle strategy (per adversarial review):
  - 1m candles for ALL window high/low and SL/TP touch ordering
  - 5m candles for forward exit price (first 5m open at or after target_ts)
  - Funding cost included when horizon window crosses a funding stamp

VERIFICATION_VERSION must be bumped whenever verification logic changes
to invalidate cached backtest results.
"""

import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bump when verification logic changes to invalidate caches.
VERIFICATION_VERSION = 1

HORIZONS = [
    ("+5m", 5),
    ("+15m", 15),
    ("+1h", 60),
]

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "eval_results" / "pulse_verified"

# Funding stamps fire every 8h at 00:00, 08:00, 16:00 UTC.
_FUNDING_STAMPS_HOURS = [0, 8, 16]


# ── Helpers ────────────────────────────────────────────────────────────

def _to_utc_aware(ts: Any) -> pd.Timestamp:
    """Normalize a scalar datetime-like to UTC-aware pd.Timestamp."""
    out = pd.Timestamp(ts)
    if out.tz is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _ensure_utc_aware(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    """Return df with *column* guaranteed UTC-aware. Raises on tz-naive input."""
    if df is None or df.empty or column not in df.columns:
        return df
    col = df[column]
    if hasattr(col.dt, "tz") and col.dt.tz is not None:
        return df
    out = df.copy()
    out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    out = out.dropna(subset=[column]).sort_values(column).reset_index(drop=True)
    return out


def pulse_id(ticker: str, ts: str, signal: str, entry_price: float) -> str:
    """Deterministic hash for a pulse."""
    raw = f"{ticker}|{ts}|{signal}|{entry_price}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Unified forward-hit threshold ──────────────────────────────────────

def forward_hit_threshold(
    atr_1h_at_pulse: Optional[float],
    entry_price: float,
    horizon_minutes: int,
    cfg: Any = None,
) -> float:
    """Compute the forward-hit threshold for a given horizon.

    Uses ATR-sqrt-time when ATR is available, otherwise falls back to
    fixed bps from config. This is the SINGLE source of truth — both
    server.py and pulse_backtest.py must call this.

    Args:
        atr_1h_at_pulse: 1h ATR value at pulse time (dollar terms).
        entry_price: Pulse entry price.
        horizon_minutes: Forward horizon in minutes (5, 15, or 60).
        cfg: PulseConfig instance or None for defaults.

    Returns:
        Threshold as a fraction (e.g. 0.001 = 10 bps).
    """
    atr_mul = 0.5
    fallback_bps = [5, 10, 15]

    if cfg is not None:
        atr_mul = float(cfg.get("forward_return", "atr_multiplier", default=0.5))
        fb = cfg.get("forward_return", "fallback_fixed_bps", default=[5, 10, 15])
        if fb:
            fallback_bps = fb

    if atr_1h_at_pulse and atr_1h_at_pulse > 0 and entry_price > 0:
        return atr_mul * float(atr_1h_at_pulse) * math.sqrt(horizon_minutes / 60.0) / entry_price

    idx = {5: 0, 15: 1, 60: 2}.get(horizon_minutes, 2)
    return fallback_bps[min(idx, len(fallback_bps) - 1)] / 10_000.0


# ── Funding cost helper ────────────────────────────────────────────────

def _funding_cost_in_window(
    funding_df: Optional[pd.DataFrame],
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    direction: int,
) -> float:
    """Sum funding payments crossed between entry_ts and exit_ts.

    Convention: funding_rate > 0 means longs pay shorts.
    Returns cost as a fraction to be subtracted from net P&L.
    """
    if funding_df is None or funding_df.empty:
        return 0.0
    df = _ensure_utc_aware(funding_df)
    if df.empty:
        return 0.0
    mask = (df["timestamp"] > entry_ts) & (df["timestamp"] <= exit_ts)
    crossed = df.loc[mask, "funding_rate"]
    if crossed.empty:
        return 0.0
    return float(crossed.sum()) * direction


# ── Core verification ──────────────────────────────────────────────────

@dataclass
class HorizonResult:
    """Verification result for a single forward horizon."""
    exit_price: Optional[float] = None
    raw_return: Optional[float] = None
    net_return: Optional[float] = None
    threshold: Optional[float] = None
    hit: Optional[bool] = None
    window_high: Optional[float] = None
    window_low: Optional[float] = None
    tp_touched: Optional[bool] = None
    sl_touched: Optional[bool] = None
    window_candle_count: int = 0
    window_expected: int = 0
    window_complete: bool = False


@dataclass
class VerifiedOutcome:
    """Full verification result for a single pulse."""
    pulse_id: str
    ticker: str
    ts: str
    signal: str
    entry_price: float
    confidence: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    hold_minutes: int = 45

    # Per-horizon results
    fwd_5m: Optional[HorizonResult] = None
    fwd_15m: Optional[HorizonResult] = None
    fwd_1h: Optional[HorizonResult] = None

    # Hold-period SL/TP outcome
    exit_type: str = "insufficient_data"
    exit_ts: Optional[str] = None
    exit_price: Optional[float] = None
    exit_return: Optional[float] = None
    hold_window_high: Optional[float] = None
    hold_window_low: Optional[float] = None

    # Metadata
    sl_atr_ratio: Optional[float] = None
    regime_mode: Optional[str] = None
    data_source: str = "hyperliquid"
    verified_at: Optional[str] = None
    verification_version: int = VERIFICATION_VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("fwd_5m", "fwd_15m", "fwd_1h"):
            val = d.get(key)
            if val is None:
                d[key] = asdict(HorizonResult())
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VerifiedOutcome":
        for key in ("fwd_5m", "fwd_15m", "fwd_1h"):
            val = d.get(key)
            if isinstance(val, dict):
                d[key] = HorizonResult(**val)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def verify_single_pulse(
    pulse: dict,
    candles_1m: pd.DataFrame,
    candles_5m: pd.DataFrame,
    cfg: Any = None,
    exec_cost: float = 0.0005,
    funding_df: Optional[pd.DataFrame] = None,
) -> VerifiedOutcome:
    """Verify a single pulse against candle data.

    Args:
        pulse: Dict with keys: ticker, ts, signal, price, stop_loss, take_profit,
               hold_minutes, confidence, atr_1h_at_pulse, regime_snapshot.
        candles_1m: UTC-aware 1m candle DataFrame.
        candles_5m: UTC-aware 5m candle DataFrame.
        cfg: PulseConfig or None.
        exec_cost: Round-trip execution cost as fraction.
        funding_df: Optional funding rate DataFrame.

    Returns:
        VerifiedOutcome with all fields populated.
    """
    ticker = pulse.get("ticker", "")
    ts_str = pulse.get("ts", "")
    signal = pulse.get("signal", "")
    entry_price = pulse.get("price", 0.0)

    outcome = VerifiedOutcome(
        pulse_id=pulse_id(ticker, ts_str, signal, entry_price),
        ticker=ticker,
        ts=ts_str,
        signal=signal,
        entry_price=entry_price,
        confidence=pulse.get("confidence"),
        stop_loss=pulse.get("stop_loss"),
        take_profit=pulse.get("take_profit"),
        hold_minutes=pulse.get("hold_minutes", 45),
        regime_mode=None,
        verified_at=datetime.now(timezone.utc).isoformat(),
        verification_version=VERIFICATION_VERSION,
    )

    # Extract regime mode if available
    regime = pulse.get("regime_snapshot")
    if isinstance(regime, dict):
        outcome.regime_mode = regime.get("mode")

    if not entry_price or entry_price <= 0 or signal not in ("BUY", "SHORT"):
        return outcome

    direction = 1 if signal == "BUY" else -1

    try:
        pulse_ts = _to_utc_aware(ts_str)
    except Exception:
        return outcome

    atr_1h = pulse.get("atr_1h_at_pulse")

    # Compute SL/ATR ratio
    sl = outcome.stop_loss
    if sl is not None and atr_1h and atr_1h > 0:
        outcome.sl_atr_ratio = round(abs(sl - entry_price) / atr_1h, 3)

    # ── Forward horizons (1m for window evidence, 5m for exit price) ──
    for label, minutes in HORIZONS:
        target_ts = pulse_ts + timedelta(minutes=minutes)
        expected_candles = minutes  # 1m candles expected in window

        hr = HorizonResult(window_expected=expected_candles)

        # Window high/low from 1m candles
        if not candles_1m.empty:
            mask_w = (candles_1m["timestamp"] > pulse_ts) & (candles_1m["timestamp"] <= target_ts)
            w_candles = candles_1m[mask_w]
            hr.window_candle_count = len(w_candles)
            hr.window_complete = hr.window_candle_count >= max(1, expected_candles - 1)

            if not w_candles.empty:
                hr.window_high = round(float(w_candles["high"].max()), 2)
                hr.window_low = round(float(w_candles["low"].min()), 2)

                # TP/SL touched within this horizon window
                tp = outcome.take_profit
                if tp is not None:
                    if direction == 1:
                        hr.tp_touched = bool(hr.window_high >= tp)
                    else:
                        hr.tp_touched = bool(hr.window_low <= tp)

                if sl is not None:
                    if direction == 1:
                        hr.sl_touched = bool(hr.window_low <= sl)
                    else:
                        hr.sl_touched = bool(hr.window_high >= sl)

        # Forward exit price from 5m candles
        if not candles_5m.empty:
            mask_fwd = candles_5m["timestamp"] >= target_ts
            if mask_fwd.any():
                fwd_candle = candles_5m[mask_fwd].iloc[0]
                hr.exit_price = round(float(fwd_candle["open"]), 2)

                raw_ret = (hr.exit_price - entry_price) / entry_price * direction
                hr.raw_return = round(raw_ret, 6)

                funding_cost = _funding_cost_in_window(
                    funding_df, pulse_ts, target_ts, direction,
                )
                hr.net_return = round(raw_ret - exec_cost - funding_cost, 6)

                hr.threshold = round(
                    forward_hit_threshold(atr_1h, entry_price, minutes, cfg), 6,
                )
                hr.hit = hr.net_return >= hr.threshold

        if label == "+5m":
            outcome.fwd_5m = hr
        elif label == "+15m":
            outcome.fwd_15m = hr
        else:
            outcome.fwd_1h = hr

    # ── Hold-period SL/TP scan (1m candles) ────────────────────────────
    hold_min = outcome.hold_minutes
    hold_end = pulse_ts + timedelta(minutes=hold_min)

    if candles_1m.empty:
        return outcome

    mask_hold = (candles_1m["timestamp"] > pulse_ts) & (candles_1m["timestamp"] <= hold_end)
    hold_candles = candles_1m[mask_hold]

    if hold_candles.empty:
        outcome.exit_type = "insufficient_data"
        return outcome

    outcome.hold_window_high = round(float(hold_candles["high"].max()), 2)
    outcome.hold_window_low = round(float(hold_candles["low"].min()), 2)

    tp = outcome.take_profit
    sl = outcome.stop_loss

    if tp is None or sl is None:
        outcome.exit_type = "missing_sltp"
        if not hold_candles.empty:
            last = hold_candles.iloc[-1]
            exit_p = float(last["close"])
            raw = (exit_p - entry_price) / entry_price * direction
            funding_cost = _funding_cost_in_window(
                funding_df, pulse_ts,
                _to_utc_aware(last["timestamp"]),
                direction,
            )
            outcome.exit_return = round(raw - exec_cost - funding_cost, 6)
            outcome.exit_price = round(exit_p, 2)
            outcome.exit_ts = str(last["timestamp"])
        return outcome

    # Iterate 1m candles to find first SL or TP trigger
    outcome.exit_type = "timeout"
    for _, c in hold_candles.iterrows():
        c_ts = _to_utc_aware(c["timestamp"])
        if direction == 1:  # BUY
            if c["low"] <= sl:
                outcome.exit_type = "sl_hit"
                funding_cost = _funding_cost_in_window(funding_df, pulse_ts, c_ts, direction)
                outcome.exit_return = round((sl - entry_price) / entry_price - exec_cost - funding_cost, 6)
                outcome.exit_price = round(float(sl), 2)
                outcome.exit_ts = str(c_ts)
                break
            if c["high"] >= tp:
                outcome.exit_type = "tp_hit"
                funding_cost = _funding_cost_in_window(funding_df, pulse_ts, c_ts, direction)
                outcome.exit_return = round((tp - entry_price) / entry_price - exec_cost - funding_cost, 6)
                outcome.exit_price = round(float(tp), 2)
                outcome.exit_ts = str(c_ts)
                break
        else:  # SHORT
            if c["high"] >= sl:
                outcome.exit_type = "sl_hit"
                funding_cost = _funding_cost_in_window(funding_df, pulse_ts, c_ts, direction)
                outcome.exit_return = round((entry_price - sl) / entry_price - exec_cost - funding_cost, 6)
                outcome.exit_price = round(float(sl), 2)
                outcome.exit_ts = str(c_ts)
                break
            if c["low"] <= tp:
                outcome.exit_type = "tp_hit"
                funding_cost = _funding_cost_in_window(funding_df, pulse_ts, c_ts, direction)
                outcome.exit_return = round((entry_price - tp) / entry_price - exec_cost - funding_cost, 6)
                outcome.exit_price = round(float(tp), 2)
                outcome.exit_ts = str(c_ts)
                break

    if outcome.exit_type == "timeout":
        last = hold_candles.iloc[-1]
        exit_p = float(last["close"])
        raw = (exit_p - entry_price) / entry_price * direction
        last_ts = _to_utc_aware(last["timestamp"])
        funding_cost = _funding_cost_in_window(funding_df, pulse_ts, last_ts, direction)
        outcome.exit_return = round(raw - exec_cost - funding_cost, 6)
        outcome.exit_price = round(exit_p, 2)
        outcome.exit_ts = str(last_ts)

    return outcome


def verify_pulses(
    pulses: List[dict],
    candles_1m: pd.DataFrame,
    candles_5m: pd.DataFrame,
    cfg: Any = None,
    exec_cost: float = 0.0005,
    funding_df: Optional[pd.DataFrame] = None,
) -> List[VerifiedOutcome]:
    """Verify a batch of pulses.

    Candle DataFrames must be UTC-aware and cover the full range of all pulses
    plus the maximum horizon (+1h).
    """
    candles_1m = _ensure_utc_aware(candles_1m)
    candles_5m = _ensure_utc_aware(candles_5m)

    results = []
    for p in pulses:
        if p.get("signal") not in ("BUY", "SHORT"):
            continue
        outcome = verify_single_pulse(p, candles_1m, candles_5m, cfg, exec_cost, funding_df)
        results.append(outcome)
    return results


# ── Deduplication ──────────────────────────────────────────────────────

def dedup_signals(signals: List[dict], hold_minutes_default: int = 45) -> List[dict]:
    """Remove duplicate same-direction signals within hold_minutes of each other.

    Keeps the first signal in each cluster. This matches the dedup logic
    in pulse_backtest.py but is needed for live-history paths that skip it.
    """
    if not signals:
        return signals

    sorted_sigs = sorted(signals, key=lambda s: s.get("ts", ""))
    result = []
    last_by_dir: Dict[str, datetime] = {}  # "BUY" / "SHORT" -> last kept ts

    for sig in sorted_sigs:
        signal = sig.get("signal")
        if signal not in ("BUY", "SHORT"):
            result.append(sig)
            continue

        try:
            ts = _to_utc_aware(sig.get("ts", "")).to_pydatetime()
        except Exception:
            result.append(sig)
            continue

        hold_min = sig.get("hold_minutes", hold_minutes_default)
        last_ts = last_by_dir.get(signal)

        if last_ts is not None and (ts - last_ts).total_seconds() < hold_min * 60:
            continue  # suppress duplicate

        last_by_dir[signal] = ts
        result.append(sig)

    return result


# ── Persistence ────────────────────────────────────────────────────────

def _outcomes_path(ticker: str) -> Path:
    return EVAL_DIR / ticker.upper() / "outcomes.jsonl"


def _cache_dir(ticker: str) -> Path:
    return EVAL_DIR / ticker.upper() / "backtests"


def cache_key(
    ticker: str,
    start_date: str,
    end_date: str,
    confidence_100_only: bool = False,
) -> str:
    raw = f"{ticker}|{start_date}|{end_date}|{confidence_100_only}|{VERIFICATION_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def save_verified_outcomes(ticker: str, outcomes: List[VerifiedOutcome]) -> Path:
    """Persist verified outcomes to JSONL (append-only, last-entry-wins on read)."""
    path = _outcomes_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".tmp")
    with open(tmp, "a") as f:
        for o in outcomes:
            f.write(json.dumps(o.to_dict()) + "\n")
    # Append to existing file
    if path.exists():
        with open(path, "a") as dst, open(tmp, "r") as src:
            dst.write(src.read())
        tmp.unlink()
    else:
        os.replace(str(tmp), str(path))
    return path


def load_verified_outcomes(ticker: str) -> Dict[str, VerifiedOutcome]:
    """Load verified outcomes from JSONL. Last entry wins for duplicate pulse_ids."""
    path = _outcomes_path(ticker)
    if not path.exists():
        return {}

    results: Dict[str, VerifiedOutcome] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                vo = VerifiedOutcome.from_dict(d)
                results[vo.pulse_id] = vo
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning("Skipping malformed verified outcome line: %s", e)
    return results


def save_backtest_cache(ticker: str, key: str, result: dict) -> Path:
    """Save a full backtest result to the cache."""
    d = _cache_dir(ticker)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(str(tmp), str(path))
    return path


def load_backtest_cache(ticker: str, key: str) -> Optional[dict]:
    """Load a cached backtest result, or None if not found / stale."""
    path = _cache_dir(ticker) / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("verification_version") != VERIFICATION_VERSION:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


# ── Hit-rate computation with correct denominators ─────────────────────

def compute_hit_rates(
    outcomes: List[VerifiedOutcome],
) -> Dict[str, dict]:
    """Compute hit rates with correct denominators (complete windows only).

    Returns dict keyed by horizon label with:
        overall, BUY, SHORT, n_complete, n_total, ci_95
    """
    rates = {}
    for label, _ in HORIZONS:
        attr = f"fwd_{label.replace('+', '').replace('m', 'm').replace('h', 'h')}"
        # Map label to attribute name
        attr_map = {"+5m": "fwd_5m", "+15m": "fwd_15m", "+1h": "fwd_1h"}
        attr = attr_map[label]

        all_hrs = [(o, getattr(o, attr)) for o in outcomes if getattr(o, attr) is not None]
        complete = [(o, hr) for o, hr in all_hrs if hr.window_complete and hr.hit is not None]

        def _rate(subset):
            if not subset:
                return 0.0, 0, 0.0
            hits = sum(1 for _, hr in subset if hr.hit)
            n = len(subset)
            p = hits / n
            ci = 1.96 * math.sqrt(p * (1 - p) / n) if n > 0 else 0.0
            return round(p, 4), n, round(ci, 4)

        buy_complete = [(o, hr) for o, hr in complete if o.signal == "BUY"]
        short_complete = [(o, hr) for o, hr in complete if o.signal == "SHORT"]

        overall_rate, n_complete, ci = _rate(complete)
        buy_rate, n_buy, _ = _rate(buy_complete)
        short_rate, n_short, _ = _rate(short_complete)

        rates[label] = {
            "overall": overall_rate,
            "BUY": buy_rate,
            "SHORT": short_rate,
            "n_complete": n_complete,
            "n_total": len(all_hrs),
            "n_buy": n_buy,
            "n_short": n_short,
            "ci_95": ci,
        }

    return rates
