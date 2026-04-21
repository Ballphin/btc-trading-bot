"""Pulse ensemble verifier — R.3.

Auto-verifies non-NEUTRAL pulse signals by simulating the trade on
1-minute OHLC until SL/TP is touched or ``hold_minutes`` expires.
Outputs are funding-aware (``net_return_pct`` = gross − fees − funding)
and flash-crash-safe (fill prices clamped to ``entry ± 2×ATR_5m`` even
when 1m wicks print worse).

Design contract (from the post-debate plan):

* **``_resolve_intrabar_fill`` is a pure function** — bar in, tuple out.
  Exhaustive unit tests in ``tests/test_pulse/test_intrabar_fill.py``.
* **SL wins in same-bar straddles** (conservative). Real production
  stops usually trigger before TP limit orders on volatile bars, and
  the verifier's job is to measure signal quality, not to reward
  optimistic assumptions.
* **Funding is deducted via existing ``_get_funding_on_date`` helper**
  in ``tradingagents/backtesting/engine.py`` so the verifier and the
  backtest engine stay symmetric about funding accounting.
* **Day-bucketed OHLC fetch** — the scheduler groups pending pulses
  by entry-day so one `HyperliquidClient.get_ohlcv` call covers all
  configs' pending pulses from that day.
* **No lookahead** — only bars with ``ts > entry_ts`` inform the
  outcome. Enforced by slicing the OHLC frame.

The scheduler loop lives in ``server.py``; this module exposes only
pure functions + one ``process_ticker`` entry point the scheduler
calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default fee assumption: 5bps taker each side = 10bps round-trip.
# Matches the backtest default in Portfolio.taker_fee (0.05% = 5bps).
DEFAULT_FEE_BPS_PER_SIDE = 5.0

PULSE_DIR = Path("results/pulse")


# ── Pure core: intrabar fill resolution ──────────────────────────────

@dataclass
class FillResolution:
    """Outcome of one bar's SL/TP check."""
    exit_type: Optional[str]     # "sl_hit" | "tp_hit" | None
    fill_price: Optional[float]
    clamped: bool = False        # True if flash-crash slip cap engaged


def _resolve_intrabar_fill(
    bar: Any,
    entry_price: float,
    sl: float,
    tp: float,
    side: str,
    atr_5m: Optional[float],
) -> FillResolution:
    """Detect SL / TP trigger in a single OHLC bar.

    Fill price is clamped at ``entry ± 2×ATR_5m`` (the "flash-crash
    slippage cap") — a 1-minute wick far beyond this range is
    considered unexecutable at the wick price by a real market stop
    order during cascading liquidations, so the fill is capped at the
    realistic worst case. The clamp degrades gracefully: if ``atr_5m``
    is missing or zero, the raw SL/TP level is used and the result
    carries ``clamped=False`` so the caller can tag the outcome.

    Same-bar straddle (both SL and TP wicked in the same bar): **SL
    wins** by convention — see module docstring for the rationale.
    """
    slip_cap: Optional[float]
    if atr_5m is not None and atr_5m > 0:
        slip_cap = 2.0 * float(atr_5m)
    else:
        slip_cap = None

    high = float(getattr(bar, "high", getattr(bar, "h", math.nan)))
    low = float(getattr(bar, "low", getattr(bar, "l", math.nan)))

    if side == "BUY":
        hit_sl = low <= sl
        hit_tp = high >= tp
        if not (hit_sl or hit_tp):
            return FillResolution(None, None)
        if hit_sl:
            # Long stop: worst possible fill is far below entry.
            if slip_cap is not None:
                worst_allowed = entry_price - slip_cap
                fill = max(sl, worst_allowed)  # "max" = less punitive
                clamped = fill != sl
            else:
                fill, clamped = sl, False
            # SL wins ties (both wicked) — return without checking TP.
            return FillResolution("sl_hit", fill, clamped=clamped)
        # TP only
        if slip_cap is not None:
            best_allowed = entry_price + slip_cap
            fill = min(tp, best_allowed)   # positive-slippage cap (cap on
            clamped = fill != tp           # how much luck we allow)
        else:
            fill, clamped = tp, False
        return FillResolution("tp_hit", fill, clamped=clamped)
    else:  # SHORT
        hit_sl = high >= sl
        hit_tp = low <= tp
        if not (hit_sl or hit_tp):
            return FillResolution(None, None)
        if hit_sl:
            if slip_cap is not None:
                worst_allowed = entry_price + slip_cap
                fill = min(sl, worst_allowed)
                clamped = fill != sl
            else:
                fill, clamped = sl, False
            return FillResolution("sl_hit", fill, clamped=clamped)
        if slip_cap is not None:
            best_allowed = entry_price - slip_cap
            fill = max(tp, best_allowed)
            clamped = fill != tp
        else:
            fill, clamped = tp, False
        return FillResolution("tp_hit", fill, clamped=clamped)


# ── Funding accumulator ──────────────────────────────────────────────

def _accumulate_funding_cost_pct(
    ticker: str,
    side: str,
    entry_ts: datetime,
    exit_ts: datetime,
    *,
    funding_lookup: Optional[callable] = None,
) -> float:
    """Return funding cost as a signed pct-of-notional over [entry, exit].

    Longs pay positive funding; shorts receive it. We iterate the 8h
    settlement boundaries in the window and sum ``direction × rate``.

    ``funding_lookup`` is injected for testability — production callers
    bind it to ``tradingagents.backtesting.engine._get_funding_on_date``
    so the verifier and the backtest engine share the same funding
    source of truth (BLOCKER #2 in the plan).
    """
    if funding_lookup is None:
        try:
            from tradingagents.backtesting.engine import _get_funding_on_date
            funding_lookup = _get_funding_on_date
        except Exception:
            return 0.0

    direction = 1.0 if side == "BUY" else -1.0
    total = 0.0
    # Iterate the settlement times at 00/08/16 UTC within the window.
    # Start from the day's midnight so the 00/08/16 grid is hit exactly
    # — naive stepping from ``entry.replace(minute=0)`` skips the next
    # settlement when entry_ts is within the same hour as it.
    step = timedelta(hours=8)
    cur = entry_ts.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur <= entry_ts:
        cur += step
    while cur < exit_ts:
        try:
            rate = funding_lookup(ticker, cur.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception:
            rate = None
        if rate is not None:
            total += direction * float(rate)
        cur += step
    return total


# ── Outcome computation ──────────────────────────────────────────────

def _gross_return_pct(side: str, entry: float, exit_: float) -> float:
    if entry <= 0:
        return 0.0
    if side == "BUY":
        return (exit_ - entry) / entry
    return (entry - exit_) / entry


def compute_outcome(
    pulse: dict,
    ohlc_1m: pd.DataFrame,
    *,
    ticker: str,
    atr_5m: Optional[float] = None,
    fee_bps_per_side: float = DEFAULT_FEE_BPS_PER_SIDE,
    funding_lookup: Optional[callable] = None,
    now: Optional[datetime] = None,
    post_expiry_ohlc: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """Simulate one pulse against 1-minute bars and return the outcome.

    Returns ``None`` if the pulse is not yet resolvable (neither
    SL/TP/hold-expiry has been reached as of ``now``). The caller
    treats a ``None`` as "leave pending".
    """
    side = pulse.get("signal")
    if side not in ("BUY", "SHORT"):
        return None  # NEUTRAL — nothing to verify

    entry_price = float(pulse.get("price") or 0.0)
    sl = pulse.get("stop_loss")
    tp = pulse.get("take_profit")
    hold_minutes = int(pulse.get("hold_minutes") or 60)

    if entry_price <= 0 or sl is None or tp is None:
        return None

    try:
        entry_ts = _parse_iso(pulse["ts"])
    except Exception:
        return None
    expiry_ts = entry_ts + timedelta(minutes=hold_minutes)
    now = now or datetime.now(timezone.utc)

    # No-lookahead slice: only bars strictly after entry_ts matter.
    if ohlc_1m is None or ohlc_1m.empty:
        if now < expiry_ts:
            return None
        # Expired without any OHLC — can't compute outcome, skip.
        return None

    bars = _slice_after(ohlc_1m, entry_ts, expiry_ts)
    exit_type: Optional[str] = None
    exit_price: Optional[float] = None
    exit_ts: Optional[datetime] = None
    clamped = False

    for bar in bars.itertuples(index=False):
        fr = _resolve_intrabar_fill(bar, entry_price, float(sl), float(tp), side, atr_5m)
        if fr.exit_type is not None:
            exit_type = fr.exit_type
            exit_price = fr.fill_price
            exit_ts = _parse_iso(getattr(bar, "timestamp", None) or getattr(bar, "ts"))
            clamped = fr.clamped
            break

    if exit_type is None:
        # Neither SL nor TP hit within the window.
        if now < expiry_ts:
            return None  # still pending
        # Time-expiry — exit at the last bar close within the window.
        if bars.empty:
            return None
        last = bars.iloc[-1]
        exit_type = "time_expiry"
        exit_price = float(last["close"])
        exit_ts = _parse_iso(last["timestamp"]) if "timestamp" in bars.columns else expiry_ts

    gross = _gross_return_pct(side, entry_price, exit_price)

    # Fees: taker both legs, applied as negative pct of notional.
    fees_pct = -2.0 * (fee_bps_per_side / 10000.0)

    # Funding: sum over 8h settlements in [entry, exit].
    funding_pct = -_accumulate_funding_cost_pct(
        ticker, side, entry_ts, exit_ts, funding_lookup=funding_lookup,
    )
    # Note the minus: _accumulate returns signed cost (positive = paid),
    # we flip into a return-contribution for the P&L stream.

    net = gross + fees_pct + funding_pct

    # Flash-crash flag (WCT): extreme bar range vs. ATR. Computed on
    # the exit bar itself as a diagnostic for the outcome.
    flash_flag = False
    if atr_5m and atr_5m > 0:
        exit_bar = bars[bars["timestamp"] == exit_ts] if "timestamp" in bars.columns else pd.DataFrame()
        if not exit_bar.empty:
            rng = float(exit_bar.iloc[0]["high"]) - float(exit_bar.iloc[0]["low"])
            if rng > 3.0 * atr_5m:
                flash_flag = True

    # Post-expiry diagnostics (MEDIUM #12): only emitted for time_expiry.
    p10 = p30 = None
    if exit_type == "time_expiry" and post_expiry_ohlc is not None and not post_expiry_ohlc.empty:
        p10 = _post_expiry_return(side, exit_price, post_expiry_ohlc, exit_ts, 10)
        p30 = _post_expiry_return(side, exit_price, post_expiry_ohlc, exit_ts, 30)

    entry_dt = entry_ts
    return {
        "ensemble_tick_id": pulse.get("ensemble_tick_id"),
        "config_name": pulse.get("config_name", "baseline"),
        "ticker": ticker,
        "entry_ts": entry_ts.isoformat(),
        "exit_ts": exit_ts.isoformat() if isinstance(exit_ts, datetime) else str(exit_ts),
        "entry_price": entry_price,
        "exit_price": float(exit_price),
        "signal": side,
        "confidence": pulse.get("confidence"),
        "exit_type": exit_type,
        "gross_return_pct": round(gross, 6),
        "fees_pct": round(fees_pct, 6),
        "funding_cost_pct": round(funding_pct, 6),
        "net_return_pct": round(net, 6),
        "intrabar_fill_clamped": bool(clamped),
        "flash_crash_flag": bool(flash_flag),
        "post_expiry_10min_return": p10,
        "post_expiry_30min_return": p30,
        "regime_at_entry": pulse.get("regime_mode"),
        "is_weekend": entry_dt.weekday() >= 5,
        "hold_minutes_actual": int(
            (exit_ts - entry_ts).total_seconds() // 60
        ) if isinstance(exit_ts, datetime) else hold_minutes,
    }


def _post_expiry_return(
    side: str,
    exit_price: float,
    ohlc: pd.DataFrame,
    exit_ts: datetime,
    mins: int,
) -> Optional[float]:
    target = exit_ts + timedelta(minutes=mins)
    if "timestamp" not in ohlc.columns:
        return None
    future = ohlc[ohlc["timestamp"] >= target]
    if future.empty:
        return None
    p = float(future.iloc[0]["close"])
    return round(_gross_return_pct(side, exit_price, p), 6)


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_iso(s: Any) -> datetime:
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    txt = str(s).replace("Z", "+00:00")
    dt = datetime.fromisoformat(txt)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _slice_after(
    ohlc: pd.DataFrame,
    entry_ts: datetime,
    expiry_ts: datetime,
) -> pd.DataFrame:
    """Return bars whose timestamp is strictly > entry_ts and ≤ expiry_ts.

    No lookahead: a bar stamped at the exact entry_ts is excluded — by
    convention the pulse fired at that bar's close and we simulate the
    trade starting on the next bar.
    """
    if "timestamp" not in ohlc.columns:
        return ohlc.iloc[0:0]
    # Ensure tz-aware comparison.
    ts = pd.to_datetime(ohlc["timestamp"], utc=True)
    mask = (ts > pd.Timestamp(entry_ts)) & (ts <= pd.Timestamp(expiry_ts))
    out = ohlc.loc[mask].copy()
    out["timestamp"] = ts[mask].values
    return out.reset_index(drop=True)


# ── Pending-pulse loader ─────────────────────────────────────────────

def load_pending_pulses(
    ticker: str,
    *,
    pulse_dir: Path = PULSE_DIR,
    now: Optional[datetime] = None,
) -> List[dict]:
    """Return all non-NEUTRAL pulses for ``ticker`` across every config
    stream that have no matching outcome yet AND whose entry_ts is old
    enough that an outcome could in principle be computed.

    A pulse is "pending" if:
      * its ``signal`` is BUY or SHORT,
      * its ``ensemble_tick_id`` is not present in the config's
        ``outcomes.jsonl`` (match by id + config_name, since two
        configs may share a tick_id but resolve independently),
      * its ``entry_ts + min(hold_minutes, 10m)`` ≤ now (no point
        scanning pulses that literally just fired).
    """
    now = now or datetime.now(timezone.utc)
    ticker_dir = pulse_dir / ticker / "configs"
    if not ticker_dir.exists():
        return []
    pending: List[dict] = []
    for config_dir in ticker_dir.iterdir():
        if not config_dir.is_dir():
            continue
        pulse_path = config_dir / "pulse.jsonl"
        outcomes_path = config_dir / "outcomes.jsonl"
        if not pulse_path.exists():
            continue
        resolved: set = set()
        if outcomes_path.exists():
            for line in outcomes_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    o = json.loads(line)
                    tid = o.get("ensemble_tick_id")
                    if tid:
                        resolved.add(tid)
                except json.JSONDecodeError:
                    continue
        for line in pulse_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            if p.get("signal") not in ("BUY", "SHORT"):
                continue
            tid = p.get("ensemble_tick_id")
            if tid and tid in resolved:
                continue
            p.setdefault("config_name", config_dir.name)
            try:
                entry_ts = _parse_iso(p.get("ts"))
            except Exception:
                continue
            # Small ready-gate so we don't scan pulses that fired 30s ago.
            if (now - entry_ts) < timedelta(minutes=1):
                continue
            pending.append(p)
    return pending


def bucket_by_day(pulses: List[dict]) -> Dict[str, List[dict]]:
    """Group pulses by their entry UTC date — the verifier fetches 1m
    OHLC in day-wide windows (MEDIUM #11) so 10 pending pulses across
    3 days produce 3 fetch calls, regardless of how many configs those
    pulses came from."""
    out: Dict[str, List[dict]] = defaultdict(list)
    for p in pulses:
        try:
            d = _parse_iso(p.get("ts")).strftime("%Y-%m-%d")
        except Exception:
            continue
        out[d].append(p)
    return out


# ── Outcomes writer ──────────────────────────────────────────────────

def append_outcome(
    ticker: str,
    config_name: str,
    outcome: dict,
    *,
    pulse_dir: Path = PULSE_DIR,
) -> None:
    """Atomic append to ``outcomes.jsonl`` (same contract as
    ``_append_pulse`` in server.py)."""
    d = pulse_dir / ticker / "configs" / config_name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "outcomes.jsonl"
    line = json.dumps(outcome, default=str) + "\n"
    with path.open("a") as f:
        f.write(line)


# ── Scheduler entry point ────────────────────────────────────────────

def process_ticker(
    ticker: str,
    *,
    pulse_dir: Path = PULSE_DIR,
    hl_client: Any = None,
    now: Optional[datetime] = None,
) -> List[dict]:
    """Resolve all pending pulses for ``ticker``. Returns the list of
    outcomes appended this cycle.

    ``hl_client`` is a duck-typed HyperliquidClient — the scheduler
    passes the shared one from the server process. For tests we pass
    a stub.
    """
    pending = load_pending_pulses(ticker, pulse_dir=pulse_dir, now=now)
    if not pending:
        return []

    by_day = bucket_by_day(pending)
    appended: List[dict] = []
    base_asset = ticker.replace("-USD", "").replace("USDT", "").upper()

    for date_str, day_pulses in sorted(by_day.items()):
        # One OHLC fetch per day, regardless of config count.
        try:
            ohlc_1m = _fetch_1m_ohlc(hl_client, base_asset, date_str)
        except Exception as e:
            logger.warning(f"[Verifier] 1m OHLC fetch failed {ticker} {date_str}: {e}")
            continue

        # Also fetch +1 day for post-expiry diagnostics on time_expiry exits.
        post_day_str = (datetime.strptime(date_str, "%Y-%m-%d")
                        + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            post_ohlc = _fetch_1m_ohlc(hl_client, base_asset, post_day_str)
        except Exception:
            post_ohlc = None

        # Combined frame for post-expiry lookups.
        all_1m = pd.concat([ohlc_1m, post_ohlc]) if (
            ohlc_1m is not None and post_ohlc is not None
        ) else ohlc_1m

        for pulse in day_pulses:
            atr_5m = pulse.get("atr_5m_at_pulse")  # optional; falls back to None
            try:
                outcome = compute_outcome(
                    pulse,
                    ohlc_1m,
                    ticker=ticker,
                    atr_5m=atr_5m,
                    now=now,
                    post_expiry_ohlc=all_1m,
                )
            except Exception as e:
                logger.warning(f"[Verifier] compute_outcome failed {pulse.get('ensemble_tick_id')}: {e}")
                continue
            if outcome is None:
                continue
            append_outcome(
                ticker, pulse.get("config_name", "baseline"), outcome,
                pulse_dir=pulse_dir,
            )
            appended.append(outcome)
    return appended


def _fetch_1m_ohlc(hl_client: Any, base_asset: str, date_str: str) -> Optional[pd.DataFrame]:
    """Pull one day of 1m OHLC from Hyperliquid. The client already
    caches these for 30s on the live pulse path, so batch fetches by
    the verifier are free of extra network cost most of the time."""
    if hl_client is None:
        return None
    start = date_str
    end = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
           ).strftime("%Y-%m-%d")
    df = hl_client.get_ohlcv(base_asset, "1m", start=start, end=end)
    if df is None or df.empty:
        return df
    # Normalize timestamp column — HL client returns "timestamp" in the
    # OHLC frame already, but ensure it's tz-aware UTC.
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify pending ensemble pulses")
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        from tradingagents.dataflows.hyperliquid_client import HyperliquidClient
        hl = HyperliquidClient()
    except Exception:
        hl = None
    out = process_ticker(args.ticker.upper(), hl_client=hl)
    logger.info("Resolved %d outcomes for %s", len(out), args.ticker)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
