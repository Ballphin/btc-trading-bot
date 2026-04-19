"""Forward-test scorecard for tracking and scoring live analysis decisions.

Records every live analysis decision and scores them against actual future
prices to build a continuously updating truth table. Uses T+7d as the
exclusive calibration horizon (per Ardia et al., 2019 — serial correlation
invalidates shorter horizons for independent-samples statistics).

Brier Decomposition follows Wolfers & Zitzewitz (2006):
    BS = Reliability - Resolution + Uncertainty
    - Reliability (lower = better calibrated)
    - Resolution  (higher = more informative)
    - Uncertainty  (constant for a given base rate)
"""

import json
import math
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict

import yfinance as yf

from tradingagents.backtesting.date_utils import parse_any_date

logger = logging.getLogger(__name__)

# Signals that make a directional prediction (can be scored)
_DIRECTIONAL_SIGNALS = frozenset({"BUY", "SELL", "SHORT", "COVER"})

# Scoring horizons — T+7d is used for calibration; others are informational
CALIBRATION_HORIZON_DAYS = 7
INFO_HORIZONS_DAYS = [1, 3]


def _get_price_on_date(ticker: str, date_str: str) -> Optional[float]:
    """Fetch closing price for a ticker on a specific date/datetime.
    
    If date_str contains a time component (e.g. '2026-04-08T16:00'), fetches
    1-hour interval data and returns the close of that specific hour.
    """
    try:
        has_time = "T" in date_str or (" " in date_str and ":" in date_str.split(" ", 1)[-1])
        
        if has_time:
            clean = date_str.replace("T", " ").strip()
            dt = None
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H"):
                try:
                    dt = datetime.strptime(clean, fmt)
                    break
                except ValueError:
                    continue
            if dt:
                start = dt.strftime("%Y-%m-%d")
                end = (dt + timedelta(days=2)).strftime("%Y-%m-%d")
                data = yf.download(ticker, start=start, end=end, interval="1h",
                                   progress=False, auto_adjust=True)
                if not data.empty:
                    for idx in data.index:
                        idx_naive = idx.replace(tzinfo=None) if hasattr(idx, 'tzinfo') else idx
                        if idx_naive >= dt:
                            val = data["Close"].loc[idx]
                            return float(val.item() if hasattr(val, "item") else val)
                    val = data["Close"].iloc[-1]
                    return float(val.item() if hasattr(val, "item") else val)

        # Daily fallback
        date_only = date_str.split("T")[0].split(" ")[0]
        dt = datetime.strptime(date_only, "%Y-%m-%d")
        start = dt.strftime("%Y-%m-%d")
        end = (dt + timedelta(days=5)).strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            return None
        return float(data["Close"].iloc[0].item() if hasattr(data["Close"].iloc[0], "item") else data["Close"].iloc[0])
    except Exception as e:
        logger.debug(f"Price fetch failed for {ticker} on {date_str}: {e}")
        return None



def _get_ohlc_range(ticker: str, start_date: str, end_date: str) -> Optional[Any]:
    """Fetch daily OHLC data for a ticker over a date range (single yfinance call).

    Returns a DataFrame with columns [Open, High, Low, Close] indexed by date,
    or None if no data is available.
    """
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            return None
        return data
    except Exception as e:
        logger.debug(f"OHLC range fetch failed for {ticker} ({start_date}→{end_date}): {e}")
        return None


def _scan_sl_tp_hits(
    ohlc_df,
    entry_price: float,
    signal: str,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    hold_days: int,
    entry_dt: datetime,
) -> Dict[str, Any]:
    """Three-tier scoring: SL/TP hit scan → hold-period timeout → 7d fallback.

    Scans daily OHLC from T+1 to T+hold_days for stop-loss or take-profit hits.
    First trigger wins.  If neither triggers, scores at T+hold_days close.
    If hold_days data is unavailable, falls back to T+7d close.

    Returns dict with:
        exit_type: "stop_loss_hit" | "take_profit_hit" | "held_to_expiry"
        exit_price: float
        exit_day: int (days after entry)
        was_correct: bool
        actual_return: float (signed, from entry perspective)
    """
    sig = signal.upper()
    is_long = sig in ("BUY", "COVER")

    # Filter OHLC to the relevant window (T+1 onwards, up to hold_days)
    if ohlc_df is None or ohlc_df.empty:
        return {}

    # Normalize index to naive datetimes for comparison
    rows = []
    for idx in ohlc_df.index:
        idx_naive = idx.replace(tzinfo=None) if hasattr(idx, "tzinfo") and idx.tzinfo else idx
        day_offset = (idx_naive - entry_dt).days
        if day_offset < 1:
            continue  # skip entry day
        if day_offset > hold_days:
            break
        high = ohlc_df["High"].loc[idx]
        low = ohlc_df["Low"].loc[idx]
        close = ohlc_df["Close"].loc[idx]
        rows.append({
            "day": day_offset,
            "high": float(high.item() if hasattr(high, "item") else high),
            "low": float(low.item() if hasattr(low, "item") else low),
            "close": float(close.item() if hasattr(close, "item") else close),
        })

    if not rows:
        return {}

    # Tier 1: Scan for SL/TP hits (first trigger wins)
    has_sl = stop_loss is not None and stop_loss > 0
    has_tp = take_profit is not None and take_profit > 0

    if has_sl or has_tp:
        for row in rows:
            if is_long:
                # BUY/COVER: SL triggers when low ≤ stop_loss
                if has_sl and row["low"] <= stop_loss:
                    ret = (stop_loss - entry_price) / entry_price
                    return {
                        "exit_type": "stop_loss_hit",
                        "exit_price": stop_loss,
                        "exit_day": row["day"],
                        "was_correct": False,
                        "actual_return": round(ret, 6),
                    }
                # BUY/COVER: TP triggers when high ≥ take_profit
                if has_tp and row["high"] >= take_profit:
                    ret = (take_profit - entry_price) / entry_price
                    return {
                        "exit_type": "take_profit_hit",
                        "exit_price": take_profit,
                        "exit_day": row["day"],
                        "was_correct": True,
                        "actual_return": round(ret, 6),
                    }
            else:
                # SHORT/SELL: SL triggers when high ≥ stop_loss
                if has_sl and row["high"] >= stop_loss:
                    ret = (entry_price - stop_loss) / entry_price
                    return {
                        "exit_type": "stop_loss_hit",
                        "exit_price": stop_loss,
                        "exit_day": row["day"],
                        "was_correct": False,
                        "actual_return": round(ret, 6),
                    }
                # SHORT/SELL: TP triggers when low ≤ take_profit
                if has_tp and row["low"] <= take_profit:
                    ret = (entry_price - take_profit) / entry_price
                    return {
                        "exit_type": "take_profit_hit",
                        "exit_price": take_profit,
                        "exit_day": row["day"],
                        "was_correct": True,
                        "actual_return": round(ret, 6),
                    }

    # Tier 2: Hold-period timeout — use T+hold_days close
    # Find the row closest to hold_days
    last_row = rows[-1]
    exit_price = last_row["close"]
    if is_long:
        was_correct = exit_price > entry_price
        ret = (exit_price - entry_price) / entry_price
    else:
        was_correct = exit_price < entry_price
        ret = (entry_price - exit_price) / entry_price

    return {
        "exit_type": "held_to_expiry",
        "exit_price": round(exit_price, 6),
        "exit_day": last_row["day"],
        "was_correct": was_correct,
        "actual_return": round(ret, 6),
    }


# Execution cost constants (configurable)
_SPREAD_BPS = 10  # round-trip spread in basis points
_FUNDING_RATE_PER_8H = 0.0001  # 1 bps per 8h funding interval (shorts only)


def _estimate_execution_costs(
    signal: str, hold_days: int, ticker: str
) -> float:
    """Estimate execution costs: spread + funding for crypto shorts.

    Returns cost as a positive fraction (e.g. 0.0031 = 0.31%).
    """
    spread_cost = _SPREAD_BPS / 10000.0  # e.g. 10 bps = 0.001
    is_crypto = "-USD" in ticker.upper() or "USDT" in ticker.upper()
    sig = signal.upper()

    if is_crypto and sig in ("SHORT", "SELL"):
        # 3 funding intervals per day × hold_days
        funding_cost = hold_days * 3 * _FUNDING_RATE_PER_8H
        return spread_cost + funding_cost

    return spread_cost


def record_decision(
    ticker: str,
    date: str,
    signal: str,
    price: float,
    confidence: float,
    reasoning: str = "",
    regime: str = "unknown",
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    results_dir: str = "./eval_results",
) -> str:
    """Record a live analysis decision for later scoring.

    Args:
        ticker: Ticker symbol
        date: Decision date (YYYY-MM-DD)
        signal: Signal type (BUY, SELL, SHORT, etc.)
        price: Entry price at time of decision
        confidence: LLM confidence score (0.0-1.0)
        reasoning: LLM reasoning text
        regime: Market regime at decision time
        stop_loss: Stop loss price
        take_profit: Take profit price
        results_dir: Results directory

    Returns:
        Path to the JSONL file
    """
    shadow_dir = Path(results_dir) / "shadow" / ticker
    shadow_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "ticker": ticker,
        "date": date,
        "signal": signal.upper(),
        "price": price,
        "confidence": confidence,
        "reasoning": reasoning,
        "regime": regime,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "source": "live_analysis",
        "recorded_at": datetime.now().isoformat(),
        "scored": False,
    }

    jsonl_path = shadow_dir / "decisions.jsonl"
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    logger.info(f"Recorded decision: {signal} {ticker} @ ${price:.2f} (conf={confidence:.2f})")
    return str(jsonl_path)


def score_pending_decisions(
    ticker: str,
    results_dir: str = "./eval_results",
) -> Dict[str, Any]:
    """Score un-scored directional decisions using adaptive per-trade horizons.

    Three-tier scoring per decision:
    1. SL/TP hit scan — check daily OHLC from T+1 to T+hold_days for stop/target hits
    2. Hold-period timeout — if no SL/TP hit, score at T+hold_days close
    3. 7d fallback — if max_hold_days is missing, use T+7d close (legacy behaviour)

    Also deducts estimated execution costs (spread + funding for crypto shorts).

    Idempotent: keyed by (ticker, date, signal) — re-runs skip already-scored entries.
    """
    jsonl_path = Path(results_dir) / "shadow" / ticker / "decisions.jsonl"
    scored_path = Path(results_dir) / "shadow" / ticker / "decisions_scored.jsonl"

    if not jsonl_path.exists():
        return {"error": "No decisions found", "scored": 0}

    # Load all decisions
    decisions = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    decisions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Load already-scored keys for idempotency
    scored_keys = set()
    if scored_path.exists():
        with open(scored_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        scored_keys.add((d["ticker"], d["date"], d["signal"]))
                    except (json.JSONDecodeError, KeyError):
                        continue

    scored_count = 0
    newly_scored = []

    for decision in decisions:
        key = (decision["ticker"], decision["date"], decision["signal"])
        signal = decision.get("signal", "").upper()
        price = decision.get("price")

        # Skip already scored, non-directional signals, and invalid prices
        if key in scored_keys:
            continue
        if signal not in _DIRECTIONAL_SIGNALS:
            continue
        if not price or price <= 0:
            logger.warning(f"Skipping decision with invalid price: {decision.get('date')}")
            continue

        try:
            dt = parse_any_date(decision["date"])
        except ValueError:
            logger.warning(f"Skipping decision with unparseable date: {decision.get('date')}")
            continue

        # Adaptive hold period: use per-trade max_hold_days, fallback to 7
        hold_days = decision.get("max_hold_days")
        if hold_days is None or hold_days <= 0:
            hold_days = CALIBRATION_HORIZON_DAYS  # 7d fallback

        # Check if enough time has passed for this decision's hold period
        if datetime.now() - dt < timedelta(days=hold_days + 1):
            continue

        stop_loss = decision.get("stop_loss")
        take_profit = decision.get("take_profit")

        # Batch fetch OHLC for the entire hold period (single yfinance call)
        ohlc_start = dt.strftime("%Y-%m-%d")
        ohlc_end = (dt + timedelta(days=hold_days + 3)).strftime("%Y-%m-%d")
        ohlc_df = _get_ohlc_range(ticker, ohlc_start, ohlc_end)

        # Three-tier scoring
        exit_result = _scan_sl_tp_hits(
            ohlc_df, price, signal, stop_loss, take_profit, hold_days, dt
        )

        if not exit_result:
            # No OHLC data available — skip for now
            continue

        # Execution cost deduction
        actual_hold = exit_result.get("exit_day", hold_days)
        exec_cost = _estimate_execution_costs(signal, actual_hold, ticker)
        gross_return = exit_result["actual_return"]
        net_return = gross_return - exec_cost

        # Also fetch informational horizons (T+1d, T+3d) for the dashboard
        scores = {}
        for horizon in INFO_HORIZONS_DAYS:
            target_date = (dt + timedelta(days=horizon)).strftime("%Y-%m-%d")
            actual_price = _get_price_on_date(ticker, target_date)
            if actual_price is None:
                continue
            if signal in ("BUY", "COVER"):
                was_correct = actual_price > price
                actual_return = (actual_price - price) / price
            else:
                was_correct = actual_price < price
                actual_return = (price - actual_price) / price
            scores[f"was_correct_{horizon}d"] = was_correct
            scores[f"actual_return_{horizon}d"] = round(actual_return, 6)
            scores[f"actual_price_{horizon}d"] = actual_price

        # Compute Brier score using the adaptive exit result
        confidence = decision.get("confidence", 0.5)
        outcome = 1.0 if exit_result["was_correct"] else 0.0
        brier = (confidence - outcome) ** 2

        scored_entry = {
            **decision,
            **scores,
            # Primary scoring result (adaptive)
            "was_correct_primary": exit_result["was_correct"],
            "actual_return_primary": round(gross_return, 6),
            "net_return_primary": round(net_return, 6),
            "exit_type": exit_result["exit_type"],
            "exit_price": exit_result["exit_price"],
            "exit_day": exit_result["exit_day"],
            "hold_days_planned": hold_days,
            "execution_cost": round(exec_cost, 6),
            # Legacy compatibility fields (T+7d equivalent)
            f"was_correct_{hold_days}d": exit_result["was_correct"],
            f"actual_return_{hold_days}d": round(gross_return, 6),
            f"actual_price_{hold_days}d": exit_result["exit_price"],
            "brier_score": round(brier, 6),
            "scored": True,
            "scored_at": datetime.now().isoformat(),
        }

        newly_scored.append(scored_entry)
        scored_keys.add(key)
        scored_count += 1

    # Append newly scored entries
    if newly_scored:
        with open(scored_path, "a") as f:
            for entry in newly_scored:
                f.write(json.dumps(entry, default=str) + "\n")

    return {
        "scored": scored_count,
        "total_decisions": len(decisions),
        "total_scored": len(scored_keys),
        "pending": len(decisions) - len(scored_keys),
    }


def compute_brier_decomposition(
    ticker: str,
    results_dir: str = "./eval_results",
) -> Dict[str, Any]:
    """Compute Brier score decomposition (Reliability + Resolution + Uncertainty).

    Uses adaptive bins: 3 bins (<100 decisions) or 10 bins (>=100 decisions).
    Bin edges adjusted for LLM confidence distribution which clusters around 0.55-0.75.

    Args:
        ticker: Ticker symbol
        results_dir: Results directory

    Returns:
        Dict with decomposition components and per-bin data
    """
    scored_path = Path(results_dir) / "shadow" / ticker / "decisions_scored.jsonl"
    if not scored_path.exists():
        return {"error": "No scored decisions available"}

    scored = []
    with open(scored_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    if d.get("scored") and d.get("confidence") is not None:
                        scored.append(d)
                except (json.JSONDecodeError, KeyError):
                    continue

    if len(scored) < 5:
        return {"error": f"Need at least 5 scored decisions, have {len(scored)}"}

    N = len(scored)

    # Adaptive bin edges based on sample size and LLM confidence distribution
    if N < 100:
        bin_edges = [0.0, 0.55, 0.70, 1.01]  # 3 bins: low / medium / high
    else:
        bin_edges = [i / 10 for i in range(11)]  # standard 10 bins
        bin_edges[-1] = 1.01  # include 1.0

    # Compute base rate (overall fraction of correct predictions)
    # Use adaptive was_correct_primary, fallback to legacy was_correct_7d
    def _was_correct(d: dict) -> bool:
        if "was_correct_primary" in d:
            return bool(d["was_correct_primary"])
        return bool(d.get(f"was_correct_{CALIBRATION_HORIZON_DAYS}d", False))

    outcomes = [1.0 if _was_correct(d) else 0.0 for d in scored]
    base_rate = sum(outcomes) / N

    # Bin the decisions
    bins = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bin_decisions = [
            d for d in scored
            if lo <= d.get("confidence", 0) < hi
        ]
        if not bin_decisions:
            continue

        n_k = len(bin_decisions)
        mean_conf = sum(d["confidence"] for d in bin_decisions) / n_k
        mean_outcome = sum(
            1.0 if _was_correct(d) else 0.0 for d in bin_decisions
        ) / n_k

        bins.append({
            "range": f"{lo:.2f}-{hi:.2f}",
            "n": n_k,
            "mean_confidence": round(mean_conf, 4),
            "mean_outcome": round(mean_outcome, 4),
        })

    # Compute decomposition
    reliability = sum(
        b["n"] / N * (b["mean_confidence"] - b["mean_outcome"]) ** 2
        for b in bins
    )
    resolution = sum(
        b["n"] / N * (b["mean_outcome"] - base_rate) ** 2
        for b in bins
    )
    uncertainty = base_rate * (1 - base_rate)
    brier = reliability - resolution + uncertainty

    return {
        "brier_score": round(brier, 6),
        "reliability": round(reliability, 6),
        "resolution": round(resolution, 6),
        "uncertainty": round(uncertainty, 6),
        "base_rate": round(base_rate, 4),
        "n_decisions": N,
        "n_bins": len(bins),
        "bins": bins,
        "calibration_trigger": {
            "dampen": reliability > 0.05,
            "allow_larger": resolution > 0.10,
        },
    }


def get_scorecard(
    ticker: str,
    results_dir: str = "./eval_results",
) -> Dict[str, Any]:
    """Get comprehensive scorecard for a ticker.

    Returns aggregate stats: win rates by signal and regime, Brier decomposition,
    and individual decision history.

    Args:
        ticker: Ticker symbol
        results_dir: Results directory

    Returns:
        Complete scorecard dict
    """
    scored_path = Path(results_dir) / "shadow" / ticker / "decisions_scored.jsonl"
    jsonl_path = Path(results_dir) / "shadow" / ticker / "decisions.jsonl"

    # Count total decisions
    total = 0
    if jsonl_path.exists():
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    total += 1

    if not scored_path.exists():
        return {
            "ticker": ticker,
            "total_decisions": total,
            "scored_decisions": 0,
            "win_rates": {},
            "brier_decomposition": None,
        }

    scored = []
    with open(scored_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    scored.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not scored:
        return {
            "ticker": ticker,
            "total_decisions": total,
            "scored_decisions": 0,
            "win_rates": {},
            "brier_decomposition": None,
        }

    key_7d = f"was_correct_{CALIBRATION_HORIZON_DAYS}d"

    # Adaptive correctness: use was_correct_primary, fallback to legacy was_correct_7d
    def _was_correct(d: dict):
        if "was_correct_primary" in d:
            return bool(d["was_correct_primary"])
        if d.get(key_7d) is not None:
            return bool(d[key_7d])
        return None

    # Win rate by signal type
    signal_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for d in scored:
        sig = d.get("signal", "")
        wc = _was_correct(d)
        if wc is not None:
            signal_stats[sig]["total"] += 1
            if wc:
                signal_stats[sig]["correct"] += 1

    win_by_signal = {
        sig: {
            "win_rate": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0,
            "sample_size": s["total"],
        }
        for sig, s in signal_stats.items()
    }

    # Win rate by regime
    regime_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for d in scored:
        regime = d.get("regime", "unknown")
        wc = _was_correct(d)
        if wc is not None:
            regime_stats[regime]["total"] += 1
            if wc:
                regime_stats[regime]["correct"] += 1

    win_by_regime = {
        regime: {
            "win_rate": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0,
            "sample_size": s["total"],
        }
        for regime, s in regime_stats.items()
    }

    # Win rate by signal+regime combo
    combo_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for d in scored:
        combo = f"{d.get('signal', '')}_{d.get('regime', 'unknown')}"
        wc = _was_correct(d)
        if wc is not None:
            combo_stats[combo]["total"] += 1
            if wc:
                combo_stats[combo]["correct"] += 1

    win_by_combo = {
        combo: {
            "win_rate": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0,
            "sample_size": s["total"],
        }
        for combo, s in combo_stats.items()
        if s["total"] >= 3  # only show combos with enough data
    }

    # Overall win rate
    total_correct = sum(1 for d in scored if _was_correct(d))
    overall_win_rate = total_correct / len(scored) if scored else 0

    # Average Brier score
    brier_scores = [d.get("brier_score", 0) for d in scored if d.get("brier_score") is not None]
    avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else None

    # Exit type breakdown
    exit_types = defaultdict(int)
    for d in scored:
        exit_types[d.get("exit_type", "legacy_7d")] += 1

    # EV per trade (on $10K notional)
    notional = 10000.0
    wins = [d for d in scored if _was_correct(d)]
    losses = [d for d in scored if _was_correct(d) is False]
    avg_win_ret = sum(
        d.get("net_return_primary", d.get(f"actual_return_{CALIBRATION_HORIZON_DAYS}d", 0))
        for d in wins
    ) / len(wins) if wins else 0
    avg_loss_ret = sum(
        abs(d.get("net_return_primary", d.get(f"actual_return_{CALIBRATION_HORIZON_DAYS}d", 0)))
        for d in losses
    ) / len(losses) if losses else 0
    win_rate_f = len(wins) / len(scored) if scored else 0
    loss_rate_f = len(losses) / len(scored) if scored else 0
    ev_per_trade = round((win_rate_f * avg_win_ret - loss_rate_f * avg_loss_ret) * notional, 2)

    # Brier decomposition
    brier_decomp = compute_brier_decomposition(ticker, results_dir)

    return {
        "ticker": ticker,
        "total_decisions": total,
        "scored_decisions": len(scored),
        "overall_win_rate": round(overall_win_rate, 4),
        "avg_brier_score": round(avg_brier, 4) if avg_brier is not None else None,
        "win_by_signal": win_by_signal,
        "win_by_regime": win_by_regime,
        "win_by_combo": win_by_combo,
        "exit_type_breakdown": dict(exit_types),
        "ev_per_trade_10k": ev_per_trade,
        "avg_win_return": round(avg_win_ret, 6),
        "avg_loss_return": round(avg_loss_ret, 6),
        "brier_decomposition": brier_decomp if "error" not in brier_decomp else None,
        "recent_decisions": scored[-20:],  # last 20 scored decisions
    }


def count_scored_decisions(
    ticker: str,
    results_dir: str = "./eval_results",
) -> int:
    """Count total scored decisions for a ticker."""
    scored_path = Path(results_dir) / "shadow" / ticker / "decisions_scored.jsonl"
    if not scored_path.exists():
        return 0
    count = 0
    with open(scored_path, "r") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def run_calibration_study(
    ticker: str,
    min_decisions: int = 10,
    min_regimes: int = 3,
    results_dir: str = "./eval_results",
) -> Dict[str, Any]:
    """Run a 10-decision calibration study to compute the overconfidence correction.

    Protocol:
    1. Load scored decisions with price > 0
    2. If < min_decisions: supplement with historical full_states_log files
    3. Compute correction = mean(actual_outcome) / mean(confidence), clamp [0.60, 0.95]
    4. Tag with coverage_quality based on regime diversity

    Args:
        ticker: Ticker symbol
        min_decisions: Minimum scored decisions required
        min_regimes: Minimum number of regimes for "high" coverage quality
        results_dir: Results directory

    Returns:
        Calibration result dict with correction factor
    """
    scored_path = Path(results_dir) / "shadow" / ticker / "decisions_scored.jsonl"

    scored = []
    if scored_path.exists():
        with open(scored_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        if d.get("scored") and d.get("price") and d["price"] > 0:
                            scored.append(d)
                    except (json.JSONDecodeError, KeyError):
                        continue

    if len(scored) < min_decisions:
        return {
            "error": f"Need at least {min_decisions} scored decisions, have {len(scored)}",
            "scored_available": len(scored),
        }

    # Use ALL scored decisions (no truncation — Bayesian shrinkage handles small-n)

    # Deduplicate per-(ticker, candle_time) — NOT per-day — to preserve each 4H
    # boundary as an independent observation. Per-day dedup collapsed 6 decisions/day
    # into 1 and silently biased calibration toward whichever confidence "won" the
    # day. See plan Part 2 BLOCKER #2.
    by_key: Dict[str, Any] = {}
    for d in scored:
        tkr = d.get("ticker", "")
        # Prefer candle_time (unique per 4H boundary); fall back to date if missing
        candle = d.get("candle_time") or d.get("date", "")
        key = f"{tkr}|{candle}"
        existing = by_key.get(key)
        if existing is None or d.get("confidence", 0) > existing.get("confidence", 0):
            by_key[key] = d
    deduped = list(by_key.values())

    key_7d = f"was_correct_{CALIBRATION_HORIZON_DAYS}d"

    # Adaptive correctness: use was_correct_primary, fallback to legacy
    def _cal_was_correct(d: dict) -> bool:
        if "was_correct_primary" in d:
            return bool(d["was_correct_primary"])
        return bool(d.get(key_7d, False))

    # Compute correction factor using deduped sample
    confidences = [d.get("confidence", 0.5) for d in deduped]
    outcomes = [1.0 if _cal_was_correct(d) else 0.0 for d in deduped]

    mean_confidence = sum(confidences) / len(confidences)
    mean_outcome = sum(outcomes) / len(outcomes)

    if mean_confidence > 0:
        data_correction = mean_outcome / mean_confidence
    else:
        data_correction = 0.85  # fallback

    # Bayesian shrinkage: blend data with prior at low sample sizes
    # w ramps from 0.3 at n=10 to 1.0 at n≥30
    prior_correction = 0.85
    n = len(deduped)
    w = min(1.0, max(0.3, n / 30.0))
    correction = (1.0 - w) * prior_correction + w * data_correction

    # Clamp to safe range (allow up to 1.0 so perfect track record removes penalty)
    correction = max(0.60, min(1.0, correction))

    # Assess regime coverage
    regimes = set(d.get("regime", "unknown") for d in scored)
    regimes.discard("unknown")
    n_regimes = len(regimes)

    if n_regimes >= 5:
        coverage_quality = "high"
    elif n_regimes >= min_regimes:
        coverage_quality = "medium"
    else:
        coverage_quality = "low"

    result = {
        "ticker": ticker,
        "correction": round(correction, 4),
        "mean_confidence": round(mean_confidence, 4),
        "mean_outcome": round(mean_outcome, 4),
        "n_decisions_total": len(scored),
        "n_decisions_deduped": n,
        "regimes_covered": list(regimes),
        "coverage_quality": coverage_quality,
        "computed_at": datetime.now().isoformat(),
        "note": (
            "Calibration based on limited regime coverage; will auto-update as more regimes are observed"
            if coverage_quality == "low"
            else "Calibration has adequate regime diversity"
        ),
    }

    # Save calibration result
    cal_path = Path(results_dir) / "shadow" / ticker / "calibration.json"
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Calibration study for {ticker}: correction={correction:.4f}, quality={coverage_quality}")
    return result
