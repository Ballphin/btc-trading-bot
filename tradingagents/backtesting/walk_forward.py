"""Walk-forward cross-validation using existing analysis logs.

Instead of re-running the full LLM pipeline (expensive, slow), this module:
1. Loads existing `full_states_log_*.json` files from past analyses
2. Extracts the decision (signal, confidence, stop_loss, take_profit)
3. Scores against actual future prices at T+7d
4. Computes walk-forward metrics including the Deflated Sharpe Ratio

The Deflated Sharpe Ratio (DSR) corrects for non-normality, sample length,
and multiple testing per Bailey & López de Prado (2014).
"""

import json
import math
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict

import yfinance as yf

from tradingagents.backtesting.date_utils import parse_any_date as _parse_any_date
from tradingagents.backtesting.scorecard import (
    _get_ohlc_range,
    _scan_sl_tp_hits,
    _estimate_execution_costs,
    CALIBRATION_HORIZON_DAYS,
)

logger = logging.getLogger(__name__)

# Primary scoring horizon (T+7d) — per Ardia et al. (2019)
_PRIMARY_HORIZON = 7


def _get_price_on_date(ticker: str, date_str: str) -> Optional[float]:
    """Fetch closing price for a ticker on a specific date/datetime.
    
    If date_str contains a time component (e.g. '2026-04-08T16:00'), fetches
    1-hour interval data and returns the close of that specific hour.
    This prevents mixing intraday entry prices against daily closes.
    """
    try:
        # Detect intraday timestamps (contain 'T' or ' HH:MM')
        has_time = "T" in date_str or (" " in date_str and ":" in date_str.split(" ", 1)[-1])
        
        if has_time:
            # Parse the logical candle time
            clean = date_str.replace("T", " ").replace(":", ":").strip()
            # Handle formats like '2026-04-08T16' or '2026-04-08T16:00'
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H"):
                try:
                    dt = datetime.strptime(clean, fmt)
                    break
                except ValueError:
                    continue
            else:
                dt = datetime.strptime(clean.split()[0], "%Y-%m-%d")
                has_time = False
            
            if has_time:
                # Fetch 1H data around the target hour
                start = dt.strftime("%Y-%m-%d")
                end = (dt + timedelta(days=2)).strftime("%Y-%m-%d")
                data = yf.download(ticker, start=start, end=end, interval="1h",
                                   progress=False, auto_adjust=True)
                if data.empty:
                    return None
                # Find the closest hourly close at or after the target time
                target_ts = dt
                for idx in data.index:
                    idx_naive = idx.replace(tzinfo=None) if hasattr(idx, 'tzinfo') else idx
                    if idx_naive >= target_ts:
                        val = data["Close"].loc[idx]
                        return float(val.item() if hasattr(val, "item") else val)
                # Fallback: last available
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


def compute_deflated_sharpe(
    sharpe: float,
    n_periods: int,
    n_strategies: int = 1,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute the Deflated Sharpe Ratio per Bailey & López de Prado (2014).

    The DSR corrects for:
    - Non-normality (skewness and excess kurtosis)
    - Sample length
    - Number of strategies tested (multiple testing correction)

    Args:
        sharpe: Observed Sharpe ratio
        n_periods: Number of observation periods
        n_strategies: Number of independently tested strategies (default 1 for single pipeline)
        skew: Skewness of returns
        kurtosis: Kurtosis of returns (3.0 = normal)

    Returns:
        DSR probability (0.0-1.0). > 0.95 = genuine edge, < 0.50 = likely noise
    """
    if n_periods <= 1 or n_strategies < 1:
        return 0.0

    try:
        from scipy import stats as scipy_stats
    except ImportError:
        # Fallback: approximate using normal CDF
        se = math.sqrt(max(1, n_periods))
        return 0.5 + 0.5 * math.erf(sharpe * se / math.sqrt(2))

    # Standard error of the Sharpe ratio (Lo, 2002)
    excess_kurtosis = kurtosis - 3.0
    se_sharpe = math.sqrt(
        (1 + 0.5 * sharpe ** 2 - skew * sharpe + (excess_kurtosis / 4) * sharpe ** 2)
        / n_periods
    )

    if se_sharpe <= 0:
        return 0.0

    # Expected maximum Sharpe under the null (Euler-Mascheroni correction)
    if n_strategies > 1:
        euler_mascheroni = 0.5772156649
        max_expected = math.sqrt(2 * math.log(n_strategies)) * (
            1 - euler_mascheroni / (2 * math.log(n_strategies))
        )
    else:
        max_expected = 0.0

    # Deflated Sharpe = P(SR > max_expected | observed)
    dsr = scipy_stats.norm.cdf((sharpe - max_expected) / se_sharpe)
    return round(dsr, 6)


class WalkForwardValidator:
    """Walk-forward cross-validation using existing analysis logs."""

    def __init__(
        self,
        ticker: str,
        results_dir: str = "./eval_results",
    ):
        self.ticker = ticker
        self.results_dir = Path(results_dir)

    def _load_decisions_from_logs(self) -> List[Dict]:
        """Load decisions from full_states_log files.
        
        Supports both daily files (full_states_log_YYYY-MM-DD.json) and
        intraday files (full_states_log_YYYY-MM-DDTHH.json).
        """
        decisions = []
        # FIX: logs are inside TradingAgentsStrategy_logs/ subdirectory
        log_dir = self.results_dir / self.ticker / "TradingAgentsStrategy_logs"

        if not log_dir.exists():
            logger.warning(f"No logs directory for {self.ticker} at {log_dir}")
            return decisions

        for log_file in sorted(log_dir.glob("full_states_log_*.json")):
            try:
                with open(log_file, "r") as f:
                    data = json.load(f)

                # Extract date/timestamp from filename
                # Handles: full_states_log_2026-04-08.json
                #      and: full_states_log_2026-04-08T16.json
                date_str = log_file.stem.replace("full_states_log_", "")

                # FIX: log files use the date as the top-level key
                # e.g. { "2026-04-08": { "final_trade_decision": "...", ... } }
                # Find the date-keyed sub-object (key may match date_str or date portion)
                date_match = re.match(r'(\d{4}-\d{2}-\d{2})', date_str)
                date_key = date_match.group(1) if date_match else date_str.split("T")[0]
                date_data = data.get(date_str) or data.get(date_key) or data

                # Parse signal from final_trade_decision
                decision_text = date_data.get("final_trade_decision", "")
                signal = self._parse_signal(decision_text)
                if not signal:
                    # Also try direct structured signal field
                    signal = date_data.get("signal")
                if not signal:
                    continue

                # Extract confidence (may be nested in date_data or top-level)
                confidence = date_data.get("confidence", 0.5)
                if isinstance(confidence, str):
                    try:
                        confidence = float(confidence)
                    except ValueError:
                        confidence = 0.5
                confidence = float(confidence) if confidence else 0.5

                # Extract stop/take from nested data
                stop_loss = date_data.get("stop_loss_price")
                take_profit = date_data.get("take_profit_price")

                # Fetch entry price at the logical timestamp
                price = date_data.get("entry_price")
                if not price:
                    price = _get_price_on_date(self.ticker, date_str)

                if not price or price <= 0:
                    continue

                # Extract max_hold_days from decision text
                max_hold = date_data.get("max_hold_days")
                if max_hold is None:
                    hold_match = re.search(r'"max_hold_days"\s*:\s*(\d+)', decision_text)
                    if hold_match:
                        max_hold = min(int(hold_match.group(1)), 7)  # cap crypto

                position_size = date_data.get("position_size_pct")

                decisions.append({
                    "date": date_str,
                    "signal": signal,
                    "confidence": confidence,
                    "price": price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "max_hold_days": max_hold,
                    "position_size_pct": position_size,
                    "source": "analysis_log",
                })
            except Exception as e:
                logger.debug(f"Failed to parse {log_file}: {e}")
                continue

        # Also load from shadow decisions (scorecard)
        shadow_file = self.results_dir / "shadow" / self.ticker / "decisions_scored.jsonl"
        if shadow_file.exists():
            with open(shadow_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            if d.get("scored"):
                                decisions.append(d)
                        except (json.JSONDecodeError, KeyError):
                            continue

        # Deduplicate by date
        seen = set()
        unique = []
        for d in decisions:
            date_key = d.get("date", "")
            if date_key not in seen:
                seen.add(date_key)
                unique.append(d)

        return sorted(unique, key=lambda x: x.get("date", ""))

    def _parse_signal(self, decision_text: str) -> Optional[str]:
        """Parse signal from decision text."""
        if not decision_text:
            return None
        text_upper = decision_text.upper()
        for signal in ["OVERWEIGHT", "UNDERWEIGHT", "SHORT", "COVER", "BUY", "SELL", "HOLD"]:
            if signal in text_upper:
                return signal
        return None

    def validate(
        self,
        horizon_days: int = _PRIMARY_HORIZON,
    ) -> Dict[str, Any]:
        """Run walk-forward validation with adaptive per-trade scoring.

        Uses the same three-tier scoring as score_pending_decisions:
        1. SL/TP hit scan  2. Hold-period timeout  3. 7d fallback

        Includes execution cost deduction and position-sized equity curve.

        Args:
            horizon_days: Default scoring horizon when max_hold_days is missing

        Returns:
            Comprehensive validation report with gross/net metrics
        """
        decisions = self._load_decisions_from_logs()

        if not decisions:
            return {"error": "No decisions found in logs", "decisions": 0}

        # Score each decision
        scored = []
        gross_returns = []
        net_returns = []
        position_sizes = []
        exit_type_counts = defaultdict(int)

        for d in decisions:
            date_str = d["date"].split(" ")[0]
            dt = _parse_any_date(date_str)
            price = d["price"]
            signal = d.get("signal", "HOLD")

            if signal == "HOLD":
                continue  # not scorable

            # Adaptive hold period per decision
            hold_days = d.get("max_hold_days")
            if hold_days is None or hold_days <= 0:
                hold_days = horizon_days  # 7d fallback

            # Use pre-scored adaptive data if available
            if "was_correct_primary" in d:
                scored.append(d)
                gr = d.get("actual_return_primary", d.get(f"actual_return_{horizon_days}d", 0))
                nr = d.get("net_return_primary", gr)
                gross_returns.append(gr)
                net_returns.append(nr)
                ps = d.get("position_size_pct", 0.05) or 0.05
                position_sizes.append(ps)
                exit_type_counts[d.get("exit_type", "legacy_7d")] += 1
                continue

            # Legacy pre-scored (T+Nd close) — use as-is
            key_nd = f"was_correct_{horizon_days}d"
            if key_nd in d:
                scored.append(d)
                ret = d.get(f"actual_return_{horizon_days}d", 0)
                cost = _estimate_execution_costs(signal, hold_days, self.ticker)
                gross_returns.append(ret)
                net_returns.append(ret - cost)
                ps = d.get("position_size_pct", 0.05) or 0.05
                position_sizes.append(ps)
                exit_type_counts["legacy_7d"] += 1
                continue

            # Fresh scoring: three-tier adaptive
            ohlc_start = dt.strftime("%Y-%m-%d")
            ohlc_end = (dt + timedelta(days=hold_days + 3)).strftime("%Y-%m-%d")
            ohlc_df = _get_ohlc_range(self.ticker, ohlc_start, ohlc_end)

            stop_loss = d.get("stop_loss")
            take_profit = d.get("take_profit")

            exit_result = _scan_sl_tp_hits(
                ohlc_df, price, signal, stop_loss, take_profit, hold_days, dt
            )

            if not exit_result:
                continue

            # Execution cost deduction
            actual_hold = exit_result.get("exit_day", hold_days)
            exec_cost = _estimate_execution_costs(signal, actual_hold, self.ticker)
            gross_ret = exit_result["actual_return"]
            net_ret = gross_ret - exec_cost

            # Store results back into decision dict
            d["was_correct_primary"] = exit_result["was_correct"]
            d["actual_return_primary"] = round(gross_ret, 6)
            d["net_return_primary"] = round(net_ret, 6)
            d["exit_type"] = exit_result["exit_type"]
            d["exit_price"] = exit_result["exit_price"]
            d["exit_day"] = exit_result["exit_day"]
            d["hold_days_planned"] = hold_days
            d["execution_cost"] = round(exec_cost, 6)

            scored.append(d)
            gross_returns.append(gross_ret)
            net_returns.append(net_ret)
            ps = d.get("position_size_pct", 0.05) or 0.05
            position_sizes.append(ps)
            exit_type_counts[exit_result["exit_type"]] += 1

        if not scored:
            return {"error": "No scorable decisions", "decisions": len(decisions)}

        # Compute metrics
        n = len(scored)

        def _was_correct(d):
            if "was_correct_primary" in d:
                return bool(d["was_correct_primary"])
            return bool(d.get(f"was_correct_{horizon_days}d", False))

        correct = sum(1 for d in scored if _was_correct(d))
        win_rate = correct / n

        # Returns-based metrics (use net returns for Sharpe/DSR)
        if net_returns:
            mean_gross = sum(gross_returns) / len(gross_returns)
            mean_net = sum(net_returns) / len(net_returns)
            std_net = math.sqrt(
                sum((r - mean_net) ** 2 for r in net_returns) / max(len(net_returns) - 1, 1)
            )
            std_gross = math.sqrt(
                sum((r - mean_gross) ** 2 for r in gross_returns) / max(len(gross_returns) - 1, 1)
            )
            sharpe_net = mean_net / std_net if std_net > 0 else 0
            sharpe_gross = mean_gross / std_gross if std_gross > 0 else 0

            # Sharpe standard error
            sharpe_se = math.sqrt((1 + 0.5 * sharpe_net ** 2) / n) if n > 1 else float("inf")

            # Skewness and kurtosis for DSR
            if len(net_returns) >= 4 and std_net > 0:
                skew = sum((r - mean_net) ** 3 for r in net_returns) / (n * std_net ** 3)
                kurtosis = sum((r - mean_net) ** 4 for r in net_returns) / (n * std_net ** 4)
            else:
                skew = 0
                kurtosis = 3

            # DSR on net returns
            dsr = compute_deflated_sharpe(sharpe_net, n, n_strategies=1, skew=skew, kurtosis=kurtosis)

            # Position-sized equity curve (portfolio-weighted)
            equity_gross = [1.0]
            equity_position = [1.0]
            for i, gr in enumerate(gross_returns):
                equity_gross.append(equity_gross[-1] * (1 + gr))
                ps = position_sizes[i] if i < len(position_sizes) else 0.05
                equity_position.append(equity_position[-1] * (1 + net_returns[i] * ps))

            # Max drawdown (on position-sized equity)
            peak = equity_position[0]
            max_dd = 0
            for val in equity_position:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            mean_gross = mean_net = 0
            std_net = std_gross = 0
            sharpe_net = sharpe_gross = 0
            sharpe_se = float("inf")
            dsr = 0
            equity_gross = []
            equity_position = []
            max_dd = 0
            skew = 0
            kurtosis = 3

        # EV per trade (on $10K notional)
        notional = 10000.0
        wins_ret = [r for r, d in zip(net_returns, scored) if _was_correct(d)]
        losses_ret = [abs(r) for r, d in zip(net_returns, scored) if not _was_correct(d)]
        avg_win = sum(wins_ret) / len(wins_ret) if wins_ret else 0
        avg_loss = sum(losses_ret) / len(losses_ret) if losses_ret else 0
        ev_per_trade = round(
            (win_rate * avg_win - (1 - win_rate) * avg_loss) * notional, 2
        )

        # Regime breakdown
        regime_stats = defaultdict(lambda: {"correct": 0, "total": 0, "returns": []})
        for i, d in enumerate(scored):
            regime = d.get("regime", "unknown")
            regime_stats[regime]["total"] += 1
            if _was_correct(d):
                regime_stats[regime]["correct"] += 1
            regime_stats[regime]["returns"].append(
                net_returns[i] if i < len(net_returns) else 0
            )

        regime_analysis = {}
        for regime, stats in regime_stats.items():
            regime_analysis[regime] = {
                "win_rate": round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0,
                "sample_size": stats["total"],
                "mean_return": round(sum(stats["returns"]) / len(stats["returns"]), 6) if stats["returns"] else 0,
            }

        # Signal breakdown
        signal_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for d in scored:
            sig = d.get("signal", "")
            signal_stats[sig]["total"] += 1
            if _was_correct(d):
                signal_stats[sig]["correct"] += 1

        signal_analysis = {
            sig: {
                "win_rate": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0,
                "sample_size": s["total"],
            }
            for sig, s in signal_stats.items()
        }

        # DSR interpretation with n<30 guard
        if n < 30:
            dsr_interpretation = (
                f"INSUFFICIENT DATA \u2014 need \u226530 scored decisions for statistical validity (have {n})"
            )
        elif dsr > 0.95:
            dsr_interpretation = "GENUINE EDGE \u2014 statistically significant at 95% confidence"
        elif dsr > 0.80:
            dsr_interpretation = "PROMISING \u2014 approaching significance, need more data"
        elif dsr > 0.50:
            dsr_interpretation = "INCONCLUSIVE \u2014 cannot distinguish from noise"
        else:
            dsr_interpretation = "LIKELY NOISE \u2014 returns indistinguishable from random"

        return {
            "ticker": self.ticker,
            "horizon_days": horizon_days,
            "total_decisions": len(decisions),
            "scored_decisions": n,
            "overall_metrics": {
                "win_rate": round(win_rate, 4),
                "mean_return_gross": round(mean_gross, 6),
                "mean_return_net": round(mean_net, 6),
                "std_return": round(std_net, 6),
                "sharpe_ratio_gross": round(sharpe_gross, 4),
                "sharpe_ratio_net": round(sharpe_net, 4),
                "sharpe_se": round(sharpe_se, 4) if sharpe_se != float("inf") else None,
                "deflated_sharpe_ratio": round(dsr, 4),
                "dsr_interpretation": dsr_interpretation,
                "max_drawdown": round(max_dd, 4),
                "skewness": round(skew, 4),
                "kurtosis": round(kurtosis, 4),
                "n_strategies_tested": 1,
                "ev_per_trade_10k": ev_per_trade,
                "avg_win_return": round(avg_win, 6),
                "avg_loss_return": round(avg_loss, 6),
            },
            "exit_type_breakdown": dict(exit_type_counts),
            "regime_analysis": regime_analysis,
            "signal_analysis": signal_analysis,
            "equity_curve_gross": [round(e, 6) for e in equity_gross[:100]],
            "equity_curve_position": [round(e, 6) for e in equity_position[:100]],
            "validated_at": datetime.now().isoformat(),
        }
