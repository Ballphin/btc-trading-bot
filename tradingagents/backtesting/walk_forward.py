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
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict

import yfinance as yf

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
                date_key = date_str.split("T")[0]  # strip hour for lookup
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

                decisions.append({
                    "date": date_str,
                    "signal": signal,
                    "confidence": confidence,
                    "price": price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
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
        """Run walk-forward validation on all available decisions.

        Args:
            horizon_days: Scoring horizon in days (default 7)

        Returns:
            Comprehensive validation report
        """
        decisions = self._load_decisions_from_logs()

        if not decisions:
            return {"error": "No decisions found in logs", "decisions": 0}

        # Score each decision
        scored = []
        returns = []

        for d in decisions:
            date_str = d["date"].split(" ")[0]
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            price = d["price"]
            signal = d.get("signal", "HOLD")

            # Check pre-scored data first
            key_7d = f"was_correct_{horizon_days}d"
            if key_7d in d:
                scored.append(d)
                ret = d.get(f"actual_return_{horizon_days}d", 0)
                returns.append(ret)
                continue

            # Need to score: fetch price at T+horizon
            target_date = (dt + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
            actual_price = _get_price_on_date(self.ticker, target_date)
            if actual_price is None:
                continue

            # Compute directional return
            if signal in ("BUY", "OVERWEIGHT", "COVER"):
                was_correct = actual_price > price
                actual_return = (actual_price - price) / price
            elif signal in ("SHORT", "SELL", "UNDERWEIGHT"):
                was_correct = actual_price < price
                actual_return = (price - actual_price) / price
            else:
                continue  # HOLD — not scorable

            d[key_7d] = was_correct
            d[f"actual_return_{horizon_days}d"] = round(actual_return, 6)

            scored.append(d)
            returns.append(actual_return)

        if not scored:
            return {"error": "No scorable decisions", "decisions": len(decisions)}

        # Compute metrics
        n = len(scored)
        correct = sum(1 for d in scored if d.get(f"was_correct_{horizon_days}d"))
        win_rate = correct / n

        # Returns-based metrics
        if returns:
            mean_return = sum(returns) / len(returns)
            std_return = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / max(len(returns) - 1, 1))
            sharpe = mean_return / std_return if std_return > 0 else 0

            # Compute skewness and kurtosis for DSR
            if len(returns) >= 4 and std_return > 0:
                skew = sum((r - mean_return) ** 3 for r in returns) / (n * std_return ** 3)
                kurtosis = sum((r - mean_return) ** 4 for r in returns) / (n * std_return ** 4)
            else:
                skew = 0
                kurtosis = 3

            # Deflated Sharpe Ratio (n_strategies=1 for single pipeline)
            dsr = compute_deflated_sharpe(sharpe, n, n_strategies=1, skew=skew, kurtosis=kurtosis)

            # Equity curve (cumulative)
            equity = [1.0]
            for r in returns:
                equity.append(equity[-1] * (1 + r))

            # Max drawdown
            peak = equity[0]
            max_dd = 0
            for val in equity:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            mean_return = 0
            std_return = 0
            sharpe = 0
            dsr = 0
            equity = []
            max_dd = 0
            skew = 0
            kurtosis = 3

        # Regime breakdown
        regime_stats = defaultdict(lambda: {"correct": 0, "total": 0, "returns": []})
        for d in scored:
            regime = d.get("regime", "unknown")
            regime_stats[regime]["total"] += 1
            if d.get(f"was_correct_{horizon_days}d"):
                regime_stats[regime]["correct"] += 1
            regime_stats[regime]["returns"].append(d.get(f"actual_return_{horizon_days}d", 0))

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
            if d.get(f"was_correct_{horizon_days}d"):
                signal_stats[sig]["correct"] += 1

        signal_analysis = {
            sig: {
                "win_rate": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0,
                "sample_size": s["total"],
            }
            for sig, s in signal_stats.items()
        }

        # DSR interpretation
        if dsr > 0.95:
            dsr_interpretation = "GENUINE EDGE — statistically significant at 95% confidence"
        elif dsr > 0.80:
            dsr_interpretation = "PROMISING — approaching significance, need more data"
        elif dsr > 0.50:
            dsr_interpretation = "INCONCLUSIVE — cannot distinguish from noise"
        else:
            dsr_interpretation = "LIKELY NOISE — returns indistinguishable from random"

        return {
            "ticker": self.ticker,
            "horizon_days": horizon_days,
            "total_decisions": len(decisions),
            "scored_decisions": n,
            "overall_metrics": {
                "win_rate": round(win_rate, 4),
                "mean_return": round(mean_return, 6),
                "std_return": round(std_return, 6),
                "sharpe_ratio": round(sharpe, 4),
                "deflated_sharpe_ratio": round(dsr, 4),
                "dsr_interpretation": dsr_interpretation,
                "max_drawdown": round(max_dd, 4),
                "skewness": round(skew, 4),
                "kurtosis": round(kurtosis, 4),
                "n_strategies_tested": 1,
            },
            "regime_analysis": regime_analysis,
            "signal_analysis": signal_analysis,
            "equity_curve": [round(e, 6) for e in equity[:100]],  # cap at 100 points
            "validated_at": datetime.now().isoformat(),
        }
