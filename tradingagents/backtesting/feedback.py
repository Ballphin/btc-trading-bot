"""Backtest feedback generator for extracting actionable lessons from historical performance.

Analyzes completed backtests to generate structured lessons that agents can learn from.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


def _get_backtest_regime(bt: Dict) -> str:
    """Get regime for a backtest, using cached value or detecting fresh."""
    regime = bt.get("regime")
    if regime:
        return regime
    try:
        from tradingagents.backtesting.regime import tag_backtest_with_regime
        ticker = bt.get("config", {}).get("ticker", "")
        decisions = bt.get("decisions", [])
        return tag_backtest_with_regime(decisions, ticker) if decisions else "unknown"
    except Exception:
        return "unknown"


class BacktestFeedbackGenerator:
    """Generates actionable lessons from completed backtest results."""
    
    def __init__(self, results_dir: str = "./eval_results"):
        """Initialize the feedback generator.
        
        Args:
            results_dir: Directory containing backtest results
        """
        self.results_dir = Path(results_dir)
    
    def generate_lessons(self, ticker: str, min_trades: int = 5) -> List[Dict[str, Any]]:
        """Generate lessons from all backtests for a ticker.
        
        Args:
            ticker: Ticker symbol
            min_trades: Minimum trades required to generate lessons (default: 5)
            
        Returns:
            List of lesson dictionaries with structure, lesson text, and metadata
        """
        backtest_dir = self.results_dir / "backtests" / ticker
        if not backtest_dir.exists():
            return []
        
        # Load all backtest results
        backtests = self._load_backtests(backtest_dir)
        if not backtests:
            return []
        
        # Filter backtests with sufficient trades
        valid_backtests = [bt for bt in backtests if bt.get("metrics", {}).get("total_trades", 0) >= min_trades]
        if not valid_backtests:
            logger.info(f"No backtests with >= {min_trades} trades for {ticker}")
            return []
        
        # Tag each backtest with its regime
        for bt in valid_backtests:
            if "regime" not in bt:
                bt["regime"] = _get_backtest_regime(bt)

        lessons = []
        
        # Generate signal-level lessons (per regime)
        lessons.extend(self._generate_signal_lessons(valid_backtests, ticker))
        
        # Generate risk management lessons
        lessons.extend(self._generate_risk_lessons(valid_backtests, ticker))
        
        # Generate position sizing lessons
        lessons.extend(self._generate_sizing_lessons(valid_backtests, ticker))
        
        return lessons
    
    def _load_backtests(self, backtest_dir: Path) -> List[Dict[str, Any]]:
        """Load all backtest JSON files from directory."""
        backtests = []
        
        for result_file in sorted(backtest_dir.glob("backtest_*.json"), reverse=True):
            try:
                with open(result_file, "r") as f:
                    data = json.load(f)
                    # Add file metadata
                    data["file_mtime"] = result_file.stat().st_mtime
                    data["file_path"] = str(result_file)
                    backtests.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {result_file}: {e}")
                continue
        
        return backtests
    
    def _generate_signal_lessons(self, backtests: List[Dict], ticker: str) -> List[Dict]:
        """Generate lessons about signal accuracy by signal type, broken down by regime."""
        lessons = []
        
        # Aggregate by (signal_type, regime)
        signal_stats = defaultdict(lambda: {
            "wins": 0, "losses": 0, "total_return": 0.0, "count": 0,
            "stop_outs": 0,
        })
        
        for bt in backtests:
            regime = bt.get("regime", "unknown")

            # Prefer actual trade P&L from closed positions (includes stops, funding, fees)
            trade_history = bt.get("trade_history", [])
            if trade_history:
                for trade in trade_history:
                    signal = trade.get("signal", "").upper()
                    if not signal or signal == "HOLD":
                        continue
                    pnl = trade.get("pnl", 0)
                    key = (signal, regime)
                    signal_stats[key]["count"] += 1
                    signal_stats[key]["total_return"] += pnl / max(trade.get("entry_price", 1), 1e-9)
                    if pnl > 0:
                        signal_stats[key]["wins"] += 1
                    else:
                        signal_stats[key]["losses"] += 1
                    # Track stop-outs separately
                    if trade.get("exit_reason") == "stop_loss":
                        signal_stats[key]["stop_outs"] += 1
                continue

            # Fallback: use decision price comparison (less accurate, no stop/fee accounting)
            decisions = bt.get("decisions", [])
            for i, decision in enumerate(decisions):
                signal = decision.get("signal", "").upper()
                if not signal or signal == "HOLD":
                    continue
                
                # Look ahead to next decision to determine outcome
                if i + 1 < len(decisions):
                    next_decision = decisions[i + 1]
                    price_change = (next_decision["price"] - decision["price"]) / decision["price"]
                    
                    # For SHORT/COVER, profit when price falls
                    if signal in ("SHORT", "COVER"):
                        price_change = -price_change
                    
                    key = (signal, regime)
                    signal_stats[key]["count"] += 1
                    signal_stats[key]["total_return"] += price_change
                    if price_change > 0:
                        signal_stats[key]["wins"] += 1
                    else:
                        signal_stats[key]["losses"] += 1
        
        # Generate lessons for each (signal, regime) combo with sufficient data
        for (signal, regime), stats in signal_stats.items():
            if stats["count"] < 3:
                continue

            win_rate = stats["wins"] / stats["count"]
            avg_return = stats["total_return"] / stats["count"]
            regime_label = regime.replace("_", " ")

            if win_rate >= 0.6:
                lesson_text = (
                    f"In {regime_label} markets, your {signal} signals for {ticker} are effective: "
                    f"{win_rate:.1%} win rate across {stats['count']} trades (avg {avg_return:+.2%}). "
                    f"Continue using {signal} with confidence in similar conditions."
                )
            elif win_rate < 0.4:
                lesson_text = (
                    f"In {regime_label} markets, your {signal} signals for {ticker} underperform: "
                    f"only {win_rate:.1%} win rate across {stats['count']} trades (avg {avg_return:+.2%}). "
                    f"Be more selective or avoid {signal} signals in {regime_label} conditions."
                )
            else:
                lesson_text = (
                    f"In {regime_label} markets, {signal} signals for {ticker} show moderate results: "
                    f"{win_rate:.1%} win rate, {avg_return:+.2%} avg return across {stats['count']} trades."
                )

            lessons.append({
                "category": "signal_accuracy",
                "signal_type": signal,
                "regime": regime,
                "lesson": lesson_text,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "sample_size": stats["count"],
                "confidence": "high" if stats["count"] >= 10 else "medium",
            })

            # Stop-out rate lesson — tracked separately from signal accuracy
            stop_outs = stats.get("stop_outs", 0)
            if stop_outs > 0 and stats["count"] >= 5:
                stop_out_rate = stop_outs / stats["count"]
                if stop_out_rate > 0.60:
                    lessons.append({
                        "category": "risk_management",
                        "subcategory": "stop_out_rate",
                        "signal_type": signal,
                        "regime": regime,
                        "lesson": (
                            f"Stops hit {stop_out_rate:.0%} of {signal} trades in {regime_label} markets for {ticker}. "
                            f"Consider widening the ATR multiplier to avoid premature stop-outs."
                        ),
                        "stop_out_rate": stop_out_rate,
                        "sample_size": stats["count"],
                        "confidence": "high" if stats["count"] >= 10 else "medium",
                    })
        
        return lessons
    
    def _generate_risk_lessons(self, backtests: List[Dict], ticker: str) -> List[Dict]:
        """Generate lessons about stop-loss and take-profit effectiveness."""
        lessons = []
        
        stops_hit = 0
        takes_hit = 0
        total_trades = 0
        
        for bt in backtests:
            metrics = bt.get("metrics", {})
            stops_hit += metrics.get("stops_hit", 0)
            takes_hit += metrics.get("takes_hit", 0)
            total_trades += metrics.get("total_trades", 0)
        
        if total_trades >= 5:
            stop_rate = stops_hit / total_trades if total_trades > 0 else 0
            take_rate = takes_hit / total_trades if total_trades > 0 else 0
            
            # Generate lessons based on stop/take patterns
            if stop_rate > 0.5:
                lesson_text = f"Your stop-losses are being hit frequently ({stop_rate:.1%} of trades) for {ticker}. Consider widening stops or improving entry timing to avoid premature exits."
                lessons.append({
                    "category": "risk_management",
                    "subcategory": "stop_loss",
                    "lesson": lesson_text,
                    "stop_hit_rate": stop_rate,
                    "confidence": "high" if total_trades >= 10 else "medium",
                })
            
            if take_rate > 0.4:
                lesson_text = f"Your take-profit targets are being reached effectively ({take_rate:.1%} of trades) for {ticker}. Your profit-taking discipline is working well."
                lessons.append({
                    "category": "risk_management",
                    "subcategory": "take_profit",
                    "lesson": lesson_text,
                    "take_hit_rate": take_rate,
                    "confidence": "high" if total_trades >= 10 else "medium",
                })
            elif take_rate < 0.2 and total_trades >= 10:
                lesson_text = f"Take-profit targets are rarely being hit ({take_rate:.1%} of trades) for {ticker}. Consider setting more realistic profit targets or trailing stops to lock in gains."
                lessons.append({
                    "category": "risk_management",
                    "subcategory": "take_profit",
                    "lesson": lesson_text,
                    "take_hit_rate": take_rate,
                    "confidence": "high",
                })
        
        return lessons
    
    def _generate_sizing_lessons(self, backtests: List[Dict], ticker: str) -> List[Dict]:
        """Generate lessons about position sizing effectiveness."""
        lessons = []
        
        # Analyze performance by confidence level if available
        confidence_buckets = defaultdict(lambda: {"wins": 0, "losses": 0, "returns": []})
        
        for bt in backtests:
            decisions = bt.get("decisions", [])
            for i, decision in enumerate(decisions):
                confidence = decision.get("confidence")
                if confidence is None:
                    continue
                
                # Categorize confidence
                if confidence >= 0.7:
                    bucket = "high"
                elif confidence >= 0.4:
                    bucket = "medium"
                else:
                    bucket = "low"
                
                # Look ahead to determine outcome
                if i + 1 < len(decisions):
                    next_decision = decisions[i + 1]
                    price_change = (next_decision["price"] - decision["price"]) / decision["price"]
                    
                    confidence_buckets[bucket]["returns"].append(price_change)
                    if price_change > 0:
                        confidence_buckets[bucket]["wins"] += 1
                    else:
                        confidence_buckets[bucket]["losses"] += 1
        
        # Generate lessons for confidence-based sizing
        for bucket, stats in confidence_buckets.items():
            total = stats["wins"] + stats["losses"]
            if total >= 5:
                win_rate = stats["wins"] / total
                avg_return = sum(stats["returns"]) / len(stats["returns"])
                
                if bucket == "high" and win_rate >= 0.6:
                    lesson_text = f"High-confidence trades (≥0.7) for {ticker} perform well: {win_rate:.1%} win rate, {avg_return:+.2%} avg return. Consider sizing up on high-confidence signals."
                elif bucket == "low" and win_rate < 0.5:
                    lesson_text = f"Low-confidence trades (<0.4) for {ticker} underperform: {win_rate:.1%} win rate, {avg_return:+.2%} avg return. Consider reducing position size or avoiding low-confidence signals."
                else:
                    continue  # Skip non-actionable lessons
                
                lessons.append({
                    "category": "position_sizing",
                    "confidence_bucket": bucket,
                    "lesson": lesson_text,
                    "win_rate": win_rate,
                    "avg_return": avg_return,
                    "sample_size": total,
                    "confidence": "high" if total >= 10 else "medium",
                })
        
        return lessons
    
    def save_lessons(self, ticker: str, lessons: List[Dict], results_dir: Optional[str] = None):
        """Save generated lessons to disk as JSON.
        
        Args:
            ticker: Ticker symbol
            lessons: List of lesson dictionaries
            results_dir: Optional override for results directory
        """
        if results_dir:
            save_dir = Path(results_dir) / ticker / "backtest_lessons"
        else:
            save_dir = self.results_dir / ticker / "backtest_lessons"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lessons_file = save_dir / f"lessons_{timestamp}.json"
        
        data = {
            "ticker": ticker,
            "generated_at": datetime.now().isoformat(),
            "lessons": lessons,
            "lesson_count": len(lessons),
        }
        
        with open(lessons_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(lessons)} lessons to {lessons_file}")
        
        return str(lessons_file)
    
    def load_latest_lessons(self, ticker: str) -> List[Dict]:
        """Load the most recent lessons for a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            List of lesson dictionaries
        """
        lessons_dir = self.results_dir / ticker / "backtest_lessons"
        if not lessons_dir.exists():
            return []
        
        # Find most recent lessons file
        lesson_files = sorted(lessons_dir.glob("lessons_*.json"), reverse=True)
        if not lesson_files:
            return []
        
        try:
            with open(lesson_files[0], "r") as f:
                data = json.load(f)
                return data.get("lessons", [])
        except Exception as e:
            logger.warning(f"Failed to load lessons: {e}")
            return []
