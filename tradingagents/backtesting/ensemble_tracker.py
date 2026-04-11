"""Track ensemble accuracy over time for calibration analysis.

Logs ensemble results and outcomes to enable accuracy tracking and validation
that ensemble mode actually improves performance vs single-run.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EnsembleTracker:
    """Log ensemble results and outcomes for accuracy analysis."""
    
    def __init__(self, results_dir: str = "./eval_results"):
        """Initialize the ensemble tracker.
        
        Args:
            results_dir: Directory containing backtest results
        """
        self.results_dir = Path(results_dir)
        self.log_file = self.results_dir / "ensemble_accuracy_log.json"
        
    def log_result(
        self,
        timestamp: str,
        ticker: str,
        consensus_signal: str,
        consensus_confidence: float,
        individual_signals: List[Dict],
        market_outcome: Optional[str] = None,
        divergence_metrics: Optional[Dict] = None,
    ):
        """Log an ensemble result for later accuracy analysis.
        
        Args:
            timestamp: ISO format timestamp
            ticker: Ticker symbol
            consensus_signal: Final consensus signal
            consensus_confidence: Final consensus confidence
            individual_signals: List of individual run results
            market_outcome: Optional outcome ("win", "loss", "breakeven")
            divergence_metrics: Optional divergence metrics dict
        """
        entry = {
            "timestamp": timestamp,
            "ticker": ticker,
            "consensus": {
                "signal": consensus_signal,
                "confidence": consensus_confidence,
            },
            "individual": individual_signals,
            "market_outcome": market_outcome,
            "agreement": self._compute_agreement(individual_signals),
            "divergence_metrics": divergence_metrics or {},
        }
        
        # Append to log file
        logs = []
        if self.log_file.exists():
            try:
                logs = json.loads(self.log_file.read_text())
            except json.JSONDecodeError:
                logger.warning("Corrupted ensemble log file, starting fresh")
                logs = []
        
        logs.append(entry)
        
        # Write back
        try:
            self.log_file.write_text(json.dumps(logs, indent=2))
        except Exception as e:
            logger.error(f"Failed to write ensemble log: {e}")
    
    def _compute_agreement(self, signals: List[Dict]) -> float:
        """Compute signal agreement percentage.
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            Agreement ratio (0.0 to 1.0)
        """
        if not signals:
            return 0.0
        
        counts = {}
        for s in signals:
            sig = s.get("signal", "HOLD")
            counts[sig] = counts.get(sig, 0) + 1
        
        return max(counts.values()) / len(signals)
    
    def get_accuracy_stats(self, ticker: Optional[str] = None) -> Dict:
        """Compute ensemble accuracy statistics.
        
        Args:
            ticker: Optional ticker to filter by
            
        Returns:
            Statistics dict with total, correct, accuracy, avg_agreement
        """
        if not self.log_file.exists():
            return {"total": 0, "accuracy": None, "avg_agreement": 0.0}
        
        try:
            logs = json.loads(self.log_file.read_text())
        except json.JSONDecodeError:
            logger.error("Could not parse ensemble log file")
            return {"total": 0, "accuracy": None, "avg_agreement": 0.0}
        
        if ticker:
            logs = [l for l in logs if l.get("ticker") == ticker]
        
        total = len(logs)
        if total == 0:
            return {"total": 0, "accuracy": None, "avg_agreement": 0.0}
        
        # Count outcomes
        correct = len([l for l in logs if l.get("market_outcome") == "win"])
        losses = len([l for l in logs if l.get("market_outcome") == "loss"])
        breakeven = len([l for l in logs if l.get("market_outcome") == "breakeven"])
        
        # Calculate average agreement
        avg_agreement = sum(l.get("agreement", 0.0) for l in logs) / total
        
        # Calculate confidence calibration
        confidences = [l["consensus"]["confidence"] for l in logs if "consensus" in l]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total": total,
            "correct": correct,
            "losses": losses,
            "breakeven": breakeven,
            "accuracy": correct / total if total > 0 else None,
            "avg_agreement": avg_agreement,
            "avg_confidence": avg_confidence,
        }
    
    def get_divergence_analysis(self) -> Dict:
        """Analyze divergence patterns in ensemble results.
        
        Returns:
            Analysis of when divergence occurs and how it correlates with outcomes
        """
        if not self.log_file.exists():
            return {}
        
        try:
            logs = json.loads(self.log_file.read_text())
        except json.JSONDecodeError:
            return {}
        
        # Analyze by confidence range
        low_conf = [l for l in logs if l.get("divergence_metrics", {}).get("confidence_range", 0) > 0.20]
        high_conf = [l for l in logs if l.get("divergence_metrics", {}).get("confidence_range", 0) <= 0.20]
        
        low_acc = len([l for l in low_conf if l.get("market_outcome") == "win"]) / len(low_conf) if low_conf else 0
        high_acc = len([l for l in high_conf if l.get("market_outcome") == "win"]) / len(high_conf) if high_conf else 0
        
        return {
            "high_divergence_count": len(low_conf),
            "low_divergence_count": len(high_conf),
            "high_divergence_accuracy": low_acc,
            "low_divergence_accuracy": high_acc,
            "recommendation": "Ensemble effective" if high_acc > low_acc else "Review divergence thresholds",
        }
