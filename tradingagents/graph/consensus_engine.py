"""Consensus computation logic for ensemble analysis.

Computes consensus from multiple signal results using R:R-first averaging
to maintain directional logic integrity.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConsensusResult:
    """Result of consensus computation."""
    signal: str
    confidence: float
    stop_loss_price: float
    take_profit_price: float
    max_hold_days: int
    reasoning: str
    individual_results: List[Dict]
    divergence_metrics: Dict
    ensemble_metadata: Dict


class ConsensusEngine:
    """Pure logic for computing consensus from multiple signal results."""
    
    @staticmethod
    def compute_r_ratio(entry: float, stop: float, take: float, signal: str) -> Optional[float]:
        """Compute R:R ratio for a single signal result.
        
        Args:
            entry: Entry price
            stop: Stop loss price
            take: Take profit price
            signal: Signal direction (BUY, SELL, SHORT, etc.)
            
        Returns:
            R:R ratio (reward/risk) or None if invalid
        """
        if not all([entry, stop, take]) or entry <= 0:
            return None
        
        if signal in ["SHORT", "SELL"]:
            risk = abs(stop - entry)
            reward = abs(entry - take)
        else:
            risk = abs(entry - stop)
            reward = abs(take - entry)
        
        if risk <= 0 or reward <= 0:
            return None
        return reward / risk
    
    @staticmethod
    def majority_vote(signals: List[str]) -> str:
        """Return majority signal.
        
        Args:
            signals: List of signal strings
            
        Returns:
            Most common signal
        """
        counts = {}
        for s in signals:
            counts[s] = counts.get(s, 0) + 1
        return max(counts, key=counts.get)
    
    @staticmethod
    def compute_consensus(
        results: List[Dict],
        entry_price: float,
        ticker: str = "",
    ) -> ConsensusResult:
        """Compute consensus from multiple signal results.
        
        BLOCKER FIX: Computes R:R per run first, averages ratios,
        then derives prices from averaged R:R (not average prices directly).
        
        Args:
            results: List of signal result dictionaries
            entry_price: Entry price for R:R calculations
            ticker: Optional ticker symbol for logging
            
        Returns:
            ConsensusResult with averaged parameters
        """
        # Extract signals and confidences
        signals = [r.get("signal", "HOLD") for r in results]
        confidences = [r.get("confidence", 0.5) for r in results]
        
        # Majority vote for signal
        signal_counts = {}
        for s in signals:
            signal_counts[s] = signal_counts.get(s, 0) + 1
        consensus_signal = max(signal_counts, key=signal_counts.get)
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences)
        
        # BLOCKER FIX: Compute R:R per run, then average ratios
        r_ratios = []
        for r in results:
            sl = r.get("stop_loss_price")
            tp = r.get("take_profit_price")
            rr = ConsensusEngine.compute_r_ratio(entry_price, sl, tp, r.get("signal", "HOLD"))
            if rr is not None:
                r_ratios.append(rr)
        
        avg_r_ratio = sum(r_ratios) / len(r_ratios) if r_ratios else 2.0
        
        # BLOCKER FIX: Derive consensus prices from averaged R:R
        # Use 5% base risk to compute stops, then apply averaged R:R
        base_risk_pct = 0.05
        
        if consensus_signal in ["SHORT", "SELL"]:
            # For shorts: stop > entry > take_profit
            consensus_sl = entry_price * (1 + base_risk_pct)
            # R:R = reward/risk => reward = R:R * risk
            risk = consensus_sl - entry_price
            reward = avg_r_ratio * risk
            consensus_tp = entry_price - reward
        else:
            # For longs: take_profit > entry > stop
            consensus_sl = entry_price * (1 - base_risk_pct)
            risk = entry_price - consensus_sl
            reward = avg_r_ratio * risk
            consensus_tp = entry_price + reward
        
        # Median hold days
        hold_days = sorted([r.get("max_hold_days", 3) for r in results])
        consensus_hold = hold_days[len(hold_days) // 2]
        
        # Best reasoning (highest confidence)
        best_result = max(results, key=lambda r: r.get("confidence", 0))
        best_reasoning = best_result.get("reasoning", "")
        
        # HIGH FIX: Range-based divergence detection (not std for N=3)
        conf_range = max(confidences) - min(confidences)
        signal_agreement = signal_counts[consensus_signal] / len(signals)
        
        # Price range for stops and takes
        stops = [r.get("stop_loss_price", entry_price) for r in results if r.get("stop_loss_price")]
        takes = [r.get("take_profit_price", entry_price) for r in results if r.get("take_profit_price")]
        
        stop_range_pct = (max(stops) - min(stops)) / (sum(stops) / len(stops)) if stops else 0
        take_range_pct = (max(takes) - min(takes)) / (sum(takes) / len(takes)) if takes else 0
        
        return ConsensusResult(
            signal=consensus_signal,
            confidence=round(avg_confidence, 4),
            stop_loss_price=round(consensus_sl, 2),
            take_profit_price=round(consensus_tp, 2),
            max_hold_days=consensus_hold,
            reasoning=best_reasoning,
            individual_results=results,
            divergence_metrics={
                "confidence_range": round(conf_range, 4),  # HIGH FIX: range not std
                "signal_agreement": round(signal_agreement, 4),
                "r_ratio_consensus": round(avg_r_ratio, 4),
                "stop_loss_range_pct": round(stop_range_pct, 4),
                "take_profit_range_pct": round(take_range_pct, 4),
            },
            ensemble_metadata={
                "runs": len(results),
                "valid_runs": len([r for r in results if "error" not in r]),
                "retry_count": 0,  # Set by orchestrator
                "entry_price_snapshot": entry_price,
                "divergence_metrics": {
                    "confidence_range": round(conf_range, 4),
                    "signal_agreement": round(signal_agreement, 4),
                    "r_ratio_consensus": round(avg_r_ratio, 4),
                    "stop_loss_range_pct": round(stop_range_pct, 4),
                    "take_profit_range_pct": round(take_range_pct, 4),
                },
            }
        )
    
    @staticmethod
    def should_rerun(
        results: List[Dict],
        confidence_range_threshold: float = 0.20,
        agreement_threshold: float = 0.67,
    ) -> bool:
        """Check if ensemble should be re-run due to high divergence.
        
        HIGH FIX: Uses range instead of std dev for N=3 samples.
        
        Args:
            results: List of signal results
            confidence_range_threshold: Max acceptable confidence range (default 20%)
            agreement_threshold: Min acceptable signal agreement (default 67%)
            
        Returns:
            True if re-run recommended
        """
        confidences = [r.get("confidence", 0.5) for r in results]
        signals = [r.get("signal", "HOLD") for r in results]
        
        # HIGH FIX: Range-based threshold
        conf_range = max(confidences) - min(confidences)
        
        signal_counts = {}
        for s in signals:
            signal_counts[s] = signal_counts.get(s, 0) + 1
        max_agreement = max(signal_counts.values()) / len(signals)
        
        # HIGH FIX: Return True (should rerun) if divergence is HIGH
        # i.e., confidence_range > threshold OR agreement < threshold
        return conf_range > confidence_range_threshold or max_agreement < agreement_threshold
