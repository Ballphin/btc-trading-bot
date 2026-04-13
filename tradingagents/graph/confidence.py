"""Confidence calibration and position sizing for trading signals.

Two-tier calibration: LLM confidence is first anchored via prompt tiers, then
adjusted against historical win rates from backtests, a regime-based volatility
penalty, and a hedge-word penalty extracted from the reasoning text.

Includes:
- Cold-start overconfidence correction (MIT CSAIL, 2025)
- Kelly sizing with Deflated Sharpe gate (Bailey & López de Prado, 2014)
- Kelly shrinkage multiplier (Thorp & MacLean, 2011)
- Regime-conditional signal gating
- Portfolio-level exposure cap
"""

import json
import math
import logging
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Hedge words in reasoning that signal genuine LLM uncertainty
# Removed "risk", "volatile", "caution" — they are descriptive trading terms,
# not hedging language, and appear in >70% of crypto analyses.
_HEDGE_WORDS = frozenset({
    "but", "however", "although", "despite", "uncertain",
    "concern", "weak",
})

# Signals that are binary trades — Kelly applies
_DIRECTIONAL_SIGNALS = frozenset({"BUY", "SELL", "SHORT", "COVER"})

# Minimum confidence threshold per regime (gate signals below this)
# Regime-conditional for SHORT signals (Debate 3: avoid permanently gating shorts)
_GATE_THRESHOLDS = {
    "volatile_down": 0.58,
    "volatile_up":   0.45,
    "volatile":      0.52,
    "trending_up":   0.45,
    "trending_down": 0.45,
    "ranging":       0.48,
    "unknown":       0.50,
}

# SHORT-specific gate thresholds — lower bar in downtrends
_SHORT_GATE_THRESHOLDS = {
    "trending_down": 0.30,
    "volatile_down": 0.30,
    "trending_up":   0.50,
    "volatile_up":   0.45,
    "volatile":      0.40,
    "ranging":       0.40,
    "unknown":       0.45,
}

# Portfolio-level exposure cap (sum of all active positions)
MAX_TOTAL_EXPOSURE_PCT = 0.30


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class ConfidenceScorer:
    """Calibrates LLM confidence and computes position sizing for trade signals."""

    # Baseline hold period for the hold-period scalar (days)
    HOLD_BASELINE_DAYS = 7
    # Portfolio risk fraction per trade (2%)
    PORTFOLIO_RISK_TARGET = 0.02

    def __init__(self, results_dir: Optional[str] = None):
        self.results_dir = Path(results_dir) if results_dir else Path("eval_results")

    def calibrate(
        self,
        llm_confidence: float,
        win_rate: Optional[float],
        sample_size: int,
        R: Optional[float],
        regime: str,
        above_sma20: bool,
        reasoning: str = "",
        signal: str = "",
    ) -> tuple:
        """Calibrate raw LLM confidence against historical data, vol regime, and reasoning.

        Returns:
            (calibrated_confidence: float, hedge_penalty: float)
        """
        calibrated = _clamp(llm_confidence, 0.0, 1.0)

        # Historical win-rate adjustment (only when data is available)
        if win_rate is not None and sample_size > 0:
            # Correct breakeven probability depends on R ratio
            if R and R > 0:
                breakeven_p = 1.0 / (1.0 + R)
            else:
                breakeven_p = 0.50

            raw_multiplier = win_rate / breakeven_p if breakeven_p > 0 else 1.0
            # SQR FIX: Use square root law for sample size weighting (more statistically sound)
            # weight = sqrt(sample_size / 30) where 30 is the baseline for full weight
            weight = min(1.0, math.sqrt(sample_size / 30.0))
            effective_multiplier = 1.0 + (raw_multiplier - 1.0) * weight
            calibrated = _clamp(calibrated * effective_multiplier, 0.30, 0.95)

        # Regime volatility penalty
        if regime == "volatile":
            sig_upper = signal.upper() if signal else ""
            if sig_upper in ("SHORT", "SELL") and not above_sma20:
                # Confirmed breakdown below SMA20 — high-conviction short setup
                # Small penalty (8%) for short-squeeze risk only
                vol_penalty = 0.92
            else:
                # Differentiate breakout (above SMA20) from crash (below SMA20)
                vol_penalty = 0.90 if above_sma20 else 0.75
            calibrated = _clamp(calibrated * vol_penalty, 0.25, 0.95)

        # Hedge-word penalty from reasoning text (multiplicative, word-boundary matching)
        reasoning_lower = reasoning.lower() if reasoning else ""
        hedge_count = sum(1 for w in _HEDGE_WORDS if re.search(r'\b' + w + r'\b', reasoning_lower))
        # Multiplicative penalty: 3% per hedge word, max 12% total
        hedge_multiplier = max(0.88, 1.0 - hedge_count * 0.03)
        hedge_penalty = 1.0 - hedge_multiplier
        calibrated = _clamp(calibrated * hedge_multiplier, 0.25, 0.95)

        return calibrated, hedge_penalty

    def kelly_position_size(
        self,
        p: float,
        R: Optional[float],
        entry_price: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        signal: str,
        max_hold_days: int = 7,
        sample_size: int = 0,
        leverage: float = 1.0,
        liquidation_price: Optional[float] = None,
    ) -> tuple:
        """Compute position size as a fraction of portfolio using half-Kelly.

        Includes Kelly shrinkage multiplier (Thorp & MacLean, 2011) that
        reduces allocation proportional to estimation uncertainty.

        Returns:
            (position_size_pct: float, r_ratio: float|None, hold_scalar: float)
        """
        sig = signal.upper()

        # Non-directional signals
        if sig == "HOLD":
            return 0.0, None, 1.0
        if sig in ("OVERWEIGHT", "UNDERWEIGHT"):
            hold_scalar = min(1.0, self.HOLD_BASELINE_DAYS / max(max_hold_days, 1))
            return _clamp(0.40 * hold_scalar, 0.0, 1.0), None, hold_scalar

        # Validate price levels before Kelly
        if not entry_price or entry_price <= 0:
            hold_scalar = min(1.0, self.HOLD_BASELINE_DAYS / max(max_hold_days, 1))
            return 0.50 * hold_scalar, None, hold_scalar

        sl = stop_loss or 0.0
        tp = take_profit or 0.0

        if sl <= 0 or tp <= 0:
            hold_scalar = min(1.0, self.HOLD_BASELINE_DAYS / max(max_hold_days, 1))
            return 0.50 * hold_scalar, None, hold_scalar

        # Compute R ratio based on signal direction
        if sig in ("BUY", "OVERWEIGHT"):
            stop_dist = entry_price - sl
            profit_dist = tp - entry_price
        else:  # SHORT, SELL, COVER
            stop_dist = sl - entry_price
            profit_dist = entry_price - tp

        if stop_dist <= 0 or profit_dist <= 0:
            hold_scalar = min(1.0, self.HOLD_BASELINE_DAYS / max(max_hold_days, 1))
            return 0.50 * hold_scalar, None, hold_scalar

        r_ratio = profit_dist / stop_dist

        # Corrected Kelly: f* = p - (1-p)/R
        f_star = p - (1.0 - p) / r_ratio
        if f_star <= 0:
            return 0.0, r_ratio, 1.0

        half_kelly = 0.5 * f_star

        # Kelly shrinkage multiplier (Thorp & MacLean, 2011)
        # Monotonically increasing: more data → less shrinkage (closer to 1.0)
        shrinkage = max(0.0, 1.0 - 2.0 / max(sample_size, 2))
        half_kelly *= shrinkage

        # Portfolio risk cap: risk at most PORTFOLIO_RISK_TARGET per trade
        stop_pct = stop_dist / entry_price
        vol_cap = self.PORTFOLIO_RISK_TARGET / stop_pct if stop_pct > 0 else 1.0

        # Hold-period scalar: longer holds have higher variance, scale down
        hold_scalar = min(1.0, self.HOLD_BASELINE_DAYS / max(max_hold_days, 1))

        # WCT FIX: Liquidation guard for crypto leverage
        # Position size capped at liquidation threshold to prevent forced liquidation
        if liquidation_price and leverage > 1 and entry_price:
            liq_distance = abs(entry_price - liquidation_price) / entry_price
            if liq_distance > 0:
                max_position_liq = self.PORTFOLIO_RISK_TARGET / (liq_distance * leverage)
                half_kelly = min(half_kelly, max_position_liq)

        size = _clamp(min(half_kelly, vol_cap) * hold_scalar, 0.0, 1.0)
        return size, r_ratio, hold_scalar

    def conviction_label(self, confidence: float) -> str:
        """Map confidence float to a human-readable label."""
        if confidence >= 0.80:
            return "VERY HIGH"
        if confidence >= 0.65:
            return "HIGH"
        if confidence >= 0.50:
            return "MODERATE"
        return "LOW"

    def _gate_key(self, regime: str, above_sma20: bool) -> str:
        """Resolve the gate threshold key for this regime + direction."""
        if regime == "volatile":
            return "volatile_up" if above_sma20 else "volatile_down"
        return regime if regime in _GATE_THRESHOLDS else "unknown"

    def score(
        self,
        llm_confidence: float,
        ticker: str,
        signal: str,
        knowledge_store,
        regime_ctx: dict,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        max_hold_days: int = 7,
        reasoning: str = "",
    ) -> dict:
        """Full confidence scoring pipeline.

        Args:
            llm_confidence: Raw confidence from LLM output (0.0–1.0)
            ticker: Ticker symbol
            signal: Signal string (BUY, SELL, SHORT, etc.)
            knowledge_store: BacktestKnowledgeStore instance (or None)
            regime_ctx: Dict from detect_regime_context()
            stop_loss: Stop loss price from LLM output
            take_profit: Take profit price from LLM output
            max_hold_days: Max hold days from LLM output
            reasoning: Reasoning text from LLM output

        Returns:
            Dict with: confidence, position_size_pct, conviction_label, gated,
                       r_ratio, r_ratio_warning, hold_period_scalar,
                       hedge_penalty_applied, win_rate_used, sample_size_used, regime_used
        """
        regime = regime_ctx.get("regime", "unknown")
        current_price = regime_ctx.get("current_price")
        above_sma20 = regime_ctx.get("above_sma20", True)

        # ── Cold-start overconfidence correction (MIT CSAIL, 2025) ──
        # Continuous linear ramp: correction = base + (1-base) * min(1, n/60)
        # At n=0: base (0.85), at n=30: 0.925, at n=60+: 1.0 (no penalty)
        _COLD_START_THRESHOLD = 60
        base_correction = 0.85  # default: 15% dampener (was 20%)
        try:
            cal_path = self.results_dir / "shadow" / ticker / "calibration.json"
            if cal_path.exists():
                cal = json.loads(cal_path.read_text())
                base_correction = cal.get("correction", 0.85)
        except Exception:
            pass

        try:
            from tradingagents.backtesting.scorecard import count_scored_decisions
            scored_count = count_scored_decisions(ticker, results_dir=str(self.results_dir))
        except Exception:
            scored_count = 0

        overconfidence_correction = base_correction + (1.0 - base_correction) * min(1.0, scored_count / _COLD_START_THRESHOLD)
        llm_confidence *= overconfidence_correction
        if overconfidence_correction < 1.0:
            logger.debug(
                f"Cold-start confidence correction: {overconfidence_correction:.3f} "
                f"(base={base_correction:.2f}, {scored_count}/{_COLD_START_THRESHOLD} decisions)"
            )

        # Compute R using current_price as entry proxy
        R = None
        if current_price and current_price > 0 and stop_loss and take_profit:
            sig = signal.upper()
            if sig in ("BUY", "OVERWEIGHT") and current_price > stop_loss:
                R = (take_profit - current_price) / (current_price - stop_loss)
            elif sig in ("SHORT", "SELL") and current_price < stop_loss:
                R = (current_price - take_profit) / (stop_loss - current_price)
            if R is not None and R <= 0:
                R = None

        # Fetch historical win rate from backtest knowledge store
        win_rate_data = None
        if knowledge_store is not None:
            try:
                win_rate_data = knowledge_store.get_signal_win_rate(
                    ticker, signal, regime=regime
                )
            except Exception:
                pass

        win_rate = win_rate_data["win_rate"] if win_rate_data else None
        sample_size = win_rate_data["sample_size"] if win_rate_data else 0

        # ── Kelly eligibility: DSR gate (Bailey & López de Prado, 2014) ──
        # Require 60 trades AND statistically significant Sharpe at 95% confidence
        _KELLY_MIN_TRADES = 60
        _FIXED_FALLBACK_SIZE = 0.05  # 5% fixed allocation during cold start
        kelly_eligible = sample_size >= _KELLY_MIN_TRADES

        if kelly_eligible and win_rate is not None:
            try:
                from tradingagents.backtesting.walk_forward import compute_deflated_sharpe
                # Use Kelly edge as numerator: p*R - (1-p), not win_rate - 0.5
                # This correctly evaluates profitability for low-winrate/high-R systems
                _R_for_dsr = R if R is not None and R > 0 else 1.0
                kelly_edge = win_rate * _R_for_dsr - (1 - win_rate)
                se_kelly = max(0.01, _R_for_dsr * math.sqrt(
                    win_rate * (1 - win_rate) / sample_size
                ))
                signal_sharpe = kelly_edge / se_kelly
                dsr = compute_deflated_sharpe(signal_sharpe, sample_size, n_strategies=1)
                if dsr < 0.95:
                    kelly_eligible = False
                    logger.debug(
                        f"Kelly disabled for {ticker}/{signal}: DSR={dsr:.3f} < 0.95 "
                        f"(n={sample_size})"
                    )
            except Exception as e:
                logger.debug(f"DSR check failed, keeping Kelly eligible: {e}")

        # Two-tier calibration (always runs — adjusts confidence, not sizing)
        calibrated, hedge_penalty = self.calibrate(
            llm_confidence=llm_confidence,
            win_rate=win_rate if kelly_eligible else None,
            sample_size=sample_size if kelly_eligible else 0,
            R=R,
            regime=regime,
            above_sma20=above_sma20,
            reasoning=reasoning,
            signal=signal,
        )

        # ── Position sizing: Kelly when eligible, fixed fallback otherwise ──
        # TODO: Get actual leverage and liquidation_price from caller for crypto assets
        leverage = 1.0  # Default to spot (no leverage)
        liquidation_price = None  # No liquidation for spot
        
        if kelly_eligible:
            position_size_pct, r_ratio, hold_scalar = self.kelly_position_size(
                p=calibrated,
                R=R,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal=signal,
                max_hold_days=max_hold_days,
                sample_size=sample_size,
                leverage=leverage,
                liquidation_price=liquidation_price,
            )
        else:
            hold_scalar = min(1.0, self.HOLD_BASELINE_DAYS / max(max_hold_days, 1))
            position_size_pct = _FIXED_FALLBACK_SIZE * hold_scalar
            r_ratio = None
            if sample_size > 0:
                logger.debug(
                    f"Kelly skipped for {ticker}/{signal}: only {sample_size} trades "
                    f"(need {_KELLY_MIN_TRADES}), using fixed {_FIXED_FALLBACK_SIZE*100:.0f}%"
                )

        # ── Signal gating: DISABLED - always show trade parameters ──
        # Previously gated signals below regime thresholds; now always display
        # trade parameters regardless of confidence level
        gated = False

        return {
            "confidence": round(calibrated, 4),
            "position_size_pct": round(position_size_pct, 4),
            "conviction_label": self.conviction_label(calibrated),
            "gated": gated,
            "r_ratio": round(r_ratio, 3) if r_ratio is not None else None,
            "r_ratio_warning": bool(r_ratio is not None and r_ratio < 1.0),
            "hold_period_scalar": round(hold_scalar, 3),
            "hedge_penalty_applied": round(hedge_penalty, 3),
            "win_rate_used": round(win_rate, 4) if win_rate is not None else None,
            "sample_size_used": sample_size,
            "regime_used": regime,
            "overconfidence_correction": round(overconfidence_correction, 4),
            "kelly_eligible": kelly_eligible,
            "scored_count_used": scored_count,
        }
