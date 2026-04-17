"""Quant Pulse v3 — Confluence Entry-Timing Engine.

Role in v3 architecture:
    Layer 1 (TSMOM, `pulse_tsmom`)   — generates primary alpha direction.
    Layer 2 (this file)              — decides WHEN to enter within the
                                       window. AND-gated with TSMOM: no
                                       signal unless confluence agrees
                                       with TSMOM direction.

Key differences vs v2:
    * Per-TF normalization (bug fix: denominator was miscalibrated).
    * Regime on/off gate (not weight blending).
    * Persistence multiplier (×1.2/×0.8/×1.0 based on previous direction).
    * Tiered premium scoring; order-flow score capped.
    * Funding elevation override (|ann_funding| > 20% forces counter-carry).
    * Liquidation-cascade override (clusters + falling vol → ±2 factor).
    * Book-imbalance factor from L2 snapshot.
    * YAML-driven thresholds and weights.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from tradingagents.pulse.config import PulseConfig, get_config

logger = logging.getLogger(__name__)

# ── Legacy defaults exported for back-compat with existing tests ──────

DEFAULT_SIGNAL_THRESHOLD = 0.25
CONFIDENCE_DIVISOR = 0.5

TF_WEIGHTS = {
    "1m": 0.05,
    "5m": 0.10,
    "15m": 0.20,
    "1h": 0.30,
    "4h": 0.35,
}
ORDER_FLOW_WEIGHT = 0.30

HOLD_MINUTES = {
    "1m": 5,
    "5m": 15,
    "15m": 45,
    "1h": 120,
    "4h": 480,
}

_TF_ORDER = ["1m", "5m", "15m", "1h", "4h"]


# ── Per-indicator scoring ─────────────────────────────────────────────

def _score_rsi(rsi: Optional[float]) -> int:
    if rsi is None:
        return 0
    if rsi < 35:
        return 1
    if rsi > 65:
        return -1
    return 0


def _score_macd(macd_hist: Optional[float], macd_direction: Optional[str]) -> int:
    if macd_hist is None:
        return 0
    if macd_hist > 0 and macd_direction == "rising":
        return 1
    if macd_hist < 0 and macd_direction == "falling":
        return -1
    return 0


def _score_bb(bb_pct: Optional[float]) -> int:
    if bb_pct is None:
        return 0
    if bb_pct < 0.15:
        return 1
    if bb_pct > 0.85:
        return -1
    return 0


def _score_ema_cross(
    ema_cross: Optional[str],
    ema9: Optional[float],
    ema21: Optional[float],
    atr: Optional[float],
) -> int:
    """EMA cross with ATR-adaptive neutral zone: |ema9 - ema21| < 0.3·ATR → 0."""
    if ema_cross is None or ema9 is None or ema21 is None:
        return 0
    if atr is not None and atr > 1e-10:
        if abs(ema9 - ema21) < 0.3 * atr:
            return 0
    if ema_cross == "bullish":
        return 1
    if ema_cross == "bearish":
        return -1
    return 0


def _volume_amplifier(rel_volume: Optional[float]) -> float:
    return 1.5 if (rel_volume is not None and rel_volume > 1.5) else 1.0


# ── Pattern scoring ───────────────────────────────────────────────────

_BULLISH_PATTERNS = {
    "bullish_engulfing", "hammer", "morning_star",
    "three_white_soldiers", "bullish_harami",
}
_BEARISH_PATTERNS = {
    "bearish_engulfing", "shooting_star", "evening_star",
    "three_black_crows", "bearish_harami",
}


def _score_patterns(
    patterns: list,
    close_price: Optional[float],
    support: Optional[float],
    resistance: Optional[float],
    vwap: Optional[float],
) -> int:
    """Candlestick pattern score: ±2 near key level, ±1 otherwise; doji neutral."""
    if not patterns or close_price is None:
        return 0
    near_key_level = False
    if close_price > 0:
        for level in [support, resistance, vwap]:
            if level is not None and level > 0:
                if abs(close_price - level) / close_price <= 0.003:
                    near_key_level = True
                    break
    total = 0
    for p in patterns:
        if p == "doji":
            continue
        base = 2 if near_key_level else 1
        if p in _BULLISH_PATTERNS:
            total += base
        elif p in _BEARISH_PATTERNS:
            total -= base
    return total


# ── Order flow scoring (tiered premium + adaptive funding) ────────────

def _score_order_flow(
    premium_pct: Optional[float],
    funding_delta: Optional[float],
    vwap_position: Optional[int],
    backtest_mode: bool = False,
    small_premium_thr: float = 0.05,
    large_premium_thr: float = 0.15,
    funding_threshold_abs: float = 0.00005,
    cap_abs: int = 3,
) -> tuple:
    """Order flow: tiered premium (±1 small / ±2 large), funding Δ, VWAP pos.

    Returns (score, n_active). Score clamped to ±cap_abs.
    """
    score = 0
    n_active = 0

    # Tiered premium — skipped in backtest mode (plan: premium is proxy-biased)
    if not backtest_mode and premium_pct is not None:
        n_active += 1
        if premium_pct >= large_premium_thr:
            score -= 2   # overheated longs → bearish
        elif premium_pct >= small_premium_thr:
            score -= 1
        elif premium_pct <= -large_premium_thr:
            score += 2
        elif premium_pct <= -small_premium_thr:
            score += 1

    if funding_delta is not None:
        n_active += 1
        if funding_delta < -funding_threshold_abs:
            score += 1
        elif funding_delta > funding_threshold_abs:
            score -= 1

    if vwap_position is not None and vwap_position != 0:
        n_active += 1
        score += vwap_position

    # Cap at ±3 (plan)
    score = max(-cap_abs, min(cap_abs, score))
    return score, n_active


# ── 4h staleness discount ────────────────────────────────────────────

def _4h_staleness_weight(base_weight: float, last_4h_close_ts: Optional[datetime]) -> float:
    if last_4h_close_ts is None:
        return base_weight * 0.5
    now = datetime.now(timezone.utc)
    hours_stale = (now - last_4h_close_ts).total_seconds() / 3600
    discount = max(0.5, 1.0 - hours_stale / 4.0)
    return base_weight * discount


# ── Regime on/off gate ────────────────────────────────────────────────

_INDICATOR_GROUPS = {
    "ema_cross": "ema",
    "macd": "macd",
    "rsi": "rsi",
    "bb": "bb",
}


def _apply_regime_gate(
    tf_data: dict,
    regime_mode: str,
    cfg: PulseConfig,
) -> dict:
    """Zero-out indicators disabled in current regime. Returns MODIFIED COPY."""
    if not cfg.get("confluence", "regime_onoff_gate", "enabled", default=True):
        return tf_data

    disabled: list = []
    if regime_mode == "chop":
        disabled = cfg.get(
            "confluence", "regime_onoff_gate", "chop_disabled_indicators", default=[]
        ) or []
    elif regime_mode == "trend":
        disabled = cfg.get(
            "confluence", "regime_onoff_gate", "trend_disabled_indicators", default=[]
        ) or []
    else:
        return tf_data

    out = dict(tf_data)
    for name in disabled:
        if name == "ema_cross":
            out["ema_cross"] = None
        elif name == "macd":
            out["macd_hist"] = None
        elif name == "rsi":
            out["rsi"] = None
        elif name == "bb":
            out["bb_pct"] = None
    return out


def _regime_indicator_muls(regime_mode: str, cfg: PulseConfig) -> Dict[str, float]:
    """Per-indicator multipliers for the current regime (used when gate disabled)."""
    profile = cfg.get(
        "confluence", "regime_weight_profiles", regime_mode, default=None
    ) or cfg.get("confluence", "regime_weight_profiles", "mixed", default={}) or {}
    return {
        "ema": float(profile.get("ema_cross_mul", 1.0)),
        "macd": float(profile.get("macd_mul", 1.0)),
        "rsi": float(profile.get("rsi_mul", 1.0)),
        "bb": float(profile.get("bb_mul", 1.0)),
    }


# ── Funding-elevation override ────────────────────────────────────────

def _funding_elevation_override(
    funding_rate: Optional[float], cfg: PulseConfig
) -> Optional[int]:
    """Return forced direction (+1 / -1) if |ann funding| exceeds threshold.

    Hyperliquid funding is settled hourly, so annualized = rate × 24 × 365.
    None if disabled, data missing, or below threshold.
    """
    if not cfg.get("confluence", "funding_elevation", "enabled", default=True):
        return None
    if funding_rate is None:
        return None
    ann = float(funding_rate) * 24.0 * 365.0
    threshold = cfg.get(
        "confluence", "funding_elevation", "annualized_threshold", default=0.20
    )
    if abs(ann) <= threshold:
        return None
    # Longs paying → shorts profit from carry → counter-carry direction.
    return -1 if ann > 0 else 1


# ── Liquidation-cascade override ──────────────────────────────────────

def _liquidation_override(
    liq_score: Optional[float],
    realized_vol_recent: Optional[float],
    realized_vol_prior: Optional[float],
    cfg: PulseConfig,
) -> int:
    """Return ±2 if cluster + falling vol; else 0.

    liq_score = log1p(notional_usd_liquidated_last_15m / 1e6).
    Direction is +2 (BUY) for short-side liquidations (cascade down → bounce)
    or -2 (SHORT) for long-side — but Hyperliquid REST doesn't cleanly split
    so we infer from the sign of the preceding move (vol change alone isn't
    enough). Callers should pass sign-adjusted liq_score: positive for
    down-cascade, negative for up-cascade.
    """
    if not cfg.get("confluence", "liquidation", "enabled", default=True):
        return 0
    if liq_score is None or realized_vol_recent is None or realized_vol_prior is None:
        return 0
    thr = cfg.get("confluence", "liquidation", "cluster_log_threshold", default=2.0)
    if abs(liq_score) < thr:
        return 0
    vol_fall_ratio = cfg.get("confluence", "liquidation", "vol_fall_ratio", default=0.8)
    if realized_vol_recent >= realized_vol_prior * vol_fall_ratio:
        return 0
    return 2 if liq_score > 0 else -2


# ── Book imbalance factor ─────────────────────────────────────────────

def _score_book_imbalance(book_imbalance: Optional[float], cfg: PulseConfig) -> int:
    if not cfg.get("confluence", "book_imbalance", "enabled", default=True):
        return 0
    if book_imbalance is None:
        return 0
    thr = cfg.get("confluence", "book_imbalance", "abs_threshold", default=0.30)
    if book_imbalance > thr:
        return 1
    if book_imbalance < -thr:
        return -1
    return 0


# ── Persistence multiplier ────────────────────────────────────────────

def _persistence_mul(
    current_dir: int, prev_signal: Optional[str], cfg: PulseConfig
) -> float:
    """Confidence multiplier based on previous-cycle agreement."""
    if current_dir == 0 or prev_signal is None:
        return 1.0
    same = cfg.get("confluence", "persistence", "same_direction_mul", default=1.2)
    flip = cfg.get("confluence", "persistence", "flip_mul", default=0.8)
    neutral = cfg.get("confluence", "persistence", "neutral_prev_mul", default=1.0)
    if prev_signal == "NEUTRAL":
        return float(neutral)
    prev_dir = 1 if prev_signal == "BUY" else -1
    return float(same if prev_dir == current_dir else flip)


# ── Main scoring function ────────────────────────────────────────────

def score_pulse(
    report: dict,
    signal_threshold: Optional[float] = None,
    backtest_mode: bool = False,
    support: Optional[float] = None,
    resistance: Optional[float] = None,
    last_4h_close_ts: Optional[datetime] = None,
    tsmom_direction: Optional[int] = None,
    tsmom_strength: Optional[float] = None,
    regime_mode: str = "mixed",
    liquidation_score: Optional[float] = None,
    realized_vol_recent: Optional[float] = None,
    realized_vol_prior: Optional[float] = None,
    book_imbalance: Optional[float] = None,
    prev_signal: Optional[str] = None,
    ema_liquidity_ok: bool = True,
    cfg: Optional[PulseConfig] = None,
) -> dict:
    """Score a pulse report → {signal, confidence, breakdown, ...}.

    v3 changes:
      * Per-TF normalization: tf_signal = (Σ indicator_scores) / max_per_tf.
      * Weighted sum of tf_signals (bounded in ~[-1.3, 1.3]).
      * Regime on/off gate (configurable).
      * AND-gate with TSMOM when tsmom_direction provided (non-None).
      * Funding-elevation and liquidation-cascade overrides.
      * Persistence multiplier on confidence.
      * Order-flow score capped via YAML.
    """
    cfg = cfg or get_config()
    if signal_threshold is None:
        signal_threshold = cfg.get("confluence", "signal_threshold", default=DEFAULT_SIGNAL_THRESHOLD)

    timeframes = report.get("timeframes", {})
    spot_price = report.get("spot_price")
    vwap_daily = report.get("vwap_daily")
    vwap_position = report.get("vwap_position", 0)
    premium_pct = report.get("premium_pct", 0.0)
    funding_delta = report.get("funding_delta")
    funding_rate = report.get("funding_rate")

    tf_weights_cfg = cfg.get("confluence", "tf_weights", default=TF_WEIGHTS) or TF_WEIGHTS
    of_weight = float(cfg.get("confluence", "order_flow_weight", default=ORDER_FLOW_WEIGHT))

    # Regime gating ---------------------------------------------------
    use_onoff_gate = bool(
        cfg.get("confluence", "regime_onoff_gate", "enabled", default=True)
    )
    indicator_muls = _regime_indicator_muls(regime_mode, cfg) if not use_onoff_gate else None

    # Per-TF normalization --------------------------------------------
    weighted_sum = 0.0
    breakdown: Dict[str, float] = {}
    dominant_tf: Optional[str] = None
    dominant_tf_abs = 0.0

    for tf, base_weight in tf_weights_cfg.items():
        tf_data_raw = timeframes.get(tf, {}) or {}
        if not tf_data_raw:
            continue

        if tf == "4h":
            weight = _4h_staleness_weight(base_weight, last_4h_close_ts)
        else:
            weight = base_weight

        # Regime gating: zero out disabled indicators
        tf_data = _apply_regime_gate(tf_data_raw, regime_mode, cfg) if use_onoff_gate else tf_data_raw

        # EMA liquidity gate: disable EMA on 1h/15m if volume too low
        if (
            cfg.get("confluence", "ema_liquidity_gate", "enabled", default=True)
            and not ema_liquidity_ok
            and tf in ("15m", "1h")
        ):
            tf_data = dict(tf_data)
            tf_data["ema_cross"] = None

        rsi_s = _score_rsi(tf_data.get("rsi"))
        macd_s = _score_macd(tf_data.get("macd_hist"), tf_data.get("_macd_direction"))
        bb_s = _score_bb(tf_data.get("bb_pct"))
        ema_s = _score_ema_cross(
            tf_data.get("ema_cross"),
            tf_data.get("_ema9"),
            tf_data.get("_ema21"),
            tf_data.get("atr"),
        )

        # Count valid indicators for the PER-TF normalization denominator
        pairs = [
            ("rsi", tf_data.get("rsi"), rsi_s),
            ("macd", tf_data.get("macd_hist"), macd_s),
            ("bb", tf_data.get("bb_pct"), bb_s),
            ("ema", tf_data.get("ema_cross"), ema_s),
        ]
        indicator_sum = 0
        n_valid = 0
        for group, val, s in pairs:
            if val is None:
                continue
            n_valid += 1
            if indicator_muls is not None:
                s = s * indicator_muls.get(group, 1.0)
            indicator_sum += s

        pattern_s = _score_patterns(
            tf_data.get("patterns", []),
            spot_price, support, resistance, vwap_daily,
        )
        has_pattern = bool(tf_data.get("patterns"))

        max_per_tf = max(n_valid + (2 if has_pattern else 0), 1)
        # Per-TF normalization → bounded in [-1, +1] before vol amp
        tf_signal_raw = (indicator_sum + pattern_s) / max_per_tf
        vol_amp = _volume_amplifier(tf_data.get("rel_volume"))
        tf_signal = tf_signal_raw * vol_amp  # ≤ 1.5 × 1.0
        # Hard clamp so volume-amp cant overwhelm sum of weights
        tf_signal = max(-1.5, min(1.5, tf_signal))

        weighted = weight * tf_signal
        weighted_sum += weighted
        breakdown[tf] = round(weighted, 4)

        if abs(weighted) > dominant_tf_abs:
            dominant_tf_abs = abs(weighted)
            dominant_tf = tf

    # Order flow ------------------------------------------------------
    of_score, n_of = _score_order_flow(
        premium_pct,
        funding_delta,
        vwap_position,
        backtest_mode=backtest_mode,
        small_premium_thr=cfg.get(
            "confluence", "premium_tiers", "small_threshold_pct", default=0.05
        ),
        large_premium_thr=cfg.get(
            "confluence", "premium_tiers", "large_threshold_pct", default=0.15
        ),
        cap_abs=cfg.get("confluence", "order_flow_score_cap", default=3),
    )
    # Normalize OF score: cap_abs is the max magnitude (±3) → tf_signal ∈ [-1, 1]
    of_cap = max(1, int(cfg.get("confluence", "order_flow_score_cap", default=3)))
    of_normed = of_score / of_cap
    book_s = _score_book_imbalance(book_imbalance, cfg)
    of_plus_book = max(-1.0, min(1.0, of_normed + book_s * 0.5))
    of_weighted = of_weight * of_plus_book
    weighted_sum += of_weighted
    breakdown["order_flow"] = round(of_weighted, 4)
    breakdown["book_imbalance"] = round(book_s * 0.5, 4) if book_s else 0.0

    # Normalized_score now lives in ~[-sum(weights)-OF_w, +same] ≈ [-1.3, 1.3]
    normalized = max(-1.0, min(1.0, weighted_sum))

    # Funding-elevation override --------------------------------------
    override_reason: Optional[str] = None
    elev_dir = _funding_elevation_override(funding_rate, cfg)
    if elev_dir is not None:
        # Force direction; keep |normalized| at floor(0.5) so confidence is nonzero
        forced_mag = max(abs(normalized), 0.5)
        normalized = forced_mag * elev_dir
        override_reason = "funding_elevation"

    # Liquidation-cascade override ------------------------------------
    liq_dir_score = _liquidation_override(
        liquidation_score, realized_vol_recent, realized_vol_prior, cfg
    )
    if liq_dir_score != 0:
        # Add ±0.5 contribution clamped
        normalized = max(-1.0, min(1.0, normalized + 0.25 * liq_dir_score))
        override_reason = override_reason or "liquidation_cascade"

    # TSMOM AND-gate --------------------------------------------------
    tsmom_gated_out = False
    if tsmom_direction is not None:
        if tsmom_direction == 0:
            tsmom_gated_out = True
            normalized = 0.0
        else:
            cand_dir = 1 if normalized > signal_threshold else -1 if normalized < -signal_threshold else 0
            if cand_dir != 0 and cand_dir != tsmom_direction:
                # Confluence disagrees with TSMOM → no trade
                tsmom_gated_out = True
                normalized = 0.0

    # Persistence multiplier on confidence ----------------------------
    direction = 1 if normalized > signal_threshold else -1 if normalized < -signal_threshold else 0
    pers_mul = _persistence_mul(direction, prev_signal, cfg)
    adjusted_norm = max(-1.0, min(1.0, normalized * pers_mul))

    # Signal determination --------------------------------------------
    if adjusted_norm > signal_threshold:
        signal = "BUY"
    elif adjusted_norm < -signal_threshold:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    confidence = min(abs(adjusted_norm) / CONFIDENCE_DIVISOR, 1.0)

    if dominant_tf is None:
        dominant_tf = "15m"

    # SL/TP (informational only in v3) --------------------------------
    sl_tf_idx = max(_TF_ORDER.index(dominant_tf), _TF_ORDER.index("15m"))
    sl_tf = _TF_ORDER[sl_tf_idx]
    sl_atr = None
    if sl_tf in timeframes:
        sl_atr = timeframes[sl_tf].get("atr")

    stop_loss = None
    take_profit = None
    if spot_price is not None and sl_atr is not None and sl_atr > 0:
        if signal == "BUY":
            stop_loss = round(spot_price - 2.0 * sl_atr, 2)
            take_profit = round(spot_price + 3.0 * sl_atr, 2)
        elif signal == "SHORT":
            stop_loss = round(spot_price + 2.0 * sl_atr, 2)
            take_profit = round(spot_price - 3.0 * sl_atr, 2)

    hold_minutes = HOLD_MINUTES.get(dominant_tf, 45)

    reasoning = _build_reasoning(
        signal, confidence, breakdown, timeframes, report,
        override_reason=override_reason,
        tsmom_direction=tsmom_direction,
        regime_mode=regime_mode,
    )

    return {
        "signal": signal,
        "confidence": round(confidence, 4),
        "normalized_score": round(adjusted_norm, 4),
        "raw_normalized_score": round(normalized, 4),
        "timeframe_bias": dominant_tf,
        "key_levels": {"support": support, "resistance": resistance},
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "hold_minutes": hold_minutes,
        "reasoning": reasoning,
        "volatility_flag": report.get("max_1m_move_pct", 0) > 1.0,
        "breakdown": breakdown,
        "signal_threshold": signal_threshold,
        "persistence_mul": round(pers_mul, 4),
        "override_reason": override_reason,
        "tsmom_direction": tsmom_direction,
        "tsmom_strength": tsmom_strength,
        "tsmom_gated_out": tsmom_gated_out,
        "regime_mode": regime_mode,
    }


# ── Templated reasoning ──────────────────────────────────────────────

def _build_reasoning(
    signal: str,
    confidence: float,
    breakdown: dict,
    timeframes: dict,
    report: dict,
    override_reason: Optional[str] = None,
    tsmom_direction: Optional[int] = None,
    regime_mode: str = "mixed",
) -> str:
    header_bits = []
    if tsmom_direction is not None:
        tsmom_tag = "TSMOM↑" if tsmom_direction > 0 else "TSMOM↓" if tsmom_direction < 0 else "TSMOM=0"
        header_bits.append(tsmom_tag)
    header_bits.append(f"regime={regime_mode}")
    if override_reason:
        header_bits.append(f"override={override_reason}")
    header = " ".join(header_bits)

    if signal == "NEUTRAL":
        return f"NEUTRAL ({confidence:.2f}) [{header}]: Insufficient confluence"

    sorted_tf = sorted(
        [(k, v) for k, v in breakdown.items() if k not in ("order_flow", "book_imbalance")],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    parts = []
    for tf, score in sorted_tf[:3]:
        tf_data = timeframes.get(tf, {})
        details = []
        if tf_data.get("rsi") is not None:
            details.append(f"RSI={tf_data['rsi']:.0f}")
        if tf_data.get("ema_cross"):
            details.append(f"EMA {'↑' if tf_data['ema_cross'] == 'bullish' else '↓'}")
        if tf_data.get("rel_volume") is not None and tf_data["rel_volume"] > 1.5:
            details.append(f"vol {tf_data['rel_volume']:.1f}×")
        if tf_data.get("patterns"):
            details.append(", ".join(tf_data["patterns"]))
        direction = "bullish" if score > 0 else "bearish"
        detail_str = f" [{', '.join(details)}]" if details else ""
        parts.append(f"{tf} {direction}{detail_str}")

    of = breakdown.get("order_flow", 0)
    if abs(of) > 0:
        of_parts = []
        if report.get("premium_pct") and abs(report["premium_pct"]) > 0.05:
            of_parts.append(f"premium {report['premium_pct']:+.2f}%")
        if report.get("funding_delta") and abs(report["funding_delta"]) > 0.00005:
            of_parts.append(f"funding Δ {'neg' if report['funding_delta'] < 0 else 'pos'}")
        if of_parts:
            parts.append(" + ".join(of_parts))

    return f"{signal} ({confidence:.2f}) [{header}]: {' + '.join(parts)}"
