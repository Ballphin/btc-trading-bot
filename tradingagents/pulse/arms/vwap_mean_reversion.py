"""VWAP mean-reversion arm — used in ``chop`` regime.

Long when ``spot < VWAP − k × ATR``, short when ``spot > VWAP + k × ATR``.

Crypto-specific guards (per WCT review):
    * Weekend disabled unconditionally (Sat 00:00 UTC → Sun 23:59 UTC).
    * Weekday funding-direction gate: long iff ``funding_rate < -0.005%/hr``,
      short iff ``funding_rate > +0.005%/hr``.
    * Skips when ``ATR < min_atr_pct_of_price × spot`` (dead market).

EV @ p=0.55 with TP=1.4×ATR, SL=1.8×ATR, fees=0.2%::

    EV = 0.55 × 1.4 − 0.45 × 1.8 − 0.2 = -0.24 ATR  ✗ (negative)

Therefore the validation protocol REQUIRES p ≥ 0.62 in pre-flight before
this arm is enabled. Acceptance is enforced in ``pulse_backtest`` rather
than at signal-generation time (the arm produces its proposal; the
economic gate at the dispatcher level filters by funding cost).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

DEFAULT_ATR_BAND_MUL = 1.5
DEFAULT_MIN_ATR_PCT = 0.0015
DEFAULT_SL_ATR_MUL = 1.8
DEFAULT_TP_ATR_MUL = 1.4
DEFAULT_HOLD_MINUTES = 30
DEFAULT_FUNDING_GATE = 0.00005   # 0.005% per hour


def _is_weekend_utc(ts: Optional[datetime]) -> bool:
    if ts is None:
        ts = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.weekday() >= 5     # 5 = Saturday, 6 = Sunday


def score_pulse_vwap_mr(
    *,
    spot_price: Optional[float],
    vwap_daily: Optional[float],
    atr_1h: Optional[float],
    funding_rate: Optional[float],
    now: Optional[datetime] = None,
    cfg: Any = None,
) -> Dict[str, Any]:
    band_mul = DEFAULT_ATR_BAND_MUL
    min_atr_pct = DEFAULT_MIN_ATR_PCT
    sl_mul = DEFAULT_SL_ATR_MUL
    tp_mul = DEFAULT_TP_ATR_MUL
    hold_minutes = DEFAULT_HOLD_MINUTES
    funding_gate = DEFAULT_FUNDING_GATE
    weekend_disabled = True
    if cfg is not None:
        band_mul = float(cfg.get("pulse_v4", "arms", "vwap_mean_reversion", "atr_band_mul", default=band_mul))
        min_atr_pct = float(cfg.get("pulse_v4", "arms", "vwap_mean_reversion", "min_atr_pct_of_price", default=min_atr_pct))
        sl_mul = float(cfg.get("pulse_v4", "arms", "vwap_mean_reversion", "sl_atr_mul", default=sl_mul))
        tp_mul = float(cfg.get("pulse_v4", "arms", "vwap_mean_reversion", "tp_atr_mul", default=tp_mul))
        hold_minutes = int(cfg.get("pulse_v4", "arms", "vwap_mean_reversion", "hold_minutes", default=hold_minutes))
        funding_gate = float(cfg.get("pulse_v4", "arms", "vwap_mean_reversion", "funding_gate_threshold", default=funding_gate))
        weekend_disabled = bool(cfg.get("pulse_v4", "arms", "vwap_mean_reversion", "weekend_disabled", default=weekend_disabled))

    def _neutral(reason: str) -> Dict[str, Any]:
        return {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "normalized_score": 0.0,
            "stop_loss": None,
            "take_profit": None,
            "hold_minutes": hold_minutes,
            "arm_used": "vwap_mean_reversion",
            "arm_reason": reason,
        }

    if weekend_disabled and _is_weekend_utc(now):
        return _neutral("weekend_disabled")
    if spot_price is None or vwap_daily is None or atr_1h is None or atr_1h <= 0:
        return _neutral("missing_inputs")
    if atr_1h < min_atr_pct * spot_price:
        return _neutral("atr_below_min_dead_market")

    upper = vwap_daily + band_mul * atr_1h
    lower = vwap_daily - band_mul * atr_1h
    if lower <= spot_price <= upper:
        return _neutral("inside_band")

    direction = 1 if spot_price < lower else -1

    # Funding-direction gate: only fade the crowd
    if funding_rate is not None:
        if direction == 1 and funding_rate >= -funding_gate:
            return _neutral("funding_gate_long")
        if direction == -1 and funding_rate <= funding_gate:
            return _neutral("funding_gate_short")

    if direction == 1:
        signal = "BUY"
        stop_loss = round(spot_price - sl_mul * atr_1h, 2)
        take_profit = round(spot_price + tp_mul * atr_1h, 2)
    else:
        signal = "SHORT"
        stop_loss = round(spot_price + sl_mul * atr_1h, 2)
        take_profit = round(spot_price - tp_mul * atr_1h, 2)

    return {
        "signal": signal,
        "confidence": 0.6,
        "normalized_score": float(direction) * 0.6,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "hold_minutes": hold_minutes,
        "arm_used": "vwap_mean_reversion",
        "arm_reason": "band_breach+funding_aligned",
    }
