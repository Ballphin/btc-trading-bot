"""VPD-reversal arm — used in ``high_vol_trend`` regime.

Fades exhaustion: enters opposite to the local trend when a bearish/
bullish VPD divergence appears AND either:

    * ``liquidation_score`` opposes the trade (longs liquidated → bounce
      → LONG; shorts liquidated → drop → SHORT), OR
    * realized vol is fading (``rv_recent < 0.7 × rv_prior``)

SL/TP are per-arm (not the legacy 2:3 mul). Worked EV at p=0.55 with
TP=2×ATR, SL=1.5×ATR, fees+funding=0.2%::

    EV = 0.55 × 2.0 − 0.45 × 1.5 − 0.2 = +0.225 ATR/trade   ✓
"""

from __future__ import annotations

from typing import Any, Dict, Optional

DEFAULT_SL_ATR_MUL = 1.5
DEFAULT_TP_ATR_MUL = 2.0
DEFAULT_HOLD_MINUTES = 90
DEFAULT_VOL_FADE_RATIO = 0.7


def score_pulse_vpd_reversal(
    *,
    spot_price: Optional[float],
    atr_1h: Optional[float],
    vpd_signal: Optional[int],
    liquidation_score: Optional[float],
    realized_vol_recent: Optional[float],
    realized_vol_prior: Optional[float],
    cfg: Any = None,
) -> Dict[str, Any]:
    """Return a v3-shaped pulse result dict, or NEUTRAL with a reason.

    Required positive: ``vpd_signal != 0``, ``spot_price`` and ``atr_1h``
    finite, AND the liquidation/vol-fade gate (HIGH #13).
    """
    sl_mul = DEFAULT_SL_ATR_MUL
    tp_mul = DEFAULT_TP_ATR_MUL
    hold_minutes = DEFAULT_HOLD_MINUTES
    vol_fade_ratio = DEFAULT_VOL_FADE_RATIO
    if cfg is not None:
        sl_mul = float(cfg.get("pulse_v4", "arms", "vpd_reversal", "sl_atr_mul", default=sl_mul))
        tp_mul = float(cfg.get("pulse_v4", "arms", "vpd_reversal", "tp_atr_mul", default=tp_mul))
        hold_minutes = int(cfg.get("pulse_v4", "arms", "vpd_reversal", "hold_minutes", default=hold_minutes))
        vol_fade_ratio = float(cfg.get("pulse_v4", "arms", "vpd_reversal", "vol_fade_ratio", default=vol_fade_ratio))

    def _neutral(reason: str) -> Dict[str, Any]:
        return {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "normalized_score": 0.0,
            "stop_loss": None,
            "take_profit": None,
            "hold_minutes": hold_minutes,
            "arm_used": "vpd_reversal",
            "arm_reason": reason,
        }

    if vpd_signal is None or vpd_signal == 0:
        return _neutral("no_vpd")
    if spot_price is None or atr_1h is None or atr_1h <= 0:
        return _neutral("missing_atr_or_price")

    # Direction: VPD signal +1 = bullish absorption → LONG; -1 = bearish
    # exhaustion → SHORT.
    direction = 1 if vpd_signal > 0 else -1

    # Gate: require liquidation cluster opposing the trade OR vol fading.
    liq_ok = False
    if liquidation_score is not None:
        if direction == 1 and liquidation_score < 0:
            liq_ok = True
        if direction == -1 and liquidation_score > 0:
            liq_ok = True
    vol_ok = False
    if realized_vol_recent is not None and realized_vol_prior is not None and realized_vol_prior > 0:
        vol_ok = realized_vol_recent < vol_fade_ratio * realized_vol_prior
    if not (liq_ok or vol_ok):
        return _neutral("gate_failed_no_liq_or_vol_fade")

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
        "confidence": 0.7,
        "normalized_score": float(direction) * 0.7,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "hold_minutes": hold_minutes,
        "arm_used": "vpd_reversal",
        "arm_reason": "vpd_div+liq" if liq_ok else "vpd_div+vol_fade",
    }
