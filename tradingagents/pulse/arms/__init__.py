"""Pulse v4 strategy arms: regime-switched entry logic.

* ``vpd_reversal``         — used in ``high_vol_trend`` regime
* ``vwap_mean_reversion``  — used in ``chop`` regime
* ``confluence``           — legacy multi-TF scorer (in
  ``tradingagents/agents/quant_pulse_engine.py:score_pulse_confluence``)
"""

from tradingagents.pulse.arms.vpd_reversal import score_pulse_vpd_reversal
from tradingagents.pulse.arms.vwap_mean_reversion import score_pulse_vwap_mr

__all__ = ["score_pulse_vpd_reversal", "score_pulse_vwap_mr"]
