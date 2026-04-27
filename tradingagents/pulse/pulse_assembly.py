"""Unified pulse-input assembly — used by BOTH live (server.py) and
backtest (pulse_backtest.py) so the same call signature reaches
score_pulse() in both paths.

Introduced to eliminate the v3 "silent drop" bug where the backtest called
score_pulse(report, backtest_mode=True) and quietly skipped 8 v3 inputs.

Usage:
    inputs = PulseInputs(
        report=report,
        signal_threshold=0.22,
        tsmom_direction=1, tsmom_strength=0.6,
        regime_mode="trend",
        # ... all fields required (None is explicit)
    )
    result = score_pulse_from_inputs(inputs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_VALID_REGIME_MODES = {"trend", "chop", "high_vol_trend", "mixed"}
_VALID_SR_SOURCES = {"pivot", "book", "both", "none"}
_VALID_TERNARY = {-1, 0, 1, None}
_VALID_TFS = {"1m", "5m", "15m", "1h", "4h"}


@dataclass
class PulseInputs:
    """All inputs to score_pulse(), unified into one struct.

    Required fields have no default — construction will fail if a caller
    forgets one. Optional market-data fields (book/liq/etc) default to None
    (explicit null) but the caller must *decide* whether to pass None; the
    dataclass does not silently fill them in.
    """

    # --- Core report ---
    report: dict

    # --- Scoring config ---
    signal_threshold: float
    backtest_mode: bool = False

    # --- Alpha layer inputs ---
    tsmom_direction: Optional[int] = None
    tsmom_strength: Optional[float] = None
    regime_mode: str = "mixed"

    # --- Overrides / microstructure ---
    realized_vol_recent: Optional[float] = None
    realized_vol_prior: Optional[float] = None
    liquidation_score: Optional[float] = None
    book_imbalance: Optional[float] = None

    # --- Loop state ---
    prev_signal: Optional[str] = None
    ema_liquidity_ok: bool = True

    # --- S/R (new) ---
    support: Optional[float] = None
    resistance: Optional[float] = None
    sr_source: str = "none"

    # --- Regime diagnostics (for parabolic soft-gate) ---
    z_4h_return: Optional[float] = None

    # --- Pulse v4 inputs (None when v4 disabled) ---
    vpd_signal: Optional[int] = None              # -1 / 0 / +1 / None
    liquidity_sweep_dir: Optional[int] = None     # -1 / 0 / +1 / None
    pattern_hits: Dict[str, List[str]] = field(default_factory=dict)

    # --- Config + misc ---
    cfg: Any = None  # PulseConfig; avoid circular import in type annotation

    def __post_init__(self) -> None:
        if self.regime_mode not in _VALID_REGIME_MODES:
            raise ValueError(
                f"regime_mode must be one of {_VALID_REGIME_MODES}, got {self.regime_mode!r}"
            )
        if self.sr_source not in _VALID_SR_SOURCES:
            raise ValueError(
                f"sr_source must be one of {_VALID_SR_SOURCES}, got {self.sr_source!r}"
            )
        if not (0.0 < float(self.signal_threshold) <= 1.0):
            raise ValueError(
                f"signal_threshold must be in (0, 1], got {self.signal_threshold}"
            )
        if self.tsmom_direction is not None and self.tsmom_direction not in (-1, 0, 1):
            raise ValueError(
                f"tsmom_direction must be -1/0/1/None, got {self.tsmom_direction}"
            )
        if self.prev_signal is not None and self.prev_signal not in ("BUY", "SHORT", "NEUTRAL"):
            raise ValueError(
                f"prev_signal must be BUY/SHORT/NEUTRAL/None, got {self.prev_signal!r}"
            )
        if self.vpd_signal not in _VALID_TERNARY:
            raise ValueError(
                f"vpd_signal must be -1/0/1/None, got {self.vpd_signal!r}"
            )
        if self.liquidity_sweep_dir not in _VALID_TERNARY:
            raise ValueError(
                f"liquidity_sweep_dir must be -1/0/1/None, got {self.liquidity_sweep_dir!r}"
            )
        if not isinstance(self.pattern_hits, dict):
            raise ValueError(
                f"pattern_hits must be a dict, got {type(self.pattern_hits).__name__}"
            )
        for name, tfs in self.pattern_hits.items():
            if not isinstance(name, str):
                raise ValueError(f"pattern_hits keys must be strings, got {type(name).__name__}")
            if not isinstance(tfs, list):
                raise ValueError(f"pattern_hits[{name!r}] must be a list, got {type(tfs).__name__}")
            for tf in tfs:
                if tf not in _VALID_TFS:
                    raise ValueError(
                        f"pattern_hits[{name!r}] contains invalid tf {tf!r}; expected one of {_VALID_TFS}"
                    )

    def as_score_kwargs(self) -> dict:
        """Return the kwargs dict for the legacy score_pulse() signature.

        Keeps compat with existing tests that use positional `report` + kwargs.
        """
        return {
            "signal_threshold": self.signal_threshold,
            "backtest_mode": self.backtest_mode,
            "support": self.support,
            "resistance": self.resistance,
            "tsmom_direction": self.tsmom_direction,
            "tsmom_strength": self.tsmom_strength,
            "regime_mode": self.regime_mode,
            "liquidation_score": self.liquidation_score,
            "realized_vol_recent": self.realized_vol_recent,
            "realized_vol_prior": self.realized_vol_prior,
            "book_imbalance": self.book_imbalance,
            "prev_signal": self.prev_signal,
            "ema_liquidity_ok": self.ema_liquidity_ok,
            "z_4h_return": self.z_4h_return,
            "sr_source": self.sr_source,
            "cfg": self.cfg,
            "vpd_signal": self.vpd_signal,
            "liquidity_sweep_dir": self.liquidity_sweep_dir,
            "pattern_hits": self.pattern_hits,
        }


def score_pulse_from_inputs(inputs: PulseInputs) -> dict:
    """Single entry point that both live and backtest paths must use.

    This exists so `grep score_pulse_from_inputs` gives you every call
    site; any new v3 input added to PulseInputs is automatically threaded.
    """
    from tradingagents.agents.quant_pulse_engine import score_pulse
    return score_pulse(inputs.report, **inputs.as_score_kwargs())
