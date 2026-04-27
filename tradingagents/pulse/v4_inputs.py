"""Compute v4-only pulse inputs (vpd_signal, liquidity_sweep_dir,
pattern_hits) from cached candle DataFrames.

Callers (``server.py`` live path and ``pulse_backtest.py`` historical
path) invoke ``compute_v4_inputs()`` only when ``pulse_v4.enabled`` is
True. Returns plain values that thread through ``PulseInputs`` without
modifying the legacy report shape.

Keeping this in a separate module from ``quant_pulse_data.py`` is a
deliberate parity guard — see BLOCKER #6 in the v2 plan. The legacy
report builder is byte-identical when v4 is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from tradingagents.pulse.liquidity_sweep import detect_liquidity_sweep
from tradingagents.pulse.patterns.candles import detect_all as detect_candles_all
from tradingagents.pulse.patterns.structural import detect_structural_all
from tradingagents.pulse.vpd import compute_vpd


@dataclass(frozen=True)
class V4Inputs:
    vpd_signal: Optional[int]                    # -1 / 0 / +1 / None
    liquidity_sweep_dir: Optional[int]           # -1 / 0 / +1 / None
    pattern_hits: Dict[str, List[str]]           # name -> [tf, ...]


def compute_v4_inputs(
    *,
    candles_by_tf: Dict[str, pd.DataFrame],
    atr_by_tf: Optional[Dict[str, Optional[float]]] = None,
    funding_rate: Optional[float] = None,
    cfg=None,
) -> V4Inputs:
    """Build the v4-only inputs from cached OHLCV.

    Args:
        candles_by_tf: maps tf-name → closed-bar OHLCV DataFrame. Must
            contain at least ``"1m"``; ``"15m"``, ``"1h"``, and ``"4h"``
            are used when present.
        atr_by_tf: maps tf-name → 1h/4h ATR (used for structural rules).
        funding_rate: latest hourly funding rate (used by sweep
            aligned-funding rejection).

    All None outputs when data is absent — no exceptions raised.
    """
    candles_by_tf = candles_by_tf or {}
    atr_by_tf = atr_by_tf or {}

    # --- VPD on 15m / 1h / 4h ---------------------------------------
    vpd_signal: Optional[int] = None
    for tf in ("15m", "1h", "4h"):
        df = candles_by_tf.get(tf)
        if df is None or df.empty or "close" not in df.columns or "volume" not in df.columns:
            continue
        prices = df["close"].astype(float).values
        volumes = df["volume"].astype(float).values
        lookback = 20
        threshold = -0.30
        if cfg is not None:
            lookback = int(cfg.get("pulse_v4", "vpd", "lookback_bars", default=20))
            threshold = float(cfg.get("pulse_v4", "vpd", "corr_threshold", default=-0.30))
        result = compute_vpd(prices, volumes, lookback_bars=lookback,
                             corr_threshold=threshold)
        if result.signal != 0:
            vpd_signal = int(result.signal)
            break       # first non-zero (15m has priority for high_vol_trend timing)

    # --- Liquidity sweep on 1m -------------------------------------
    liquidity_sweep_dir: Optional[int] = None
    df_1m = candles_by_tf.get("1m")
    if df_1m is not None and not df_1m.empty:
        lookback = 60
        reclaim_within = 10
        vol_mul = 2.0
        reject_aligned = True
        if cfg is not None:
            lookback = int(cfg.get("pulse_v4", "liquidity_sweep", "extreme_lookback_bars", default=60))
            reclaim_within = int(cfg.get("pulse_v4", "liquidity_sweep", "reclaim_within_bars", default=10))
            vol_mul = float(cfg.get("pulse_v4", "liquidity_sweep", "reclaim_volume_mul", default=2.0))
            reject_aligned = bool(cfg.get("pulse_v4", "liquidity_sweep", "reject_aligned_funding", default=True))
        sweep = detect_liquidity_sweep(
            df_1m,
            extreme_lookback_bars=lookback,
            reclaim_within_bars=reclaim_within,
            reclaim_volume_mul=vol_mul,
            funding_rate=funding_rate,
            reject_aligned_funding=reject_aligned,
        )
        if sweep.direction != 0:
            liquidity_sweep_dir = int(sweep.direction)

    # --- Pattern hits (candles per TF + structural on 1h/4h) -------
    pattern_hits: Dict[str, List[str]] = {}
    for tf, df in candles_by_tf.items():
        if df is None or df.empty:
            continue
        try:
            names = detect_candles_all(df)
        except Exception:
            names = []
        for name in names:
            pattern_hits.setdefault(name, []).append(tf)
    for tf in ("1h", "4h"):
        df = candles_by_tf.get(tf)
        if df is None or df.empty:
            continue
        atr = atr_by_tf.get(tf)
        kw = {"bandwidth": 8, "atr": atr,
              "candles_1m": candles_by_tf.get("1m")}
        if cfg is not None:
            kw["bandwidth"] = int(cfg.get("pulse_v4", "patterns", "structural", "kernel_bandwidth_bars", default=8))
            kw["symmetry_pct"] = float(cfg.get("pulse_v4", "patterns", "structural", "symmetry_tolerance_pct", default=0.015))
            kw["channel_atr_band_mul"] = float(cfg.get("pulse_v4", "patterns", "structural", "channel_atr_band_mul", default=0.5))
            kw["channel_min_extrema"] = int(cfg.get("pulse_v4", "patterns", "structural", "channel_min_extrema", default=6))
            kw["channel_min_bars"] = int(cfg.get("pulse_v4", "patterns", "structural", "channel_min_bars", default=18))
            kw["channel_max_volume_ratio"] = float(cfg.get("pulse_v4", "patterns", "structural", "channel_max_volume_ratio", default=0.7))
            kw["double_bottom_match_pct"] = float(cfg.get("pulse_v4", "patterns", "structural", "double_bottom_match_pct", default=0.02))
            kw["double_bottom_wick_ratio"] = float(cfg.get("pulse_v4", "patterns", "structural", "double_bottom_wick_ratio", default=0.5))
            kw["double_bottom_reclaim_minutes"] = int(cfg.get("pulse_v4", "patterns", "structural", "double_bottom_reclaim_minutes", default=15))
        try:
            hits = detect_structural_all(df, **kw)
        except Exception:
            hits = []
        for hit in hits:
            pattern_hits.setdefault(hit.name, []).append(tf)

    return V4Inputs(
        vpd_signal=vpd_signal,
        liquidity_sweep_dir=liquidity_sweep_dir,
        pattern_hits=pattern_hits,
    )
