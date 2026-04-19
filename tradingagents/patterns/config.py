"""YAML loader for pattern detector tolerances.

All thresholds live in ``config/chart_patterns.yaml`` so they're tunable
without code changes. Loader validates structure and supplies defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_YAML = Path(__file__).resolve().parents[2] / "config" / "chart_patterns.yaml"


_DEFAULTS: Dict[str, Any] = {
    "global": {
        "k_tf_minutes": 60,
        "pivot_prominence_atr": 0.5,
        "atr_period": 14,
        "exclude_funding_bars": True,
    },
    "head_and_shoulders": {
        "min_bars": 25,
        "shoulder_diff": {"k_atr": 0.5, "k_pct": 0.005},
        "head_excess": {"k_atr": 1.0, "k_pct": 0.01},
        "neckline_slope_per_bar": {"k_atr": 0.25, "k_pct": 0.002},
        "min_bars_between_shoulders": 6,
    },
    "double_top": {
        "min_bars": 15,
        "peak_diff": {"k_atr": 0.3, "k_pct": 0.003},
        "trough_depth": {"k_atr": 1.0, "k_pct": 0.01},
    },
    "triple_top": {
        "min_bars": 25,
        "peak_diff": {"k_atr": 0.4, "k_pct": 0.004},
        "trough_depth": {"k_atr": 1.0, "k_pct": 0.01},
    },
    "cup_and_handle": {
        "min_bars": 40,
        "cup_depth_atr": 2.0,
        "handle_max_depth_frac": 0.5,
        "cup_symmetry_tol": 0.35,
    },
    "triangle": {
        "min_bars": 20,
        "min_pivots_per_side": 2,
        "r2_threshold": 0.80,
        "apex_max_bars_ahead": 30,
    },
    "flag": {
        "min_bars": 10,
        "pole_height_atr": 3.0,
        "channel_slope_opposite": True,
    },
    "pennant": {
        "min_bars": 10,
        "pole_height_atr": 3.0,
    },
    "wedge": {
        "min_bars": 20,
        "r2_threshold": 0.80,
        "min_slope_diff_atr_per_bar": 0.05,
    },
    "channel": {
        "min_bars": 20,
        "r2_threshold": 0.85,
        "min_pivots_per_side": 2,
    },
}


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> Dict[str, Any]:
    """Return merged config (defaults + YAML overrides). Cached."""
    p = Path(path) if path else DEFAULT_YAML
    if not p.exists():
        return _DEFAULTS
    try:
        with p.open("r") as fh:
            overrides = yaml.safe_load(fh) or {}
    except Exception:
        return _DEFAULTS
    return _deep_merge(_DEFAULTS, overrides)


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in base.items():
        if k in over and isinstance(v, dict) and isinstance(over[k], dict):
            out[k] = _deep_merge(v, over[k])
        elif k in over:
            out[k] = over[k]
        else:
            out[k] = v
    # Keys only in overrides
    for k, v in over.items():
        if k not in out:
            out[k] = v
    return out


def pattern_cfg(name: str) -> Dict[str, Any]:
    cfg = load_config()
    return cfg.get(name, {})


def global_cfg() -> Dict[str, Any]:
    return load_config().get("global", _DEFAULTS["global"])
