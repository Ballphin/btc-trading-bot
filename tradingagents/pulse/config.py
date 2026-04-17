"""Pulse scoring-config loader with atomic read + hot-reload watcher.

One global `get_config()` — reads YAML, caches parsed dict, re-parses
when the file's st_mtime changes, and validates on every load.

Validation is strict for numeric ranges but lenient for unknown keys
(forward-compat). Any failure falls back to the last-known-good config.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "pulse_scoring.yaml"
)


@dataclass
class PulseConfig:
    """Immutable snapshot of parsed YAML + metadata."""

    data: Dict[str, Any]
    source_path: Path
    mtime: float
    content_hash: str

    def hash_short(self) -> str:
        return self.content_hash[:12]

    # Convenience accessors ------------------------------------------
    @property
    def engine_version(self) -> int:
        return int(self.data.get("engine_version", 3))

    @property
    def is_calibrated(self) -> bool:
        return self.data.get("calibrated_at") is not None

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


_lock = threading.Lock()
_cached: Optional[PulseConfig] = None


def _validate(data: Dict[str, Any]) -> None:
    """Raise ValueError on any param outside expected range."""
    iv = data.get("scheduler", {}).get("pulse_interval_minutes", 5)
    if not isinstance(iv, (int, float)) or iv < 1 or iv > 60:
        raise ValueError(f"pulse_interval_minutes out of range: {iv}")

    conf = data.get("confluence", {})
    threshold = conf.get("signal_threshold", 0.25)
    if not 0 < threshold < 1:
        raise ValueError(f"signal_threshold out of range: {threshold}")

    tf_weights = conf.get("tf_weights", {})
    if tf_weights:
        for tf, w in tf_weights.items():
            if not isinstance(w, (int, float)) or w < 0 or w > 1:
                raise ValueError(f"tf_weight[{tf}] out of range: {w}")

    persistence = conf.get("persistence", {})
    for k, v in persistence.items():
        if not isinstance(v, (int, float)) or v <= 0 or v > 3:
            raise ValueError(f"persistence.{k} out of range: {v}")

    fund_elev_thr = conf.get("funding_elevation", {}).get("annualized_threshold", 0.20)
    if not 0 < fund_elev_thr < 2:
        raise ValueError(f"funding_elevation.annualized_threshold out of range: {fund_elev_thr}")

    fr = data.get("forward_return", {})
    atr_mul = fr.get("atr_multiplier", 0.5)
    if not 0 < atr_mul < 5:
        raise ValueError(f"forward_return.atr_multiplier out of range: {atr_mul}")

    eg = data.get("edge_gate", {})
    req_sr = eg.get("required_deflated_is_sharpe", 0.3)
    if not -5 < req_sr < 10:
        raise ValueError(f"edge_gate.required_deflated_is_sharpe out of range: {req_sr}")


def _read_and_parse(path: Path) -> PulseConfig:
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} did not parse to a dict")
    _validate(data)
    return PulseConfig(
        data=data,
        source_path=path,
        mtime=path.stat().st_mtime,
        content_hash=hashlib.sha256(raw).hexdigest(),
    )


def get_config(path: Optional[Path] = None, force_reload: bool = False) -> PulseConfig:
    """Return the current config.

    Thread-safe. Re-reads when file mtime changes. Falls back to last-good
    on parse/validation error.
    """
    global _cached
    target_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with _lock:
        if (
            not force_reload
            and _cached is not None
            and _cached.source_path == target_path
        ):
            try:
                current_mtime = target_path.stat().st_mtime
                if current_mtime == _cached.mtime:
                    return _cached
            except FileNotFoundError:
                return _cached  # file deleted — keep last known
        try:
            _cached = _read_and_parse(target_path)
            logger.info(
                f"[PulseConfig] Loaded {target_path.name} "
                f"hash={_cached.hash_short()} "
                f"calibrated={_cached.is_calibrated}"
            )
            return _cached
        except Exception as e:
            logger.error(f"[PulseConfig] Failed to load {target_path}: {e}")
            if _cached is None:
                raise
            logger.warning("[PulseConfig] Using last-known-good config")
            return _cached


def write_config_atomic(new_data: Dict[str, Any], path: Optional[Path] = None) -> PulseConfig:
    """Atomic YAML write (tmp + os.replace) after validation.

    Used exclusively by the calibration CLI.
    """
    target_path = Path(path) if path else DEFAULT_CONFIG_PATH
    _validate(new_data)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_suffix(target_path.suffix + ".tmp")
    raw = yaml.safe_dump(new_data, sort_keys=False, default_flow_style=False)
    tmp.write_text(raw)
    os.replace(tmp, target_path)
    return get_config(target_path, force_reload=True)
