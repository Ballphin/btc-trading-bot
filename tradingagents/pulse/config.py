"""Pulse scoring-config loader with atomic read + hot-reload watcher.

One global `get_config()` — reads YAML, caches parsed dict, re-parses
when the file's st_mtime changes, and validates on every load.

Validation is strict for numeric ranges but lenient for unknown keys
(forward-compat). Any failure falls back to the last-known-good config.

Additions for auto-tune v2:
    * ContextVar ``_active_config`` — per-task override used by backtests
      so the orchestrator can test candidate params without mutating the
      on-disk YAML (works across threadpool executors via
      ``contextvars.copy_context().run(...)``).
    * ``compute_config_hash()`` — semantic hash over the *merged* effective
      config + active regime + venue + data source. Byte-level YAML hashes
      produce spurious churn when ``regime_profiles:`` blocks are added or
      key-order differs; this function hashes canonical JSON instead.
    * ``deep_merge()`` / ``get_effective_config()`` — regime-profile overlay.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "pulse_scoring.yaml"
)


@dataclass
class PulseConfig:
    """Immutable snapshot of parsed YAML + metadata.

    ``active_regime`` / ``venue`` / ``data_source`` participate in
    :func:`compute_config_hash` so semantically-distinct runs (e.g. a
    bull-regime config replayed on Binance-2022 data) are not collapsed
    with the base-regime config in scorecard aggregations.
    """

    data: Dict[str, Any]
    source_path: Path
    mtime: float
    content_hash: str
    # Context stamps — carried through backtests and live pulses.
    active_regime: str = "base"       # base | bull | bear | sideways | ambiguous
    venue: str = "hyperliquid"        # hyperliquid | binance_futures | stitched
    data_source: str = "hyperliquid"  # matches HistoricalDataRouter labels

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

    def with_overrides(
        self,
        data: Optional[Dict[str, Any]] = None,
        active_regime: Optional[str] = None,
        venue: Optional[str] = None,
        data_source: Optional[str] = None,
    ) -> "PulseConfig":
        """Return a new PulseConfig with selected fields replaced.

        The returned config's ``content_hash`` is recomputed via
        :func:`compute_config_hash` so callers can rely on it to be a
        stable, semantic identifier for the effective behaviour.
        """
        merged = data if data is not None else self.data
        regime = active_regime if active_regime is not None else self.active_regime
        v = venue if venue is not None else self.venue
        ds = data_source if data_source is not None else self.data_source
        new_hash = compute_config_hash(merged, active_regime=regime, venue=v, data_source=ds)
        return PulseConfig(
            data=merged,
            source_path=self.source_path,
            mtime=self.mtime,
            content_hash=new_hash,
            active_regime=regime,
            venue=v,
            data_source=ds,
        )


_lock = threading.Lock()
_cached: Optional[PulseConfig] = None

# ── ContextVar override (auto-tune / backtest isolation) ─────────────
# When set, :func:`get_config` returns this instead of the on-disk config.
# Crosses threadpool boundaries only via ``contextvars.copy_context().run(...)``
# — callers that schedule work on executors MUST propagate the context
# explicitly. See :func:`use_config_override` for the correct pattern.
_active_config: ContextVar[Optional[PulseConfig]] = ContextVar(
    "pulse_active_config", default=None
)


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
        # Semantic hash — does NOT include active_regime (base = default).
        # :func:`compute_config_hash` re-stamps when a regime overlay is applied.
        content_hash=compute_config_hash(data, active_regime="base",
                                          venue="hyperliquid",
                                          data_source="hyperliquid"),
    )


def get_config(path: Optional[Path] = None, force_reload: bool = False) -> PulseConfig:
    """Return the current config.

    Thread-safe. Re-reads when file mtime changes. Falls back to last-good
    on parse/validation error.

    If a ContextVar override is active (see :func:`use_config_override`),
    returns that override instead — callers in the live scorer, backtest
    replay, and auto-tune orchestrator all route through this single entry.
    """
    override = _active_config.get()
    if override is not None and not force_reload:
        return override

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


# ── Semantic hashing ─────────────────────────────────────────────────

def _canonicalize(obj: Any) -> Any:
    """Recursively normalize ``obj`` so JSON serialization is deterministic.

    * dicts → key-sorted (str keys only)
    * lists → recursed element-wise (order preserved — order is meaningful
      for weight lists)
    * floats → rounded to 12 decimal places to avoid platform-specific
      float-repr churn
    * everything else — passed through
    """
    if isinstance(obj, dict):
        return {str(k): _canonicalize(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, float):
        return round(obj, 12)
    return obj


def compute_config_hash(
    merged_data: Dict[str, Any],
    *,
    active_regime: str = "base",
    venue: str = "hyperliquid",
    data_source: str = "hyperliquid",
) -> str:
    """Compute a semantic SHA-256 over the effective config + context stamps.

    Unlike a byte-level hash of the YAML, this:
      * is insensitive to YAML key ordering and formatting whitespace;
      * is insensitive to adding *unused* sections (e.g. ``regime_profiles:``
        when ``active_regime == "base"``);
      * IS sensitive to (a) the actually-merged effective config values,
        (b) the active regime name, (c) the venue / data source used for
        any backtest that produced this config.

    This prevents scorecard aggregations from scrambling when the YAML is
    re-serialized with different key order, and keeps Binance-tuned bear
    configs separate from hypothetical Hyperliquid-tuned bear configs.
    """
    # Scorecard churn guard: volatile metadata fields must NOT contribute
    # to the hash, otherwise every calibration run invalidates aggregations.
    _VOLATILE_TOP_LEVEL = frozenset({
        "calibrated_at", "calibration_window", "deflated_sharpe",
        "oos_sharpe", "pbo", "n_eff",
    })
    filtered = {
        k: v for k, v in (merged_data or {}).items()
        if k not in _VOLATILE_TOP_LEVEL
    }
    # If ``active_regime == "base"``, drop regime_profiles from the hash —
    # untuned profiles should not change the base hash.
    if active_regime == "base":
        filtered.pop("regime_profiles", None)
    payload = {
        "config": _canonicalize(filtered),
        "active_regime": active_regime,
        "venue": venue,
        "data_source": data_source,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── Deep merge + regime overlay ──────────────────────────────────────

def deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``patch`` into a deep copy of ``base``.

    dict-on-dict recurses; non-dict values are replaced.
    Never mutates the inputs. Used for regime-profile overlays.
    """
    out = copy.deepcopy(base) if base else {}
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_effective_config(
    active_regime: str = "base",
    *,
    base_config: Optional[PulseConfig] = None,
    venue: str = "hyperliquid",
    data_source: str = "hyperliquid",
) -> PulseConfig:
    """Return a PulseConfig with the given regime profile deep-merged.

    When ``active_regime`` is ``"base"`` (or absent / missing in the YAML),
    the base config is returned unchanged aside from venue/data_source
    stamping. Otherwise the profile block at
    ``regime_profiles.<active_regime>`` is deep-merged on top of base.
    """
    base = base_config if base_config is not None else get_config()
    data = base.data
    if active_regime in (None, "", "base"):
        return base.with_overrides(active_regime="base", venue=venue, data_source=data_source)
    overlay = (data.get("regime_profiles") or {}).get(active_regime)
    if not overlay:
        logger.warning(
            f"[PulseConfig] No regime_profiles.{active_regime} block — falling back to base"
        )
        return base.with_overrides(active_regime="base", venue=venue, data_source=data_source)
    merged = deep_merge(data, overlay)
    try:
        _validate(merged)
    except ValueError as e:
        logger.error(f"[PulseConfig] Regime {active_regime} overlay invalid: {e} — using base")
        return base.with_overrides(active_regime="base", venue=venue, data_source=data_source)
    return base.with_overrides(
        data=merged,
        active_regime=active_regime,
        venue=venue,
        data_source=data_source,
    )


# ── ContextVar override helpers ──────────────────────────────────────

@contextmanager
def use_config_override(cfg: PulseConfig) -> Iterator[PulseConfig]:
    """Temporarily install ``cfg`` as the active config for this context.

    Usage (sync)::

        with use_config_override(cfg):
            score_pulse(report)  # sees cfg via get_config()

    Usage (across executors — MUST copy the context explicitly)::

        ctx = contextvars.copy_context()
        with use_config_override(cfg):
            await loop.run_in_executor(None, ctx.run, worker_fn)

    The ContextVar is a thread-local-like primitive that is inherited only
    when a new task/thread is started via ``ctx.run(...)``. Forgetting to
    propagate the context means the worker silently sees the on-disk YAML
    — a class of bug that caused a correctness regression in an earlier
    iteration of this plan; keep the propagation explicit.
    """
    token = _active_config.set(cfg)
    try:
        yield cfg
    finally:
        _active_config.reset(token)


def get_active_override() -> Optional[PulseConfig]:
    """Return the currently-active override, or None. Exposed for tests."""
    return _active_config.get()


# ── Ensemble variant overlays (R.1) ──────────────────────────────────
#
# The pulse ensemble (plan §3) runs the same data-prep through N parallel
# configurations. Each variant lives as a YAML overlay under
# ``config/pulse_variants/<name>.yaml`` and is deep-merged onto the base
# ``pulse_scoring.yaml``. Overlays share the same schema and are validated
# the same way regime profiles are.
#
# Keeping variants as YAML files (rather than Python constants) means the
# operator can tweak gates without a redeploy and the diffs are reviewable.
# The canonical five variants are baseline / sr_symmetric / sr_breakout_gate
# / chart_patterns / strict — see the files in config/pulse_variants/.

VARIANTS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "config" / "pulse_variants"
)


def list_variant_names(variants_dir: Optional[Path] = None) -> list[str]:
    """Enumerate available variants by filename stem, deterministically sorted."""
    d = variants_dir or VARIANTS_DIR
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


def load_variant_overlay(
    name: str,
    variants_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Read a variant YAML and return its raw overlay dict.

    An empty / missing file is treated as the no-op overlay ``{}``, which
    is the correct semantics for the ``baseline`` variant. Keys under a
    top-level ``overlay:`` entry are unwrapped so variant files can also
    carry meta (``description:``) alongside the overlay itself.
    """
    d = variants_dir or VARIANTS_DIR
    path = d / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Pulse variant not found: {path}")
    raw = yaml.safe_load(path.read_bytes()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Variant {path} did not parse to a mapping")
    # Support both flat overlays and ``overlay:``-wrapped schemas.
    if "overlay" in raw and isinstance(raw["overlay"], dict):
        return raw["overlay"]
    # Strip meta-only keys so the overlay is clean to deep_merge.
    return {k: v for k, v in raw.items() if k != "description"}


def get_variant_config(
    name: str,
    *,
    base_config: Optional[PulseConfig] = None,
    variants_dir: Optional[Path] = None,
    active_regime: str = "base",
    venue: str = "hyperliquid",
    data_source: str = "hyperliquid",
) -> PulseConfig:
    """Return a PulseConfig with variant ``name`` deep-merged over base.

    Validation mirrors :func:`get_effective_config`: any out-of-range merge
    falls back to base with a warning rather than crashing the pulse loop
    (we never want a bad variant to take the whole ensemble down).
    """
    base = base_config if base_config is not None else get_config()
    if name == "baseline":
        return base.with_overrides(
            active_regime=active_regime, venue=venue, data_source=data_source,
        )
    try:
        overlay = load_variant_overlay(name, variants_dir=variants_dir)
    except FileNotFoundError:
        logger.warning(f"[PulseConfig] Variant {name} not found — using baseline")
        return base.with_overrides(
            active_regime=active_regime, venue=venue, data_source=data_source,
        )
    merged = deep_merge(base.data, overlay)
    try:
        _validate(merged)
    except ValueError as e:
        logger.error(
            f"[PulseConfig] Variant {name} overlay invalid: {e} — using baseline"
        )
        return base.with_overrides(
            active_regime=active_regime, venue=venue, data_source=data_source,
        )
    return base.with_overrides(
        data=merged,
        active_regime=active_regime,
        venue=venue,
        data_source=data_source,
    )
