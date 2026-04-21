"""Pulse ensemble scoring — R.2.

Runs the same ``PulseInputs`` object through N variant configurations and
emits one pulse entry per variant. All variants share a single
``ensemble_tick_id`` so the verifier can correlate disagreements.

Key invariants (from the adversarial review):

* **Sequential execution.** Five variants × ~5 ms each is <50 ms total;
  parallelism would be premature and reintroduces the mutation-race
  concern the SSE critique flagged.
* **Deepcopy inputs per variant** (MEDIUM #10) — ``score_pulse_from_inputs``
  is documented pure today, but the ensemble is the thing that would
  quietly corrupt if that contract is ever violated, so we defend.
* **UUID-suffixed tick id** (HIGH #6) — prevents collisions when the
  scheduler and a manual ``/api/pulse/run`` fire within the same ISO
  second.
* **Per-variant write isolation.** Failures in one variant never abort
  the others; baseline is always attempted first so legacy consumers
  keep working even if a variant overlay is malformed.

The actual file layout is owned by the caller (server.py). This module
only *computes* — it does not write to disk.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

from tradingagents.pulse.config import PulseConfig, get_variant_config, list_variant_names
from tradingagents.pulse.pulse_assembly import PulseInputs, score_pulse_from_inputs

logger = logging.getLogger(__name__)


def generate_ensemble_tick_id(now: Optional[datetime] = None) -> str:
    """``<iso_utc>-<uuid4[:8]>`` — collision-proof across scheduler + manual runs."""
    ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds").replace("+00:00", "Z")
    return f"{ts}-{uuid4().hex[:8]}"


def score_variant(
    base_inputs: PulseInputs,
    variant_name: str,
    *,
    active_regime: str = "base",
    venue: str = "hyperliquid",
    data_source: str = "hyperliquid",
) -> dict:
    """Score a single variant against ``base_inputs``.

    The inputs are deepcopied so any in-place mutation by future
    scoring code cannot leak across variants. The returned dict is the
    raw ``score_pulse_from_inputs`` result — the caller is responsible
    for stamping ``config_name`` / ``ensemble_tick_id`` / persistence.
    """
    variant_cfg = get_variant_config(
        variant_name,
        base_config=base_inputs.cfg if base_inputs.cfg is not None else None,
        active_regime=active_regime,
        venue=venue,
        data_source=data_source,
    )
    # Deepcopy: see module docstring. PulseInputs is a dataclass so the
    # cheap form is `replace` on top of the deepcopied report/state.
    inputs_copy = replace(
        base_inputs,
        report=copy.deepcopy(base_inputs.report) if base_inputs.report else base_inputs.report,
        cfg=variant_cfg,
        signal_threshold=float(
            variant_cfg.get("confluence", "signal_threshold",
                             default=base_inputs.signal_threshold or 0.22),
        ),
    )
    return score_pulse_from_inputs(inputs_copy)


def score_ensemble(
    base_inputs: PulseInputs,
    *,
    variant_names: Optional[Iterable[str]] = None,
    active_regime: str = "base",
    venue: str = "hyperliquid",
    data_source: str = "hyperliquid",
    ensemble_tick_id: Optional[str] = None,
) -> Dict[str, dict]:
    """Score ``base_inputs`` under every configured variant.

    Returns ``{variant_name: result_dict}``. A failing variant logs and
    is omitted from the return — it does NOT raise, so a bad overlay
    cannot take down the live pulse loop.

    ``ensemble_tick_id`` is stamped onto every result dict (and
    generated if not supplied) so downstream writers can round-trip it
    verbatim into the JSONL schema.
    """
    names: List[str] = list(variant_names) if variant_names is not None else list_variant_names()
    if not names:
        names = ["baseline"]
    tick_id = ensemble_tick_id or generate_ensemble_tick_id()

    out: Dict[str, dict] = {}
    # Baseline first — if everything else fails we still have the legacy
    # behaviour.
    ordered = sorted(names, key=lambda n: (n != "baseline", n))
    for name in ordered:
        try:
            result = score_variant(
                base_inputs, name,
                active_regime=active_regime, venue=venue, data_source=data_source,
            )
            result["config_name"] = name
            result["ensemble_tick_id"] = tick_id
            out[name] = result
        except Exception as e:
            logger.error(
                "[Ensemble] variant %s failed: %s — skipping, ensemble continues",
                name, e,
            )
            continue
    return out
