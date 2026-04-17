#!/usr/bin/env python
"""Pulse calibration — record calibration metadata into ``config/pulse_scoring.yaml``.

This CLI inspects recent pulse history and:
  1. Prints current calibration metadata (engine_version, calibrated_at, deflated_sharpe).
  2. Optionally stamps ``calibrated_at`` / ``calibration_window`` / ``deflated_sharpe``
     into the YAML if ``--commit`` is passed.
  3. Gates on minimum sample (default 500 scored pulses) and probability-of-backtest-
     overfitting checks to prevent spurious calibration updates.

Usage:
    python scripts/pulse_calibrate.py                          # dry-run, prints status
    python scripts/pulse_calibrate.py --commit                 # write metadata
    python scripts/pulse_calibrate.py --min-scored 200         # relax sample gate
    python scripts/pulse_calibrate.py --n-strategies 8         # for deflated Sharpe
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tradingagents.pulse.stats import (  # noqa: E402
    deflated_sharpe,
    effective_sample_size,
    sharpe_confidence_interval,
    sharpe_ratio,
)

CONFIG_PATH = ROOT / "config" / "pulse_scoring.yaml"
PULSE_DIR = ROOT / "eval_results" / "pulse"


def _collect_rets_1h(engine_version: int = 3) -> List[float]:
    """Aggregate +1h net returns across all tickers."""
    rets: List[float] = []
    if not PULSE_DIR.exists():
        return rets
    for td in PULSE_DIR.iterdir():
        if not td.is_dir():
            continue
        path = td / "pulse.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("engine_version") != engine_version:
                    continue
                if not e.get("scored"):
                    continue
                r = e.get("return_+1h")
                if isinstance(r, (int, float)):
                    rets.append(float(r))
    return rets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                    help="Write calibration metadata to the YAML (default: dry-run)")
    ap.add_argument("--min-scored", type=int, default=500,
                    help="Minimum scored sample to allow calibration (default 500)")
    ap.add_argument("--n-strategies", type=int, default=1,
                    help="Number of strategies tried (for Deflated-Sharpe correction)")
    ap.add_argument("--engine-version", type=int, default=3,
                    help="Engine version to calibrate (default 3)")
    args = ap.parse_args()

    if not CONFIG_PATH.exists():
        print(f"[error] {CONFIG_PATH} not found", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    print("# Current calibration metadata")
    for k in ("engine_version", "calibrated_at", "calibration_window", "deflated_sharpe"):
        print(f"  {k}: {cfg.get(k)}")

    rets = _collect_rets_1h(engine_version=args.engine_version)
    n = len(rets)
    print(f"\n# Collected scored +1h returns: n={n}")

    if n < args.min_scored:
        print(
            f"\n[skip] Below min sample ({args.min_scored}). "
            f"Collect more pulses before calibration."
        )
        if args.commit:
            print("[skip] --commit ignored (sample gate failed)")
        return

    sr = sharpe_ratio(rets, periods_per_year=525_600 / 60)  # +1h → 8760 periods/yr
    n_eff = effective_sample_size(rets, max_lag=24)
    ds = deflated_sharpe(
        in_sample_sharpe=sr, n_params=args.n_strategies, n_obs=int(n_eff),
    )
    sr_lo, sr_hi = sharpe_confidence_interval(sr, n_eff=n_eff)

    print("\n# Stats")
    print(f"  Sharpe (annualized, +1h): {sr:.3f}")
    print(f"  Sharpe CI 95%:            [{sr_lo:.3f}, {sr_hi:.3f}]")
    print(f"  N_eff (Newey-West):       {n_eff:.1f}")
    print(f"  Deflated Sharpe:          {ds:.3f}  (n_strategies={args.n_strategies})")

    # Gate: deflated Sharpe must be positive to calibrate
    if ds <= 0:
        print(
            "\n[skip] Deflated Sharpe ≤ 0 — don't calibrate to spurious edge."
        )
        return

    if not args.commit:
        print("\n[dry-run] Pass --commit to record calibration metadata.")
        return

    now = datetime.now(timezone.utc).isoformat()
    cfg["calibrated_at"] = now
    cfg["calibration_window"] = {
        "n_observations": int(n),
        "n_effective": round(float(n_eff), 2),
        "engine_version": int(args.engine_version),
    }
    cfg["deflated_sharpe"] = round(float(ds), 4)

    tmp = CONFIG_PATH.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    tmp.replace(CONFIG_PATH)
    print(f"\n[commit] Wrote calibration metadata to {CONFIG_PATH}")
    print(f"  calibrated_at = {now}")
    print(f"  deflated_sharpe = {ds:.4f}")


if __name__ == "__main__":
    main()
