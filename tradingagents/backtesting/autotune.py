"""Pulse auto-tune orchestrator (Stage 2 — Phase A).

Walk-forward Bayesian-lite search over a curated 6-parameter space,
gated by PBO + bootstrap Deflated Sharpe + per-regime trade minimums.

Design (from ``pulse-final-autotune-regime-plan-6a59a0.md``):

    1. Generate N candidate configs via **Latin Hypercube Sampling**
       over :data:`SEARCH_SPACE` (6 continuous dims). LHS gives better
       space-filling than vanilla random sampling at small N.

    2. Split the user's historical window into K non-overlapping
       walk-forward folds with a 1-day purge gap between train and test
       (prevents leakage from overlapping indicator windows).

    3. For each fold, run :class:`PulseBacktestEngine` with the candidate
       as ``config_override``. Train Sharpe ranks candidates; the
       promoted subset runs on the held-out test window.

    4. **Winner selection** — not ``argmax(mean_oos_sharpe)`` which
       suffers max-of-N bias. Use :func:`bootstrap_sharpe_ci` to get a
       95%-lower-bound Sharpe per candidate; pick the highest
       lower-bound. Correctly accounts for noise at small N.

    5. **Verdict gate**::

        n_trades_oos < MIN_TRADES[regime]               → PROVISIONAL
        pbo > 0.5                                       → REJECT
        oos_sharpe / is_sharpe < 0.5                    → REJECT
        deflated_oos_sharpe < 0.30                      → REJECT
        regime_stratified_degradation                   → REJECT
        all gates pass                                  → PROPOSE

    6. ``PROPOSE`` → surface diff in UI; user clicks Apply (server
       re-checks gates defensively before writing YAML).

    7. ``PROVISIONAL`` → surface with banner "tune on 90+ days for a
       real proposal"; `/apply` endpoint returns 409.

Propose-only: configs are NEVER auto-applied. This is a deliberate
contract from the plan — preserves user control and the safety rail
against a bad Bayesian step auto-writing a YAML at 3am.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tradingagents.pulse.config import (
    PulseConfig,
    compute_config_hash,
    deep_merge,
    get_config,
    get_effective_config,
)
from tradingagents.pulse.stats import (
    bootstrap_sharpe_ci,
    deflated_sharpe,
    effective_sample_size,
    pbo_bootstrap,
    sharpe_ratio,
)

logger = logging.getLogger(__name__)


# ── Per-regime sample-size gates ─────────────────────────────────────
# Derived from Sharpe precision requirement: SE(Sharpe) ≈ 1/√N, so
# ``1.96/√N ≤ 0.10`` → N ≥ 384. We round to tidy numbers per regime
# reflecting data availability:
#   bull:    HL 2023-10+ has ample bull data → 400 is achievable
#   bear:    Pre-HL funding is Binance (slightly noisier) → 300 floor
#   sideways: lowest trade frequency by construction (higher threshold
#             filters more signals) → 200 floor, documented caveat
# These limits are referenced in the verdict logic and must match the
# adversarial-review consensus (see plan §BLOCKERS).
MIN_TRADES_OOS_BY_REGIME: Dict[str, int] = {
    "base":     400,
    "bull":     400,
    "bear":     300,
    "sideways": 200,
    "ambiguous": 200,
}

#: Minimum trades per fold to count it toward OOS stats. Folds below
#: this are dropped from aggregation; if fewer than 2 folds survive,
#: the entire tune is ``REJECT``.
MIN_TRADES_PER_FOLD: int = 20


# ── Search space ─────────────────────────────────────────────────────
# 6-parameter "Balanced" space (user-confirmed). Tighter ranges = less
# exploration but less overfitting risk. Ranges derived from historical
# BTC behavior + manual sanity checks on edge cases.
#
# Each entry maps a dotted YAML path → (low, high) inclusive.
SEARCH_SPACE: Dict[str, Tuple[float, float]] = {
    "confluence.signal_threshold":                         (0.18, 0.32),
    "confluence.tsmom_gate.counter_trend_confluence_mul":  (1.0,  1.6),
    "confluence.exits.buy_sl_atr_mul":                     (1.2,  2.8),
    "confluence.exits.buy_tp_atr_mul":                     (2.0,  4.5),
    "confluence.exits.short_crash_sl_atr_mul":             (1.0,  2.2),
    "confluence.exits.short_tp_atr_mul":                   (2.0,  4.5),
}


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class TuneSpec:
    """Configuration for a single auto-tune run.

    Instances serialize to JSON cleanly (via :func:`asdict`) so they can
    be persisted alongside the tune artifact.
    """

    ticker: str
    start_date: str                 # yyyy-mm-dd inclusive
    end_date: str                   # yyyy-mm-dd exclusive
    active_regime: str = "base"     # base|bull|bear|sideways|ambiguous
    n_folds: int = 3
    n_configs: int = 30             # Latin-Hypercube sample count
    pulse_interval_minutes: int = 15
    # Safety gates
    max_pbo: float = 0.50
    min_oos_over_is_ratio: float = 0.50
    min_deflated_sharpe: float = 0.30
    regime_floor_ratio: float = 0.80
    # Reproducibility
    seed: int = 42
    # Checkpointing
    checkpoint_dir: str = "results/autotune/_checkpoints"


@dataclass
class FoldResult:
    """Per-fold per-candidate outcome. One row per (fold, config) pair."""

    fold: int
    config_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    is_sharpe: float
    oos_sharpe: float
    is_n_trades: int
    oos_n_trades: int
    oos_returns: List[float] = field(default_factory=list)


@dataclass
class TuneReport:
    """Full tune result. Written to ``results/autotune/<ticker>_<ts>.json``."""

    spec: TuneSpec
    verdict: str                    # PROPOSE | PROVISIONAL | REJECT
    reasons: List[str]
    current_config_hash: str
    proposed_config: Dict[str, Any]
    proposed_config_hash: str
    diff: List[Dict[str, Any]]      # list of {path, old, new}
    metrics: Dict[str, Any]
    per_fold: List[Dict[str, Any]]
    artifact_path: Optional[str] = None
    ran_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── Latin Hypercube sampling ─────────────────────────────────────────

def sample_candidates(
    space: Dict[str, Tuple[float, float]],
    n: int,
    *,
    seed: int = 42,
) -> List[Dict[str, float]]:
    """Latin-Hypercube sample ``n`` points from the given space.

    For each dimension: stratify [0, 1) into ``n`` equal-width bins,
    place exactly one random point in each bin, then permute the bins.
    Scale to each dim's (low, high). Returns a list of dicts keyed by
    the dotted YAML path.

    LHS is better than vanilla random at small N because it guarantees
    marginal coverage: no axis is ever sampled only near one extreme.
    """
    if n < 1:
        return []
    rng = np.random.default_rng(seed)
    dims = list(space.keys())
    # unit-cube LHS
    out_unit = np.empty((n, len(dims)))
    for j in range(len(dims)):
        # One point per bin, random position within the bin, permuted.
        edges = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(edges)
        out_unit[:, j] = edges
    # Scale to parameter ranges
    samples: List[Dict[str, float]] = []
    for i in range(n):
        sample = {}
        for j, path in enumerate(dims):
            lo, hi = space[path]
            sample[path] = float(lo + out_unit[i, j] * (hi - lo))
        samples.append(sample)
    return samples


def candidate_to_config(
    base: PulseConfig,
    candidate: Dict[str, float],
    *,
    active_regime: str = "base",
    venue: str = "hyperliquid",
    data_source: str = "hyperliquid",
) -> PulseConfig:
    """Apply a flat candidate dict onto ``base`` and return a new PulseConfig.

    The candidate uses dotted YAML paths (e.g.
    ``"confluence.signal_threshold"``). We construct the nested overlay,
    deep-merge into base.data, and re-wrap via ``with_overrides`` so the
    content hash reflects the new values + regime + venue stamps.
    """
    overlay: Dict[str, Any] = {}
    for path, value in candidate.items():
        keys = path.split(".")
        node = overlay
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = float(value)
    merged = deep_merge(base.data, overlay)
    return base.with_overrides(
        data=merged,
        active_regime=active_regime,
        venue=venue,
        data_source=data_source,
    )


# ── Walk-forward fold splitting ──────────────────────────────────────

@dataclass
class Fold:
    """One train/test split. ``purge_days`` sits between the two."""

    train_start: str
    train_end: str
    test_start: str
    test_end: str


def make_folds(
    start_date: str,
    end_date: str,
    *,
    n_folds: int,
    train_ratio: float = 0.60,
    purge_days: int = 1,
) -> List[Fold]:
    """Build ``n_folds`` non-overlapping walk-forward folds.

    Layout (time →)::

        |fold1_train|purge|fold1_test|  |fold2_train|purge|fold2_test|  ...

    Folds are non-overlapping (no re-use of data across folds) — simpler
    than the classic purged-CPCV but adequate for the fold counts we
    use (3–5). No leakage because of the purge day between train/test
    within each fold.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    total_days = (end - start).days
    if total_days < n_folds * 4:
        raise ValueError(
            f"window {total_days}d too short for {n_folds} folds "
            f"(need ≥ {n_folds * 4}d)"
        )
    fold_days = total_days // n_folds
    folds: List[Fold] = []
    for i in range(n_folds):
        f_start = start + timedelta(days=i * fold_days)
        f_end = start + timedelta(days=(i + 1) * fold_days)
        train_days = max(1, int(fold_days * train_ratio))
        train_end = f_start + timedelta(days=train_days)
        test_start = train_end + timedelta(days=purge_days)
        if test_start >= f_end:
            # Fold too small for purge gap — skip it rather than silently
            # collapse train/test. Caller can reduce n_folds.
            continue
        folds.append(Fold(
            train_start=f_start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=f_end.strftime("%Y-%m-%d"),
        ))
    return folds


# ── Verdict logic ────────────────────────────────────────────────────

def _piecewise_regime_gate(
    proposed: float,
    current: float,
    *,
    ratio_floor: float = 0.80,
) -> bool:
    """Piecewise Apply gate across three Sharpe regimes.

    Behavior depends on the current config's Sharpe magnitude:
      * current > 0.5: require ``proposed >= ratio_floor × current``.
      * 0 ≤ current ≤ 0.5: require ``proposed ≥ current − 0.1``
        (absolute-delta band — 0.8× of a 0.2 Sharpe is meaningless).
      * current < 0: require ``proposed > 0``
        (absolute floor — never accept "less negative" as an improvement).

    Addresses SQR-5 from the adversarial review: a naive 0.8× ratio
    breaks at current<0 because 0.8 × -0.5 = -0.4 is *easier* to clear
    than -0.5, so a genuinely worse proposal could pass.
    """
    if current > 0.5:
        return proposed >= ratio_floor * current
    if current >= 0.0:
        return proposed >= current - 0.10
    return proposed > 0.0


def compute_verdict(
    *,
    is_sharpes: Sequence[float],
    oos_sharpes: Sequence[float],
    oos_n_trades_total: int,
    active_regime: str,
    pbo: float,
    deflated_oos_sharpe: float,
    n_folds_used: int,
    spec: TuneSpec,
    regime_sharpes_current: Optional[Dict[str, float]] = None,
    regime_sharpes_proposed: Optional[Dict[str, float]] = None,
) -> Tuple[str, List[str]]:
    """Apply the layered safety gates. Returns (verdict, reasons).

    Verdict values:
        PROPOSE      — all gates passed, safe to surface for Apply.
        PROVISIONAL  — sample-size gate failed but no evidence of harm;
                       surface for viewing; /apply returns 409.
        REJECT       — at least one hard gate failed.

    Evaluation order (short-circuit on first failure):
        0. folds_used >= 2               (basic sample integrity)
        1. oos_n_trades >= MIN[regime]   (precision gate)
        2. pbo < max_pbo                 (overfitting guard)
        3. oos/is >= min_oos_over_is     (generalization ratio)
        4. deflated_oos >= min_ds        (edge-quality gate)
        5. per-regime non-degradation    (if current/proposed provided)
    """
    reasons: List[str] = []

    if n_folds_used < 2:
        return "REJECT", [f"only {n_folds_used} folds produced useful trades"]

    min_trades = MIN_TRADES_OOS_BY_REGIME.get(
        active_regime, MIN_TRADES_OOS_BY_REGIME["base"],
    )
    if oos_n_trades_total < min_trades:
        reasons.append(
            f"OOS trades ({oos_n_trades_total}) below regime floor "
            f"({min_trades}) for regime='{active_regime}' — insufficient "
            f"precision for Apply"
        )
        return "PROVISIONAL", reasons

    if pbo > spec.max_pbo:
        return "REJECT", [
            f"PBO={pbo:.3f} exceeds max {spec.max_pbo:.2f} — overfit risk"
        ]

    is_mean = float(np.mean(is_sharpes)) if is_sharpes else 0.0
    oos_mean = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    if is_mean > 1e-6:
        ratio = oos_mean / is_mean
        if ratio < spec.min_oos_over_is_ratio:
            return "REJECT", [
                f"OOS/IS ratio={ratio:.3f} < min {spec.min_oos_over_is_ratio:.2f} — "
                f"IS-overfit signature"
            ]

    if deflated_oos_sharpe < spec.min_deflated_sharpe:
        return "REJECT", [
            f"Deflated OOS Sharpe={deflated_oos_sharpe:.3f} below "
            f"{spec.min_deflated_sharpe:.2f} — edge not distinguishable from "
            f"multiple-testing noise"
        ]

    # Per-regime degradation check — only when both maps are provided.
    if regime_sharpes_current and regime_sharpes_proposed:
        for reg, cur in regime_sharpes_current.items():
            prop = regime_sharpes_proposed.get(reg)
            if prop is None:
                continue
            if not _piecewise_regime_gate(prop, cur, ratio_floor=spec.regime_floor_ratio):
                return "REJECT", [
                    f"regime='{reg}' Sharpe degrades from {cur:.2f} → {prop:.2f} "
                    f"(piecewise gate failed)"
                ]

    reasons.append(
        f"all gates passed: n_oos={oos_n_trades_total} "
        f"pbo={pbo:.3f} ds_oos={deflated_oos_sharpe:.3f}"
    )
    return "PROPOSE", reasons


# ── Diff helper ──────────────────────────────────────────────────────

def compute_diff(
    current: Dict[str, Any],
    proposed: Dict[str, Any],
    paths: Sequence[str],
) -> List[Dict[str, Any]]:
    """Return ``[{path, old, new}]`` entries for each changed path.

    Uses the same dotted-path convention as :data:`SEARCH_SPACE`.
    Missing entries in ``current`` show ``old=None``.
    """
    def _get(d: Dict[str, Any], path: str) -> Any:
        node: Any = d
        for k in path.split("."):
            if not isinstance(node, dict) or k not in node:
                return None
            node = node[k]
        return node

    diff: List[Dict[str, Any]] = []
    for p in paths:
        old = _get(current, p)
        new = _get(proposed, p)
        # Compare with a float tolerance — LHS may give 0.220000001 vs 0.22.
        try:
            if old is not None and new is not None:
                if abs(float(old) - float(new)) < 1e-9:
                    continue
        except (TypeError, ValueError):
            pass
        if old != new:
            diff.append({
                "path": p,
                "old": old,
                "new": new,
                "delta": (float(new) - float(old)) if (
                    isinstance(old, (int, float)) and isinstance(new, (int, float))
                ) else None,
            })
    return diff


# ── Checkpoint I/O ───────────────────────────────────────────────────

class CheckpointLog:
    """Append-only JSONL log of completed (fold, config) backtests.

    Crash-safe: after each backtest, flush a row. On restart the
    orchestrator reads the log and skips already-completed cells.
    File lives at ``results/autotune/_checkpoints/<job_id>.jsonl``.
    """

    def __init__(self, checkpoint_path: Path):
        self.path = Path(checkpoint_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def load_completed(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Return {(fold, config_idx): row} of completed cells."""
        if not self.path.exists():
            return {}
        out: Dict[Tuple[int, int], Dict[str, Any]] = {}
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (int(row["fold"]), int(row["config_idx"]))
                out[key] = row
        return out


# ── Orchestrator ─────────────────────────────────────────────────────

BacktestFn = Callable[[str, str, str, PulseConfig, str], Dict[str, Any]]
"""Signature: (ticker, start, end, config_override, active_regime) → result dict.

The result must contain ``trade_returns`` (list of fractional returns).
:func:`default_backtest_fn` is the production wiring; tests inject
deterministic fakes.
"""


def default_backtest_fn(
    ticker: str,
    start: str,
    end: str,
    cfg: PulseConfig,
    active_regime: str,
) -> Dict[str, Any]:
    """Production backtest wiring: run ``PulseBacktestEngine`` with override.

    Extracts per-trade returns from the engine's signals so the
    orchestrator can bootstrap Sharpe CIs.
    """
    # Lazy import to avoid circular dependency: autotune.py sits in
    # ``tradingagents.backtesting`` and so does pulse_backtest.
    from tradingagents.backtesting.pulse_backtest import PulseBacktestEngine

    engine = PulseBacktestEngine(
        ticker=ticker,
        start_date=start,
        end_date=end,
        config_override=cfg,
        active_regime=active_regime,
    )
    result = engine.run()
    # Pull per-trade returns for CI computation. ``exit_return`` is set
    # on every signal that resolved (not timeout-with-no-candles).
    trade_returns: List[float] = []
    for s in result.get("signals", []):
        r = s.get("exit_return")
        if isinstance(r, (int, float)):
            trade_returns.append(float(r))
    result["trade_returns"] = trade_returns
    return result


@dataclass
class AutoTuner:
    """Run the walk-forward LHS sweep + verdict for a single :class:`TuneSpec`.

    Usage::

        tuner = AutoTuner(spec)
        report = tuner.run()

    Progress callback (SSE): the orchestrator calls
    ``progress_cb({"phase": ..., "fold": ..., "config_idx": ..., "total": ...})``
    after each backtest. Server wires it to SSE events.
    """

    spec: TuneSpec
    backtest_fn: BacktestFn = default_backtest_fn
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None
    job_id: Optional[str] = None

    def run(self) -> TuneReport:
        spec = self.spec
        logger.info(
            f"[AutoTune] Starting {spec.ticker} {spec.start_date}→{spec.end_date} "
            f"regime={spec.active_regime} folds={spec.n_folds} "
            f"n_configs={spec.n_configs}"
        )
        base_cfg = get_effective_config(spec.active_regime)
        candidates = sample_candidates(SEARCH_SPACE, spec.n_configs, seed=spec.seed)
        folds = make_folds(spec.start_date, spec.end_date, n_folds=spec.n_folds)
        if len(folds) < 2:
            return self._reject_report(
                base_cfg,
                reasons=[f"only {len(folds)} usable fold(s) — need ≥2"],
            )

        job_id = self.job_id or f"{spec.ticker}_{int(time.time())}"
        checkpoint = CheckpointLog(
            Path(spec.checkpoint_dir) / f"{job_id}.jsonl"
        )
        completed = checkpoint.load_completed()

        # Storage: per (fold, config) result row.
        rows: List[FoldResult] = []
        total_cells = len(folds) * len(candidates)
        cell_idx = 0

        for fi, fold in enumerate(folds):
            for ci, candidate in enumerate(candidates):
                cell_idx += 1
                self._emit_progress(
                    phase="backtest",
                    fold=fi, config_idx=ci,
                    total=total_cells, done=cell_idx,
                )
                key = (fi, ci)
                if key in completed:
                    row = completed[key]
                    rows.append(FoldResult(
                        fold=fi, config_idx=ci,
                        train_start=fold.train_start, train_end=fold.train_end,
                        test_start=fold.test_start, test_end=fold.test_end,
                        is_sharpe=float(row.get("is_sharpe", 0.0)),
                        oos_sharpe=float(row.get("oos_sharpe", 0.0)),
                        is_n_trades=int(row.get("is_n_trades", 0)),
                        oos_n_trades=int(row.get("oos_n_trades", 0)),
                        oos_returns=list(row.get("oos_returns", [])),
                    ))
                    continue

                cfg = candidate_to_config(
                    base_cfg, candidate,
                    active_regime=spec.active_regime,
                    venue=base_cfg.venue,
                    data_source=base_cfg.data_source,
                )
                try:
                    is_result = self.backtest_fn(
                        spec.ticker, fold.train_start, fold.train_end,
                        cfg, spec.active_regime,
                    )
                    oos_result = self.backtest_fn(
                        spec.ticker, fold.test_start, fold.test_end,
                        cfg, spec.active_regime,
                    )
                except Exception as e:
                    logger.warning(
                        f"[AutoTune] fold={fi} config={ci} backtest failed: {e}"
                    )
                    continue

                is_returns = is_result.get("trade_returns", [])
                oos_returns = oos_result.get("trade_returns", [])
                is_sr = sharpe_ratio(is_returns, periods_per_year=8760.0) \
                    if is_returns else 0.0
                oos_sr = sharpe_ratio(oos_returns, periods_per_year=8760.0) \
                    if oos_returns else 0.0
                row = FoldResult(
                    fold=fi, config_idx=ci,
                    train_start=fold.train_start, train_end=fold.train_end,
                    test_start=fold.test_start, test_end=fold.test_end,
                    is_sharpe=is_sr, oos_sharpe=oos_sr,
                    is_n_trades=len(is_returns),
                    oos_n_trades=len(oos_returns),
                    oos_returns=oos_returns,
                )
                rows.append(row)
                checkpoint.append({
                    "fold": fi, "config_idx": ci,
                    "is_sharpe": is_sr, "oos_sharpe": oos_sr,
                    "is_n_trades": len(is_returns),
                    "oos_n_trades": len(oos_returns),
                    "oos_returns": oos_returns,
                    "candidate": candidate,
                })

        self._emit_progress(phase="selection")

        winner_idx, winner_metrics = self._select_winner(candidates, rows)
        if winner_idx is None:
            return self._reject_report(
                base_cfg, reasons=["no candidate produced useful results"],
            )
        winner_cand = candidates[winner_idx]
        winner_cfg = candidate_to_config(
            base_cfg, winner_cand,
            active_regime=spec.active_regime,
            venue=base_cfg.venue,
            data_source=base_cfg.data_source,
        )

        verdict, reasons = compute_verdict(
            is_sharpes=winner_metrics["is_sharpes"],
            oos_sharpes=winner_metrics["oos_sharpes"],
            oos_n_trades_total=winner_metrics["oos_n_trades_total"],
            active_regime=spec.active_regime,
            pbo=winner_metrics["pbo"],
            deflated_oos_sharpe=winner_metrics["deflated_oos_sharpe"],
            n_folds_used=winner_metrics["n_folds_used"],
            spec=spec,
        )

        diff = compute_diff(
            current=base_cfg.data,
            proposed=winner_cfg.data,
            paths=list(SEARCH_SPACE.keys()),
        )

        report = TuneReport(
            spec=spec,
            verdict=verdict,
            reasons=reasons,
            current_config_hash=base_cfg.content_hash,
            proposed_config=winner_cand,
            proposed_config_hash=winner_cfg.content_hash,
            diff=diff,
            metrics=winner_metrics,
            per_fold=[asdict(r) for r in rows if r.config_idx == winner_idx],
        )

        artifact = self._write_artifact(report, job_id)
        report.artifact_path = str(artifact)
        self._emit_progress(phase="done", verdict=verdict)
        return report

    # ── internals ────────────────────────────────────────────────────

    def _select_winner(
        self,
        candidates: List[Dict[str, float]],
        rows: List[FoldResult],
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        """Pick the candidate with the highest bootstrap-lower OOS Sharpe.

        Aggregates all OOS returns across folds for each candidate,
        computes a 95% CI lower bound, and ranks. Also populates the
        metric bag the verdict needs: IS/OOS means, PBO, DSR, fold count.
        """
        by_cfg: Dict[int, List[FoldResult]] = {}
        for r in rows:
            by_cfg.setdefault(r.config_idx, []).append(r)

        best_ci_lower = -math.inf
        winner: Optional[int] = None
        # per-candidate summary for later aggregation
        cand_summaries: Dict[int, Dict[str, Any]] = {}

        for ci, frows in by_cfg.items():
            # Drop folds below the minimum trade count.
            useful = [r for r in frows if r.oos_n_trades >= MIN_TRADES_PER_FOLD]
            if not useful:
                continue
            pooled_oos = []
            for r in useful:
                pooled_oos.extend(r.oos_returns)
            ci_low, ci_point, ci_high = bootstrap_sharpe_ci(
                pooled_oos,
                n_bootstrap=500,
                periods_per_year=8760.0,
                seed=self.spec.seed,
            )
            summary = {
                "n_folds_used": len(useful),
                "is_sharpes": [r.is_sharpe for r in useful],
                "oos_sharpes": [r.oos_sharpe for r in useful],
                "oos_n_trades_total": sum(r.oos_n_trades for r in useful),
                "oos_sharpe_ci_lower": ci_low,
                "oos_sharpe_point": ci_point,
                "oos_sharpe_ci_upper": ci_high,
            }
            cand_summaries[ci] = summary
            if ci_low > best_ci_lower:
                best_ci_lower = ci_low
                winner = ci

        if winner is None:
            return None, {}
        # Compute PBO + DSR using all candidates (not just the winner).
        pbo_val = self._compute_pbo_matrix(by_cfg)
        winner_summary = cand_summaries[winner]
        pooled_oos_winner = []
        for r in by_cfg[winner]:
            if r.oos_n_trades >= MIN_TRADES_PER_FOLD:
                pooled_oos_winner.extend(r.oos_returns)
        n_eff = effective_sample_size(pooled_oos_winner, max_lag=24)
        ds = deflated_sharpe(
            in_sample_sharpe=winner_summary["oos_sharpe_point"],
            n_params=max(1, len([s for s in cand_summaries if cand_summaries[s]["n_folds_used"] >= 2])),
            n_obs=int(n_eff),
        )
        winner_summary.update({
            "pbo": pbo_val,
            "deflated_oos_sharpe": ds,
            "n_eff": round(float(n_eff), 2),
        })
        return winner, winner_summary

    def _compute_pbo_matrix(
        self,
        by_cfg: Dict[int, List[FoldResult]],
    ) -> float:
        """Build the (folds × configs) IS/OOS Sharpe matrices for PBO.

        Returns NaN (reported to UI as "insufficient folds") if fewer
        than 4 folds × 2 candidates survive the MIN_TRADES filter.
        """
        cand_ids = sorted(by_cfg.keys())
        all_folds: set[int] = set()
        for rs in by_cfg.values():
            for r in rs:
                if r.oos_n_trades >= MIN_TRADES_PER_FOLD:
                    all_folds.add(r.fold)
        fold_ids = sorted(all_folds)
        if len(fold_ids) < 2 or len(cand_ids) < 2:
            return float("nan")

        is_mat = np.zeros((len(fold_ids), len(cand_ids)))
        oos_mat = np.zeros_like(is_mat)
        for i, fi in enumerate(fold_ids):
            for j, ci in enumerate(cand_ids):
                match = [r for r in by_cfg[ci] if r.fold == fi]
                if match and match[0].oos_n_trades >= MIN_TRADES_PER_FOLD:
                    is_mat[i, j] = match[0].is_sharpe
                    oos_mat[i, j] = match[0].oos_sharpe
                else:
                    is_mat[i, j] = np.nan
                    oos_mat[i, j] = np.nan
        return float(pbo_bootstrap(is_mat, oos_mat, n_bootstrap=200, seed=self.spec.seed))

    def _write_artifact(self, report: TuneReport, job_id: str) -> Path:
        """Persist the report JSON to ``results/autotune/``.

        Directory mkdir is idempotent; file name encodes the job_id so
        re-runs don't clobber each other.
        """
        artifact_dir = Path("results/autotune")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{job_id}.json"
        # Serialize — TuneSpec / TuneReport are dataclasses.
        payload = {
            "spec": asdict(report.spec),
            "verdict": report.verdict,
            "reasons": report.reasons,
            "current_config_hash": report.current_config_hash,
            "proposed_config": report.proposed_config,
            "proposed_config_hash": report.proposed_config_hash,
            "diff": report.diff,
            "metrics": report.metrics,
            "per_fold": report.per_fold,
            "ran_at": report.ran_at,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def _reject_report(self, base_cfg: PulseConfig, reasons: List[str]) -> TuneReport:
        return TuneReport(
            spec=self.spec,
            verdict="REJECT",
            reasons=reasons,
            current_config_hash=base_cfg.content_hash,
            proposed_config={},
            proposed_config_hash=base_cfg.content_hash,
            diff=[],
            metrics={},
            per_fold=[],
        )

    def _emit_progress(self, **event: Any) -> None:
        if self.progress_cb is None:
            return
        try:
            self.progress_cb(event)
        except Exception as e:  # progress must never crash the tune
            logger.debug(f"[AutoTune] progress_cb raised: {e}")
