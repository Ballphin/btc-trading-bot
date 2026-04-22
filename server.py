"""FastAPI backend for TradingAgents dashboard.

Provides:
- POST /api/analyze        — Start a new analysis (returns job_id)
- GET  /api/stream/{id}    — SSE stream of analysis progress
- GET  /api/history         — List tickers with past analyses
- GET  /api/history/{ticker}           — List analysis dates for a ticker
- GET  /api/history/{ticker}/{date}    — Full log JSON for one analysis
- GET  /api/price/{ticker}             — OHLCV + SMA data via yfinance
- GET  /api/health                     — Health check
- POST /api/backtest       — Start a backtest
- GET  /api/backtest/stream/{id} — SSE stream of backtest progress
- GET  /api/backtest/{id}  — Get backtest results
- GET  /api/backtest/results — List historical backtests
- POST /api/shadow/record  — Record a shadow (paper-trade) decision
- GET  /api/shadow/decisions/{ticker}  — Retrieve shadow decisions
- GET  /api/shadow/score/{ticker}      — Score shadow decisions vs outcomes
"""

import asyncio
import contextvars
import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import pandas as pd
import uvicorn
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

from tradingagents.graph.stream_progress import detect_step_from_chunk as _detect_step_from_chunk

# ── Lifespan context manager (replaces deprecated @app.on_event("startup")) ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: startup initialization and graceful shutdown."""
    background_tasks: list[asyncio.Task] = []

    # Startup: Gist sync pull (from original _start_pulse_scheduler)
    try:
        from tradingagents.pulse import gist_sync
        if gist_sync.is_enabled():
            result = gist_sync.pull_all(PULSE_DIR)
            logger.info(f"[GistSync] Startup pull: {result}")
            shadow_result = gist_sync.pull_shadow_all(SHADOW_DIR)
            logger.info(f"[GistSync] Shadow startup pull: {shadow_result}")
        if gist_sync.is_history_enabled():
            history_result = gist_sync.pull_history_all(EVAL_RESULTS_DIR)
            logger.info(f"[GistSync] History startup pull: {history_result}")
    except Exception as e:
        logger.warning(f"[GistSync] Startup pull failed: {e}")

    # Startup: Pulse scheduler and related loops (from original _start_pulse_scheduler)
    if _pulse_state.get("enabled") and not _pulse_state.get("task"):
        _pulse_state["task"] = asyncio.create_task(_pulse_scheduler())
        logger.info("[Pulse] Auto-restored scheduler after restart")
        if not _pulse_state.get("tsmom_task"):
            _pulse_state["tsmom_task"] = asyncio.create_task(_tsmom_refresh_loop())
            logger.info("[TSMOM] Refresh loop started (1h cadence)")
        if not _pulse_state.get("verifier_task"):
            _pulse_state["verifier_task"] = asyncio.create_task(_ensemble_verifier_loop())
            logger.info("[Verifier] Ensemble verifier loop started (5min cadence)")

    # Startup: Scoring loop (from original _start_pulse_scoring)
    async def _scoring_loop():
        while True:
            try:
                await _score_pending_pulses()
            except Exception as e:
                logger.error(f"[Pulse Scoring] Error: {e}")
            await asyncio.sleep(900)  # every 15 min

    scoring_task = asyncio.create_task(_scoring_loop())
    background_tasks.append(scoring_task)
    logger.info("[Pulse Scoring] Scoring loop started (15min cadence)")

    # Startup: Background eviction task and auto-analysis scheduler (from _start_background_tasks)
    eviction_task = asyncio.create_task(_evict_old_backtest_jobs())
    background_tasks.append(eviction_task)
    if _scheduler_state.get("enabled") and not _scheduler_state.get("task"):
        _scheduler_state["task"] = asyncio.create_task(_auto_analysis_scheduler())
        logger.info("[Scheduler] Auto-restored after server restart (was previously enabled)")

    yield

    # Shutdown: Cancel all background tasks gracefully
    logger.info("[Lifespan] Shutdown initiated, cancelling background tasks...")

    # Cancel pulse state tasks
    for key in ["task", "tsmom_task", "verifier_task"]:
        task = _pulse_state.get(key)
        if task and not task.done():
            task.cancel()
            background_tasks.append(task)

    # Cancel scheduler state task if present
    scheduler_task = _scheduler_state.get("task")
    if scheduler_task and not scheduler_task.done():
        scheduler_task.cancel()
        background_tasks.append(scheduler_task)

    # Cancel scoring and any other background tasks
    for task in background_tasks:
        if not task.done():
            task.cancel()

    # Wait for all tasks to complete cancellation
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)

    logger.info("[Lifespan] Background tasks cancelled, shutdown complete")


# ── App setup ─────────────────────────────────────────────────────────
app = FastAPI(title="TradingAgents API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EVAL_RESULTS_DIR = Path("eval_results")

# ── In-memory job store ───────────────────────────────────────────────
jobs: Dict[str, Dict[str, Any]] = {}
backtest_jobs: Dict[str, Dict[str, Any]] = {}

# ── Scheduler state ───────────────────────────────────────────────────
# The 4H auto-analysis scheduler. Manually activated from the frontend.
_SCHEDULER_TICKERS = ["BTC-USD"]  # tickers to run automatically
_SCHEDULER_INTERVAL_HOURS = 4     # run every 4 hours, synced to UTC boundary
# Data-ready delay after candle close. 600s (10 min) gives crypto exchanges
# enough time to publish the closed 4H candle even during volatile regimes
# where feeds sometimes lag 30–90s (plan Part 2 action #6). Override via env
# `ANALYSIS_DATA_DELAY_SECONDS` if you need to tune.
_DATA_DELAY_SECONDS = int(os.environ.get("ANALYSIS_DATA_DELAY_SECONDS", "600"))

_SCHEDULER_STATE_FILE = EVAL_RESULTS_DIR / ".scheduler_state.json"
_MODEL_CONFIG_FILE = EVAL_RESULTS_DIR / ".model_config.json"

# ── User display timezone ───────────────────────────────────────────
# All human-facing timestamps (Telegram alerts, analysis detail views,
# scheduler "next run") are rendered in this timezone regardless of the
# server host timezone. Hosts like Render run in UTC, which produced
# mismatched display strings ("EDT" vs "UTC") depending on where each
# alert was formatted and created confusion when the UI was shared with
# users in other timezones. Override with env `USER_DISPLAY_TZ` if the
# owner of the deployment lives elsewhere.
#
# The zone string is passed to ZoneInfo so DST is handled automatically
# (EST in winter, EDT in summer). America/New_York is the canonical
# IANA identifier for US Eastern.
try:
    from zoneinfo import ZoneInfo  # py3.9+
    _USER_DISPLAY_TZ = ZoneInfo(os.environ.get("USER_DISPLAY_TZ", "America/New_York"))
except Exception:  # pragma: no cover — zoneinfo missing on pre-3.9
    import pytz as _pytz_fallback
    _USER_DISPLAY_TZ = _pytz_fallback.timezone(
        os.environ.get("USER_DISPLAY_TZ", "America/New_York")
    )

# Boundary-claim lock: the in-process 4H scheduler AND the /api/health self-tick
# both write to _scheduler_state["last_run"]. Without a lock they can race and
# launch duplicate analyses for the same boundary (plan Part 2 BLOCKER #1).
# Acquire, check `last_run == expected_label`, persist the claim, then release —
# only then launch analysis threads.
_BOUNDARY_CLAIM_LOCK = threading.Lock()

# Per-ticker in-flight analysis lock (plan Part 2 action #7): refuse to
# launch a new auto/self-tick analysis for a ticker if one is already
# running. Prevents thread pile-up when the LLM is slow.
_ANALYSIS_IN_FLIGHT: set = set()
_ANALYSIS_IN_FLIGHT_LOCK = threading.Lock()


def _try_claim_ticker(ticker: str) -> bool:
    """Atomically claim a ticker for an auto/self-tick analysis. Returns
    True if the caller acquired it, False if another thread owns it."""
    with _ANALYSIS_IN_FLIGHT_LOCK:
        if ticker in _ANALYSIS_IN_FLIGHT:
            return False
        _ANALYSIS_IN_FLIGHT.add(ticker)
        return True


def _release_ticker(ticker: str) -> None:
    with _ANALYSIS_IN_FLIGHT_LOCK:
        _ANALYSIS_IN_FLIGHT.discard(ticker)


def _load_model_config_into_default() -> None:
    """Merge persisted model/ensemble settings into DEFAULT_CONFIG (survives restarts).
    If the lock is enforced and persisted provider is not allowed, force-coerce
    to DeepSeek defaults (plan Part 3)."""
    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        if _MODEL_CONFIG_FILE.exists():
            saved = json.loads(_MODEL_CONFIG_FILE.read_text())
            keys = (
                "llm_provider",
                "deep_think_llm",
                "quick_think_llm",
                "enable_ensemble",
                "ensemble_runs",
            )
            for k in keys:
                if k in saved and saved[k] is not None:
                    DEFAULT_CONFIG[k] = saved[k]

        # Coerce any non-allowed provider back to DeepSeek defaults when locked
        lock_on = os.environ.get("DEEPSEEK_LOCK_OVERRIDE", "").strip() not in ("1", "true", "TRUE")
        if lock_on and DEFAULT_CONFIG.get("llm_provider") not in {"deepseek"}:
            logger.warning(
                "[ModelLock] Persisted provider '%s' not allowed; coercing to deepseek",
                DEFAULT_CONFIG.get("llm_provider"),
            )
            DEFAULT_CONFIG["llm_provider"] = "deepseek"
            DEFAULT_CONFIG["deep_think_llm"] = "deepseek-chat"
            DEFAULT_CONFIG["quick_think_llm"] = "deepseek-chat"
            DEFAULT_CONFIG["enable_ensemble"] = False

        logger.info(
            "Loaded persisted model config: provider=%s ensemble=%s",
            DEFAULT_CONFIG.get("llm_provider"),
            DEFAULT_CONFIG.get("enable_ensemble"),
        )
    except Exception as e:
        logger.warning("Failed to load persisted model config: %s", e)


def _save_model_config() -> None:
    """Write user-facing model fields to disk AND mirror to primary gist so
    the model choice survives Render free-tier restarts."""
    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        _MODEL_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "llm_provider": DEFAULT_CONFIG.get("llm_provider"),
            "deep_think_llm": DEFAULT_CONFIG.get("deep_think_llm"),
            "quick_think_llm": DEFAULT_CONFIG.get("quick_think_llm"),
            "enable_ensemble": DEFAULT_CONFIG.get("enable_ensemble"),
            "ensemble_runs": DEFAULT_CONFIG.get("ensemble_runs"),
        }
        _MODEL_CONFIG_FILE.write_text(json.dumps(payload, indent=2))
        # Fire-and-forget gist mirror
        try:
            from tradingagents.pulse import gist_sync
            if gist_sync.is_enabled():
                threading.Thread(
                    target=gist_sync.push_state,
                    args=("model_config.json", payload),
                    daemon=True,
                ).start()
        except Exception:
            pass
    except Exception as e:
        logger.warning("Failed to persist model config: %s", e)


def _load_scheduler_state() -> Dict[str, Any]:
    """Load persisted scheduler state from disk (survives server reloads)."""
    defaults: Dict[str, Any] = {
        "enabled": False,
        "task": None,
        "last_run": None,
        "last_status": None,
        "next_run": None,
        "tickers": _SCHEDULER_TICKERS,
    }
    try:
        if _SCHEDULER_STATE_FILE.exists():
            saved = json.loads(_SCHEDULER_STATE_FILE.read_text())
            defaults["enabled"] = saved.get("enabled", False)
            defaults["last_run"] = saved.get("last_run")
            defaults["last_status"] = saved.get("last_status")
            defaults["tickers"] = saved.get("tickers", _SCHEDULER_TICKERS)
    except Exception:
        pass
    return defaults

def _save_scheduler_state():
    """Persist scheduler state to disk AND mirror to primary gist so the
    `enabled` flag survives Render free-tier filesystem wipes (plan Part 2)."""
    payload = {
        "enabled": _scheduler_state.get("enabled", False),
        "last_run": _scheduler_state.get("last_run"),
        "last_status": _scheduler_state.get("last_status"),
        "tickers": _scheduler_state.get("tickers", _SCHEDULER_TICKERS),
    }
    try:
        _SCHEDULER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SCHEDULER_STATE_FILE.write_text(json.dumps(payload, indent=2))
    except Exception as e:
        logger.warning(f"Failed to persist scheduler state: {e}")
    # Fire-and-forget gist mirror — ignore failures
    try:
        from tradingagents.pulse import gist_sync
        if gist_sync.is_enabled():
            threading.Thread(
                target=gist_sync.push_state,
                args=("scheduler_state.json", payload),
                daemon=True,
            ).start()
    except Exception:
        pass

def _hydrate_state_from_gist() -> None:
    """On Render free tier, `.scheduler_state.json` and `.model_config.json`
    live under EVAL_RESULTS_DIR which is wiped on restart. Try pulling them
    from the primary gist (see gist_sync.pull_state) before we load locally,
    so the 4H scheduler "enabled" flag and model choice survive restarts
    without user intervention. Safe no-op when gist isn't configured.
    """
    try:
        from tradingagents.pulse import gist_sync
        if not gist_sync.is_enabled():
            return
        for gist_filename, local_path in (
            ("scheduler_state.json", _SCHEDULER_STATE_FILE),
            ("model_config.json", _MODEL_CONFIG_FILE),
        ):
            if local_path.exists():
                continue  # local already present (redeploy with cached disk), keep it
            remote = gist_sync.pull_state(gist_filename)
            if remote is None:
                continue
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_text(json.dumps(remote, indent=2))
                logger.info(f"[GistSync] Hydrated {local_path.name} from gist")
            except Exception as e:
                logger.warning(f"[GistSync] Write hydrated {local_path.name} failed: {e}")
    except Exception as e:
        logger.warning(f"[GistSync] State hydration failed: {e}")


# Hydrate from gist FIRST, then load from (possibly just-written) disk files.
_hydrate_state_from_gist()
_scheduler_state: Dict[str, Any] = _load_scheduler_state()

# Restore LLM / ensemble choices after restarts (must run after DEFAULT_CONFIG exists)
_load_model_config_into_default()


def _next_4h_utc_boundary() -> datetime:
    """Return the next 4-hour UTC boundary from now (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)."""
    import pytz
    now_utc = datetime.now(pytz.utc).replace(second=0, microsecond=0)
    # current hour snapped down to the 4H block
    current_block = (now_utc.hour // _SCHEDULER_INTERVAL_HOURS) * _SCHEDULER_INTERVAL_HOURS
    boundary = now_utc.replace(hour=current_block, minute=0, second=0, microsecond=0)
    if boundary <= now_utc:
        boundary += timedelta(hours=_SCHEDULER_INTERVAL_HOURS)
    return boundary


async def _auto_analysis_scheduler():
    """Run BTC-USD analysis every 4 hours synced to UTC candle closes.
    
    Physical sleep targets: (next 4H UTC boundary + DATA_DELAY_SECONDS).
    The logical candle time passed to the analysis is exactly the UTC boundary
    (e.g. '2026-04-08T16') regardless of when the physical process runs.
    """
    import pytz
    print("[Scheduler] 4H auto-analysis scheduler started", flush=True)
    while _scheduler_state["enabled"]:
        try:
            boundary_utc = _next_4h_utc_boundary()
            physical_trigger = boundary_utc + timedelta(seconds=_DATA_DELAY_SECONDS)
            now_utc = datetime.now(pytz.utc)
            sleep_secs = (physical_trigger - now_utc).total_seconds()
            if sleep_secs > 0:
                logical_label = boundary_utc.strftime("%Y-%m-%dT%H")
                logical_date = boundary_utc.strftime("%Y-%m-%d")
                _scheduler_state["next_run"] = boundary_utc.isoformat()
                print(
                    f"[Scheduler] Next run: {logical_label} UTC "
                    f"(sleeping {sleep_secs:.0f}s / {sleep_secs/3600:.1f}h until data ready)",
                    flush=True,
                )
                await asyncio.sleep(sleep_secs)

            if not _scheduler_state["enabled"]:
                break

            logical_label = boundary_utc.strftime("%Y-%m-%dT%H")
            logical_date = boundary_utc.strftime("%Y-%m-%d")

            # Atomic boundary claim: skip this boundary if self-tick already ran it
            with _BOUNDARY_CLAIM_LOCK:
                if _scheduler_state.get("last_run") == logical_label:
                    print(f"[Scheduler] Boundary {logical_label} already claimed (self-tick ran it); skipping", flush=True)
                    continue
                _scheduler_state["last_run"] = logical_label
                _scheduler_state["last_status"] = "ok (claimed by in-process)"
                _save_scheduler_state()

            from tradingagents.default_config import DEFAULT_CONFIG
            from tradingagents.graph.ensemble_orchestrator import should_use_ensemble
            current_provider = DEFAULT_CONFIG.get("llm_provider", "deepseek")
            ensemble_active = should_use_ensemble(DEFAULT_CONFIG, current_provider)
            
            if ensemble_active:
                _n = DEFAULT_CONFIG.get("ensemble_runs", 1)
                print(
                    f"[Scheduler] Running ENSEMBLE auto-analysis ({_n}x parallel) for {_SCHEDULER_TICKERS} @ {logical_label}",
                    flush=True,
                )
            else:
                print(f"[Scheduler] Running auto-analysis for {_SCHEDULER_TICKERS} @ {logical_label} (provider: {current_provider})", flush=True)
            
            for ticker in _SCHEDULER_TICKERS:
                if not _try_claim_ticker(ticker):
                    print(f"[Scheduler] Skipping {ticker} — analysis already in flight", flush=True)
                    continue
                job_id = str(uuid.uuid4())[:8]
                eq = JobEventQueue()
                jobs[job_id] = {
                    "ticker": ticker,
                    "date": logical_date,
                    "status": "running",
                    "queue": eq,
                    "result": None,
                    "error": None,
                    "scheduled": True,
                    "candle_time": logical_label,
                }
                def _run_and_release(jid=job_id, tkr=ticker, ld=logical_date, ll=logical_label):
                    try:
                        _run_analysis(jid, tkr, ld, True, ll)
                    finally:
                        _release_ticker(tkr)
                thread = threading.Thread(target=_run_and_release, daemon=True)
                thread.start()
                print(f"[Scheduler] Launched job {job_id} for {ticker} @ {logical_label}", flush=True)

            # last_run already set during boundary claim above; just clear errors
            _scheduler_state["last_status"] = "ok"
            _scheduler_state["error_count"] = 0
            _save_scheduler_state()

        except asyncio.CancelledError:
            print("[Scheduler] Task cancelled", flush=True)
            break
        except Exception as e:
            error_count = _scheduler_state.get("error_count", 0) + 1
            _scheduler_state["error_count"] = error_count
            print(f"[Scheduler] Error ({error_count}/3): {e}", flush=True)
            import traceback; traceback.print_exc()
            _scheduler_state["last_status"] = f"error: {e}"
            
            if error_count >= 3:
                _scheduler_state["enabled"] = False
                _save_scheduler_state()
                print("[Scheduler] DISABLED after 3 consecutive errors", flush=True)
                break
            
            await asyncio.sleep(60)

    print("[Scheduler] Stopped", flush=True)


# ── Background eviction of stale completed backtest jobs (HIGH 8) ─────
_BACKTEST_JOB_TTL_HOURS = 1

async def _evict_old_backtest_jobs():
    """Periodically remove completed/failed backtest jobs older than TTL to prevent memory leak."""
    while True:
        await asyncio.sleep(300)  # run every 5 minutes
        cutoff = datetime.now() - timedelta(hours=_BACKTEST_JOB_TTL_HOURS)
        expired = [
            jid for jid, j in list(backtest_jobs.items())
            if j.get("status") in ("completed", "failed")
            and datetime.fromisoformat(j.get("created_at", datetime.now().isoformat())) < cutoff
        ]
        for jid in expired:
            backtest_jobs.pop(jid, None)
        if expired:
            logger.info(f"Evicted {len(expired)} stale backtest job(s)")

# ── Scheduler API Endpoints ───────────────────────────────────────────

@app.get("/api/scheduler/status")
async def scheduler_status():
    """Return current scheduler state including next run time (local + UTC)."""
    import pytz
    try:
        boundary_utc = _next_4h_utc_boundary()
        # Convert to the user's configured display timezone (EST/EDT by default).
        # Previously used `datetime.now().astimezone().tzinfo` which picks the
        # SERVER host TZ — on Render that's UTC, so the "local" ISO sent to the
        # frontend was really UTC and the UI's toLocaleTimeString() silently
        # re-offset it to the viewer's browser TZ. That caused apparent drift
        # when a friend in another timezone opened the same link.
        boundary_local = boundary_utc.astimezone(_USER_DISPLAY_TZ)
        next_run_local = boundary_local.isoformat() if _scheduler_state["enabled"] else None
        next_run_utc = boundary_utc.isoformat() if _scheduler_state["enabled"] else None
    except Exception:
        next_run_local = None
        next_run_utc = None

    return {
        "enabled": _scheduler_state["enabled"],
        "tickers": _scheduler_state["tickers"],
        "interval_hours": _SCHEDULER_INTERVAL_HOURS,
        "data_delay_seconds": _DATA_DELAY_SECONDS,
        "last_run": _scheduler_state["last_run"],
        "last_status": _scheduler_state["last_status"],
        "next_run_utc": next_run_utc,
        "next_run_local": next_run_local,
    }


@app.post("/api/scheduler/toggle")
async def toggle_scheduler():
    """Enable or disable the 4H auto-analysis scheduler. Returns new state."""
    if _scheduler_state["enabled"]:
        # Disable: cancel the running task
        _scheduler_state["enabled"] = False
        task = _scheduler_state.get("task")
        if task and not task.done():
            task.cancel()
        _scheduler_state["task"] = None
        _save_scheduler_state()
        print("[Scheduler] Disabled via API", flush=True)
        return {"enabled": False, "message": "Scheduler disabled"}
    else:
        # Enable: start the background task (singleton guard)
        task = _scheduler_state.get("task")
        if task and not task.done():
            return {"enabled": True, "message": "Already running"}
        _scheduler_state["enabled"] = True
        _scheduler_state["task"] = asyncio.create_task(_auto_analysis_scheduler())
        _save_scheduler_state()
        print("[Scheduler] Enabled via API", flush=True)
        return {"enabled": True, "message": "Scheduler enabled"}


@app.post("/api/scheduler/run-now")
async def scheduler_run_now():
    """Immediately trigger an analysis run outside the normal schedule."""
    import pytz
    logical_label = datetime.now(pytz.utc).strftime("%Y-%m-%dT%H")
    logical_date = datetime.now(pytz.utc).strftime("%Y-%m-%d")

    launched = []
    for ticker in _SCHEDULER_TICKERS:
        job_id = str(uuid.uuid4())[:8]
        eq = JobEventQueue()
        jobs[job_id] = {
            "ticker": ticker,
            "date": logical_date,
            "status": "running",
            "queue": eq,
            "result": None,
            "error": None,
            "scheduled": False,
            "candle_time": logical_label,
        }
        thread = threading.Thread(
            target=_run_analysis,
            args=(job_id, ticker, logical_date, True, logical_label),
            daemon=True,
        )
        thread.start()
        launched.append({"ticker": ticker, "job_id": job_id})

    return {"launched": launched, "candle_time": logical_label}


# ── Model Configuration API Endpoints ──────────────────────────────────

class ModelConfigRequest(BaseModel):
    """Request model for updating model configuration."""
    provider: str
    model: str
    parallel_mode: bool = True


# DeepSeek hard-lock: production deployments only allow these providers.
# Environment escape hatch for local dev: set DEEPSEEK_LOCK_OVERRIDE=1.
_ALLOWED_PROVIDERS = {"deepseek"}
_ALLOWED_MODELS = {"deepseek-chat", "deepseek-coder"}


def _provider_lock_enforced() -> bool:
    return os.environ.get("DEEPSEEK_LOCK_OVERRIDE", "").strip() not in ("1", "true", "TRUE")


@app.get("/api/model/config")
async def get_model_config():
    """Get current model configuration. If legacy persisted state has a
    non-allowed provider, silently coerce to DeepSeek and re-save."""
    from tradingagents.default_config import DEFAULT_CONFIG

    provider = DEFAULT_CONFIG.get("llm_provider", "deepseek")
    model = DEFAULT_CONFIG.get("deep_think_llm", "deepseek-chat")

    # Legacy coercion: any stale non-deepseek state (from pre-lock deployment
    # or from gist hydration) gets flipped back to the locked default.
    if _provider_lock_enforced() and provider not in _ALLOWED_PROVIDERS:
        logger.info(f"[ModelLock] Coercing legacy provider '{provider}' to 'deepseek'")
        DEFAULT_CONFIG["llm_provider"] = "deepseek"
        DEFAULT_CONFIG["deep_think_llm"] = "deepseek-chat"
        DEFAULT_CONFIG["quick_think_llm"] = "deepseek-chat"
        DEFAULT_CONFIG["enable_ensemble"] = False
        _save_model_config()
        provider, model = "deepseek", "deepseek-chat"

    return {
        "provider": provider,
        "model": model,
        "ensemble_enabled": DEFAULT_CONFIG.get("enable_ensemble", False),
        "ensemble_runs": DEFAULT_CONFIG.get("ensemble_runs", 1),
        "ensemble_providers": DEFAULT_CONFIG.get("ensemble_enabled_providers", ["openrouter"]),
        "single_run_providers": DEFAULT_CONFIG.get("ensemble_disabled_providers", ["deepseek"]),
        "fallback_model": DEFAULT_CONFIG.get("openrouter_fallback_model", "anthropic/claude-3.5-sonnet"),
        "provider_locked": _provider_lock_enforced(),
        "allowed_providers": sorted(_ALLOWED_PROVIDERS),
        "allowed_models": sorted(_ALLOWED_MODELS),
    }


@app.post("/api/model/config")
async def set_model_config(req: ModelConfigRequest):
    """Update model configuration; persisted to disk + mirrored to gist.

    Enforces DeepSeek-only in production (plan Part 3 BLOCKER #3). Returns
    HTTP 400 with structured error_code if the caller sends a non-allowed
    provider — frontend shows an actionable message.
    """
    from tradingagents.default_config import DEFAULT_CONFIG

    provider_lc = (req.provider or "").lower()
    model_lc = (req.model or "").lower()

    if _provider_lock_enforced():
        if provider_lc not in _ALLOWED_PROVIDERS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "PROVIDER_LOCKED",
                    "message": "This deployment is locked to DeepSeek. Other providers are disabled.",
                    "allowed_providers": sorted(_ALLOWED_PROVIDERS),
                    "received": req.provider,
                },
            )
        if model_lc not in _ALLOWED_MODELS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "MODEL_NOT_ALLOWED",
                    "message": f"Model '{req.model}' is not in the allow-list for this deployment.",
                    "allowed_models": sorted(_ALLOWED_MODELS),
                    "received": req.model,
                },
            )

    # Update configuration
    DEFAULT_CONFIG["llm_provider"] = provider_lc
    DEFAULT_CONFIG["deep_think_llm"] = req.model
    DEFAULT_CONFIG["quick_think_llm"] = req.model

    # Check if provider supports ensemble
    disabled_providers = DEFAULT_CONFIG.get("ensemble_disabled_providers", ["deepseek"])
    enabled_providers = DEFAULT_CONFIG.get("ensemble_enabled_providers", ["openrouter"])

    if provider_lc in disabled_providers:
        DEFAULT_CONFIG["enable_ensemble"] = False
        ensemble_active = False
        message = f"Ensemble disabled (not supported for {req.provider})"
    elif provider_lc in enabled_providers:
        DEFAULT_CONFIG["enable_ensemble"] = req.parallel_mode
        ensemble_active = req.parallel_mode
        message = f"Ensemble {'enabled' if req.parallel_mode else 'disabled'} for {req.provider}"
    else:
        DEFAULT_CONFIG["enable_ensemble"] = req.parallel_mode
        ensemble_active = req.parallel_mode
        message = "Configuration updated"

    _save_model_config()

    return {
        "status": "ok",
        "provider": req.provider,
        "model": req.model,
        "ensemble_active": ensemble_active,
        "message": message,
    }


@app.get("/api/model/sanity_check")
async def model_sanity_check(ticker: str = "BTC-USD"):
    """Quick verification that the locked DeepSeek provider produces a
    parseable signal. Plan Part 3 BLOCKER #3 — ships with the lock so
    users can confirm reasoning quality immediately after deploy.

    Runs ONE quick-path LLM call (no full graph) and returns the raw
    response. Does NOT write to shadow / pulse / history.
    """
    from tradingagents.default_config import DEFAULT_CONFIG
    provider = DEFAULT_CONFIG.get("llm_provider", "deepseek")
    model = DEFAULT_CONFIG.get("deep_think_llm", "deepseek-chat")
    try:
        # Use the lightweight quant pulse pipeline as a proxy: it's fast,
        # deterministic, and exercises the LLM path without spawning a full
        # multi-agent analysis.
        from tradingagents.agents.quant_pulse_engine import score_pulse
        sample = score_pulse(ticker.upper())
        return {
            "status": "ok",
            "provider": provider,
            "model": model,
            "ticker": ticker.upper(),
            "signal": sample.get("signal"),
            "confidence": sample.get("confidence"),
            "reasoning": (sample.get("reasoning") or "")[:500],
            "breakdown": sample.get("breakdown"),
            "provider_locked": _provider_lock_enforced(),
        }
    except Exception as e:
        return {
            "status": "error",
            "provider": provider,
            "model": model,
            "error": str(e)[:500],
            "provider_locked": _provider_lock_enforced(),
        }


@app.get("/api/model/providers")
async def get_available_providers():
    """Get list of available LLM providers and their models."""
    return {
        "providers": [
            {
                "id": "openrouter",
                "name": "OpenRouter",
                "models": ["qwen/qwen3.6-plus", "anthropic/claude-3.5-sonnet", "openai/gpt-4o"],
                "ensemble_supported": True,
            },
            {
                "id": "deepseek",
                "name": "DeepSeek",
                "models": ["deepseek-chat", "deepseek-coder"],
                "ensemble_supported": False,
            },
            {
                "id": "openai",
                "name": "OpenAI",
                "models": ["gpt-5.2", "gpt-5-mini"],
                "ensemble_supported": False,
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "models": ["claude-3-5-sonnet-20241022"],
                "ensemble_supported": False,
            },
        ]
    }


class AnalyzeRequest(BaseModel):
    ticker: str
    date: Optional[str] = None
    force_refresh: Optional[bool] = None  # None = default to True, False = explicitly disabled


class BacktestRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    mode: str = "replay"  # "replay", "simulation", or "hybrid"
    config: Optional[Dict[str, Any]] = None
    # Crypto-specific fields
    leverage: float = 1.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    use_funding: bool = True
    position_sizing: str = "fixed"


# ── SSE event queue helper ────────────────────────────────────────────

class JobEventQueue:
    """Thread-safe event queue for SSE streaming with replay support.

    Every event is appended to ``_history``.  Each call to ``set_loop()``
    creates a **fresh** asyncio queue and replays the full history into it,
    so a reconnecting SSE client always receives every event from the start.
    """

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._history: list = []
        self._lock = threading.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        with self._lock:
            self._loop = loop
            self._queue = asyncio.Queue()
            for event in self._history:
                self._unsafe_put(event)

    def _unsafe_put(self, event: dict):
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def put(self, event: dict):
        with self._lock:
            skip_history = event.get("event") == "heartbeat"
            if not skip_history:
                self._history.append(event)
            if self._loop and not self._loop.is_closed():
                self._unsafe_put(event)

    async def get(self) -> dict:
        return await self._queue.get()


# ── Analysis pipeline steps ───────────────────────────────────────────

PIPELINE_STEPS = [
    {"key": "market", "label": "Market Analyst", "step": 1},
    {"key": "social", "label": "Social Media Analyst", "step": 2},
    {"key": "news", "label": "News Analyst", "step": 3},
    {"key": "fundamentals", "label": "Fundamentals Analyst", "step": 4},
    {"key": "bull_bear", "label": "Bull vs Bear Debate", "step": 5},
    {"key": "research_manager", "label": "Research Manager", "step": 6},
    {"key": "trader", "label": "Trader", "step": 7},
    {"key": "risk_debate", "label": "Risk Debate", "step": 8},
    {"key": "portfolio_manager", "label": "Portfolio Manager", "step": 9},
]
TOTAL_STEPS = len(PIPELINE_STEPS)


# ── Background auto-scoring ────────────────────────────────────────────
_scoring_lock = threading.Lock()


def _trigger_background_scoring(ticker: str):
    """Score pending shadow decisions for ALL tickers (not just the analyzed one).

    Walks all eval_results/shadow/*/ directories and scores each sequentially.
    Auto-calibrates when enough scored decisions accumulate.
    """
    def _score_all_tickers():
        with _scoring_lock:
            from tradingagents.backtesting.scorecard import (
                score_pending_decisions, run_calibration_study, count_scored_decisions
            )
            shadow_root = EVAL_RESULTS_DIR / "shadow"
            if not shadow_root.exists():
                return

            # Collect all tickers with a decisions.jsonl file
            tickers_to_score = []
            for ticker_dir in shadow_root.iterdir():
                if ticker_dir.is_dir() and (ticker_dir / "decisions.jsonl").exists():
                    tickers_to_score.append(ticker_dir.name)

            # Ensure the triggering ticker is first (fastest feedback)
            if ticker in tickers_to_score:
                tickers_to_score.remove(ticker)
                tickers_to_score.insert(0, ticker)

            for t in tickers_to_score:
                try:
                    result = score_pending_decisions(t, str(EVAL_RESULTS_DIR))
                    newly_scored = result.get("scored", 0)
                    total_scored = result.get("total_scored", 0)
                    if newly_scored > 0:
                        print(f"[AutoScore] {t}: scored {newly_scored} new decisions (total: {total_scored})")

                    # Auto-calibrate when enough scored decisions exist
                    if total_scored >= 10:
                        cal_path = EVAL_RESULTS_DIR / "shadow" / t / "calibration.json"
                        should_calibrate = False
                        if not cal_path.exists():
                            should_calibrate = True
                        else:
                            try:
                                cal = json.loads(cal_path.read_text())
                                last_n = cal.get("n_decisions_total", cal.get("n_decisions", 0))
                                if total_scored >= last_n + 5:
                                    should_calibrate = True
                            except Exception:
                                should_calibrate = True

                        if should_calibrate:
                            cal_result = run_calibration_study(t, results_dir=str(EVAL_RESULTS_DIR))
                            if "error" not in cal_result:
                                print(f"[AutoScore] {t}: calibration updated -> correction={cal_result.get('correction')}")
                except Exception as e:
                    print(f"[AutoScore] {t}: scoring failed (non-fatal): {e}")

    thread = threading.Thread(target=_score_all_tickers, daemon=True)
    thread.start()


def _run_analysis(job_id: str, ticker: str, trade_date: str, force_refresh: bool = False, run_timestamp: str = None):
    """Run the TradingAgents analysis in a background thread.
    
    Args:
        job_id: Unique job identifier
        ticker: Ticker symbol (e.g. 'BTC-USD')
        trade_date: Analysis date (YYYY-MM-DD)
        force_refresh: Clear cache before running
        run_timestamp: Optional intraday candle label (e.g. '2026-04-08T16').
                       When set, saves as full_states_log_2026-04-08T16.json.
    """
    eq: JobEventQueue = jobs[job_id]["queue"]
    
    # Immediately emit a starting event
    eq.put({"event": "agent_start", "agent": "Starting Analysis", "step": 0, "total": 9})
    
    import sys
    print(f"[Analysis {job_id}] Starting analysis for {ticker} on {trade_date} (force_refresh={force_refresh})", flush=True)
    sys.stdout.flush()

    # Cache-bust: clear data_cache subdirs for fresh data (HIGH 6)
    if force_refresh:
        import shutil
        cache_base = Path("tradingagents/dataflows/data_cache")
        if cache_base.exists():
            cleared = 0
            for subdir in cache_base.iterdir():
                if subdir.is_dir():
                    for cache_file in subdir.glob("*.json"):
                        cache_file.unlink(missing_ok=True)
                        cleared += 1
            print(f"[Analysis {job_id}] Force refresh: cleared {cleared} cache files", flush=True)
            eq.put({"event": "agent_start", "agent": f"Cache cleared ({cleared} files)", "step": 0, "total": 9})

    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.graph.ensemble_orchestrator import (
            EnsembleAnalysisOrchestrator, should_use_ensemble
        )

        config = DEFAULT_CONFIG.copy()
        
        # Check if we should use ensemble mode
        current_provider = config.get("llm_provider", "deepseek")
        use_ensemble = should_use_ensemble(config, current_provider)
        
        if use_ensemble:
            # HIGH FIX: Use ensemble orchestrator for OpenRouter
            n_runs = config.get("ensemble_runs", 1)
            print(f"[Analysis {job_id}] Ensemble mode enabled for {current_provider} ({n_runs} members)")
            
            orchestrator = EnsembleAnalysisOrchestrator(
                config=config,
                provider=current_provider,
                model=config.get("deep_think_llm", "qwen/qwen3.6-plus"),
                parallel_runs=n_runs,
            )
            
            # Pass event queue so orchestrator can emit member-level progress
            orchestrator._event_queue = eq
            
            # Background heartbeat thread: emits every 10s so the frontend
            # knows the analysis is alive even when LLM calls take minutes
            heartbeat_stop = threading.Event()
            ensemble_start = time.time()
            def _heartbeat_emitter():
                while not heartbeat_stop.is_set():
                    heartbeat_stop.wait(10)
                    if heartbeat_stop.is_set():
                        break
                    elapsed = int(time.time() - ensemble_start)
                    eq.put({"event": "heartbeat", "elapsed": elapsed})
            heartbeat_thread = threading.Thread(target=_heartbeat_emitter, daemon=True)
            heartbeat_thread.start()
            
            # BLOCKER FIX: Run ensemble in thread-safe async manner
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                consensus_result = loop.run_until_complete(orchestrator.analyze(ticker, trade_date))
            finally:
                loop.close()
                heartbeat_stop.set()

            if consensus_result.ensemble_metadata.get("error") == "no_valid_ensemble_results":
                # Plan Part 4.1 — surface the real underlying error
                member_errors = consensus_result.ensemble_metadata.get("member_errors") or []
                first = member_errors[0] if member_errors else "unknown provider error"
                runs = consensus_result.ensemble_metadata.get("runs", 0)
                message = (
                    f"Analysis failed: {first}. "
                    f"(All {runs} ensemble run(s) returned unparseable output. "
                    f"Switch to DeepSeek single-run mode to avoid ensemble parsing issues.)"
                )
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = message
                jobs[job_id]["member_errors"] = member_errors
                eq.put({"event": "error", "message": message})
                return
            
            # Convert ConsensusResult to result dict
            result = _ensemble_result_to_dict(consensus_result, ticker, run_timestamp or trade_date)
            
            # Run ConfidenceScorer on ensemble consensus so Telegram, frontend,
            # and history all receive the SAME calibrated confidence & R:R values
            try:
                from tradingagents.backtesting.regime import detect_regime_context
                from tradingagents.graph.confidence import ConfidenceScorer
                from tradingagents.dataflows.asset_detection import is_crypto as _is_crypto_asset

                regime_ctx = detect_regime_context(ticker, trade_date)
                _ens_sl = result.get("stop_loss_price")
                _ens_tp = result.get("take_profit_price")
                _ens_hold = result.get("max_hold_days")

                # Crypto-specific guards (same as single-run path)
                if _is_crypto_asset(ticker):
                    if _ens_hold and _ens_hold > 7:
                        _ens_hold = 7
                        result["max_hold_days"] = _ens_hold
                    if (result.get("decision") in ('SHORT', 'SELL')
                            and regime_ctx.get('current_price')
                            and _ens_sl):
                        max_crypto_stop = regime_ctx['current_price'] * 1.12
                        if _ens_sl > max_crypto_stop:
                            _ens_sl = max_crypto_stop
                            result["stop_loss_price"] = _ens_sl

                scored = ConfidenceScorer().score(
                    llm_confidence=result.get("confidence", 0.50),
                    ticker=ticker,
                    signal=result.get("decision", "HOLD"),
                    knowledge_store=None,
                    regime_ctx=regime_ctx,
                    stop_loss=_ens_sl,
                    take_profit=_ens_tp,
                    max_hold_days=int(_ens_hold) if _ens_hold else 7,
                    reasoning=result.get("reasoning", ""),
                )
                result["confidence"] = scored.get("confidence", result.get("confidence"))
                result["position_size_pct"] = scored.get("position_size_pct")
                result["conviction_label"] = scored.get("conviction_label", result.get("conviction_label"))
                result["r_ratio"] = scored.get("r_ratio") or result.get("r_ratio")
                result["r_ratio_warning"] = scored.get("r_ratio_warning", False)
                result["gated"] = scored.get("gated", False)
                result["hold_period_scalar"] = scored.get("hold_period_scalar")
                result["hedge_penalty_applied"] = scored.get("hedge_penalty_applied")
                print(f"[Analysis {job_id}] Ensemble scored: conf={result['confidence']:.3f} "
                      f"size={result.get('position_size_pct', 0):.1%} R={result.get('r_ratio')} "
                      f"gated={result.get('gated')}")
            except Exception as _score_err:
                print(f"[Analysis {job_id}] Ensemble confidence scorer error (non-fatal): {_score_err}")
                import traceback; traceback.print_exc()

            # Fallback R:R from raw prices if scorer didn't provide one
            if result.get("r_ratio") is None:
                try:
                    _fb_sl = result.get("stop_loss_price")
                    _fb_tp = result.get("take_profit_price")
                    _fb_entry = consensus_result.ensemble_metadata.get("entry_price_snapshot")
                    if _fb_sl and _fb_tp and _fb_entry and _fb_entry > 0:
                        _risk = abs(_fb_entry - _fb_sl)
                        _reward = abs(_fb_tp - _fb_entry)
                        if _risk > 0:
                            result["r_ratio"] = round(_reward / _risk, 3)
                            result["r_ratio_warning"] = result["r_ratio"] < 1.0
                except Exception:
                    pass

            # Persist scored fields into the log file so history page reads them
            try:
                file_key = run_timestamp or trade_date
                log_path = EVAL_RESULTS_DIR / ticker / "TradingAgentsStrategy_logs" / f"full_states_log_{file_key}.json"
                if log_path.exists():
                    log_data = json.loads(log_path.read_text())
                    date_key = next(iter(log_data), file_key)
                    entry = log_data.get(date_key, {})
                    entry["decision"] = result.get("decision")
                    entry["signal"] = result.get("decision")
                    entry["confidence"] = result.get("confidence")
                    entry["stop_loss_price"] = result.get("stop_loss_price")
                    entry["take_profit_price"] = result.get("take_profit_price")
                    entry["max_hold_days"] = result.get("max_hold_days")
                    entry["reasoning"] = result.get("reasoning")
                    entry["conviction_label"] = result.get("conviction_label")
                    entry["position_size_pct"] = result.get("position_size_pct")
                    entry["r_ratio"] = result.get("r_ratio")
                    entry["r_ratio_warning"] = result.get("r_ratio_warning")
                    entry["gated"] = result.get("gated")
                    entry["hold_period_scalar"] = result.get("hold_period_scalar")
                    entry["hedge_penalty_applied"] = result.get("hedge_penalty_applied")
                    log_data[date_key] = entry
                    log_path.write_text(json.dumps(log_data, indent=4))
            except Exception as _log_err:
                print(f"[Analysis {job_id}] Failed to persist ensemble fields to log (non-fatal): {_log_err}")

            print(f"[Analysis {job_id}] Ensemble complete: {consensus_result.signal} "
                  f"(conf={result.get('confidence', 0):.2f}, "
                  f"agreement={consensus_result.divergence_metrics.get('signal_agreement', 0):.0%})")
            
            jobs[job_id]["result"] = result
            jobs[job_id]["status"] = "done"
            
            # Dispatch Telegram
            try:
                _send_telegram_alert(result)
            except Exception as e:
                print(f"[Analysis {job_id}] Telegram alert warning: {e}")

            # Auto-record shadow decision for ensemble path
            try:
                _ens_shadow_dir = EVAL_RESULTS_DIR / "shadow" / ticker
                _ens_shadow_dir.mkdir(parents=True, exist_ok=True)
                _ens_shadow_price = None
                try:
                    _ens_shadow_price = _get_price_on_date(ticker, trade_date)
                except Exception:
                    pass
                _ens_shadow_regime = "unknown"
                try:
                    _ens_shadow_regime = regime_ctx.get("regime", "unknown")
                except Exception:
                    pass
                _ens_shadow_entry = {
                    "ticker": ticker,
                    "date": trade_date,
                    "signal": result.get("decision", "HOLD"),
                    "price": _ens_shadow_price,
                    "confidence": result.get("confidence"),
                    "stop_loss": result.get("stop_loss_price"),
                    "take_profit": result.get("take_profit_price"),
                    "max_hold_days": result.get("max_hold_days"),
                    "position_size_pct": result.get("position_size_pct"),
                    "reasoning": result.get("reasoning"),
                    "regime": _ens_shadow_regime,
                    "source": "ensemble_analysis",
                    "recorded_at": datetime.now().isoformat(),
                    "scored": False,
                }
                with open(_ens_shadow_dir / "decisions.jsonl", "a") as _sf:
                    _sf.write(json.dumps(_ens_shadow_entry, default=str) + "\n")
                _push_shadow_async(ticker)
                _push_history_async(ticker)
            except Exception as _shadow_err:
                print(f"[Analysis {job_id}] Ensemble shadow record failed (non-fatal): {_shadow_err}")

            # Background auto-scoring for all tickers with pending decisions
            _trigger_background_scoring(ticker)
            
            # Emit decision event for the signal badge
            eq.put({"event": "decision", "signal": result.get("decision", "HOLD")})
            
            # Signal completion with full result so frontend can populate UI
            eq.put({"event": "done", "result": result})
            return
        
        # Standard single-run mode (existing logic)
        # Only set defaults if not already configured (respect user's model selection)
        if not config.get("llm_provider") or config["llm_provider"] == "openai":
            config["llm_provider"] = "deepseek"
            config["deep_think_llm"] = "deepseek-reasoner"
            config["quick_think_llm"] = "deepseek-chat"
        config["max_debate_rounds"] = 1
        config["data_vendors"] = {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
            "prediction_market_data": "kalshi",
        }
        # Preserve crypto vendor config for crypto assets
        config.setdefault("crypto_vendors", DEFAULT_CONFIG.get("crypto_vendors", {}))

        ta = TradingAgentsGraph(debug=True, config=config)
        # Set logical candle timestamp for intraday file naming (if scheduled)
        if run_timestamp:
            ta._run_timestamp = run_timestamp

        # Intercept graph streaming to emit SSE events
        init_state = ta.propagator.create_initial_state(ticker, trade_date)
        args = ta.propagator.get_graph_args()
        ta.ticker = ticker

        # Rebuild graph for crypto if needed
        from tradingagents.dataflows.asset_detection import is_crypto
        if is_crypto(ticker):
            print(f"[Analysis {job_id}] Crypto asset detected, rebuilding with crypto tools")
            ta._rebuild_graph_for_asset(ticker, ["market", "social", "news", "fundamentals"])
            # Log which tools are now assigned
            for node_name, tool_node in ta.tool_nodes.items():
                try:
                    # LangGraph ToolNode stores tools in tools_by_name dict
                    if hasattr(tool_node, 'tools_by_name'):
                        tool_names = list(tool_node.tools_by_name.keys())
                    else:
                        tool_names = [type(tool_node).__name__]
                    print(f"[Analysis {job_id}] Tools for {node_name}: {tool_names}")
                except Exception as e:
                    print(f"[Analysis {job_id}] Could not list tools for {node_name}: {e}")

        print(f"[Analysis {job_id}] Starting analysis for {ticker} on {trade_date}")
        
        seen_steps = set()
        reports_cache = {}  # Track reports to avoid duplicates
        trace = []

        # Retry logic for rate-limited requests
        max_retries = config.get("ensemble_max_retries", 2)
        base_delay = 2.0
        stream_success = False
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"[Analysis {job_id}] Retry attempt {attempt}/{max_retries}")
                
                trace = []
                for chunk in ta.graph.stream(init_state, **args):
                    trace.append(chunk)
                    
                    # Detect step progress based on what fields are present
                    step_info = _detect_step_from_chunk(chunk, seen_steps)
                    
                    if step_info and step_info["key"] not in seen_steps:
                        seen_steps.add(step_info["key"])
                        print(f"[Analysis {job_id}] Step completed: {step_info['label']}", flush=True)
                        eq.put({
                            "event": "agent_start",
                            "agent": step_info["label"],
                            "step": step_info["step"],
                            "total": TOTAL_STEPS,
                        })

                    # Check for report fields to emit
                    for report_key in ["market_report", "sentiment_report", "news_report", "fundamentals_report"]:
                        report = chunk.get(report_key, "")
                        if report and report != reports_cache.get(report_key):
                            reports_cache[report_key] = report
                            eq.put({
                                "event": "agent_report",
                                "report_key": report_key,
                                "report": report[:5000],
                            })
                
                # Success - break out of retry loop
                stream_success = True
                break
                
            except Exception as stream_err:
                error_str = str(stream_err)
                is_rate_limit = "429" in error_str or "rate-limit" in error_str.lower()
                
                if is_rate_limit and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    print(f"[Analysis {job_id}] Rate limited (attempt {attempt + 1}), waiting {delay}s before retry...")
                    eq.put({
                        "event": "agent_update",
                        "step": 0,
                        "content": f"Rate limited by provider, retrying in {int(delay)}s...",
                    })
                    time.sleep(delay)
                    continue
                else:
                    # Not a rate limit or out of retries - re-raise
                    raise
        
        if not stream_success:
            raise RuntimeError("Failed to complete analysis after all retries")

        # Extract final state
        final_state = trace[-1] if trace else {}
        ta.curr_state = final_state
        ta._log_state(trade_date, final_state)

        # Process signal
        from tradingagents.graph.signal_processing import SignalProcessor
        decision = ta.process_signal(final_state.get("final_trade_decision", "HOLD"))

        # Normalize: process_signal returns str OR dict - always extract a string signal
        if isinstance(decision, dict):
            decision_signal = str(decision.get("signal", "HOLD")).upper()
            stop_loss_price = decision.get("stop_loss_price")
            take_profit_price = decision.get("take_profit_price")
            confidence = decision.get("confidence")
            max_hold_days = decision.get("max_hold_days")
            reasoning = decision.get("reasoning")
        else:
            decision_signal = str(decision).upper() if decision else "HOLD"
            stop_loss_price = None
            take_profit_price = None
            confidence = None
            max_hold_days = None
            reasoning = None

        # Run confidence scorer (regime context + calibration + position sizing)
        scored = {}
        try:
            from tradingagents.backtesting.regime import detect_regime_context
            from tradingagents.graph.confidence import ConfidenceScorer
            from tradingagents.dataflows.asset_detection import is_crypto as _is_crypto_asset

            regime_ctx = detect_regime_context(ticker, trade_date)

            # Crypto-specific guards: cap hold days and SHORT stop distance
            if _is_crypto_asset(ticker):
                if max_hold_days and max_hold_days > 7:
                    max_hold_days = 7
                if (decision_signal in ('SHORT', 'SELL')
                        and regime_ctx.get('current_price')
                        and stop_loss_price):
                    max_crypto_stop = regime_ctx['current_price'] * 1.12
                    if stop_loss_price > max_crypto_stop:
                        stop_loss_price = max_crypto_stop

            scored = ConfidenceScorer().score(
                llm_confidence=confidence if confidence is not None else 0.50,
                ticker=ticker,
                signal=decision_signal,
                knowledge_store=ta.backtest_knowledge_store,
                regime_ctx=regime_ctx,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                max_hold_days=int(max_hold_days) if max_hold_days else 7,
                reasoning=reasoning or "",
            )
            confidence = scored.get("confidence", confidence)
            print(f"[Analysis {job_id}] Scored: conf={confidence:.3f} size={scored.get('position_size_pct', 0):.1%} R={scored.get('r_ratio')} gated={scored.get('gated')}")
        except Exception as _score_err:
            print(f"[Analysis {job_id}] Confidence scorer error (non-fatal): {_score_err}")
            import traceback; traceback.print_exc()

        # Fallback R:R from raw prices if scorer didn't provide one
        _sr_rr = scored.get("r_ratio")
        _sr_rr_warn = scored.get("r_ratio_warning", False)
        if _sr_rr is None and stop_loss_price and take_profit_price:
            try:
                from tradingagents.backtesting.regime import detect_regime_context as _drc
                _entry = _drc(ticker, trade_date).get("current_price")
                if _entry and _entry > 0:
                    _risk = abs(_entry - stop_loss_price)
                    _reward = abs(take_profit_price - _entry)
                    if _risk > 0:
                        _sr_rr = round(_reward / _risk, 3)
                        _sr_rr_warn = _sr_rr < 1.0
            except Exception:
                pass

        # SL/TP sanity warnings: SL inside daily ATR → likely to be hunted
        _sl_atr_warning = False
        _tp_atr_warning = False
        try:
            _entry_px = regime_ctx.get("current_price") if regime_ctx else None
            _daily_vol = regime_ctx.get("daily_vol", 0.03) if regime_ctx else 0.03
            if _entry_px and _entry_px > 0 and stop_loss_price:
                sl_dist_pct = abs(stop_loss_price - _entry_px) / _entry_px
                if sl_dist_pct < _daily_vol:
                    _sl_atr_warning = True
            if _entry_px and _entry_px > 0 and take_profit_price and max_hold_days:
                tp_dist_pct = abs(take_profit_price - _entry_px) / _entry_px
                # TP unrealistic if distance > hold_days * daily_vol * 1.5
                max_reasonable_tp = max_hold_days * _daily_vol * 1.5
                if tp_dist_pct > max_reasonable_tp:
                    _tp_atr_warning = True
        except Exception:
            pass

        result = {
            "ticker": ticker,
            "date": run_timestamp if run_timestamp else trade_date,
            "decision": decision_signal,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "confidence": confidence,
            "max_hold_days": max_hold_days,
            "reasoning": reasoning,
            "position_size_pct": scored.get("position_size_pct"),
            "conviction_label": scored.get("conviction_label"),
            "gated": scored.get("gated", False),
            "r_ratio": _sr_rr,
            "r_ratio_warning": _sr_rr_warn,
            "sl_atr_warning": _sl_atr_warning,
            "tp_atr_warning": _tp_atr_warning,
            "hold_period_scalar": scored.get("hold_period_scalar"),
            "hedge_penalty_applied": scored.get("hedge_penalty_applied"),
            "market_report": final_state.get("market_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "news_report": final_state.get("news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "investment_debate": {
                "bull_history": final_state.get("investment_debate_state", {}).get("bull_history", ""),
                "bear_history": final_state.get("investment_debate_state", {}).get("bear_history", ""),
                "judge_decision": final_state.get("investment_debate_state", {}).get("judge_decision", ""),
            },
            "risk_debate": {
                "aggressive": final_state.get("risk_debate_state", {}).get("aggressive_history", ""),
                "conservative": final_state.get("risk_debate_state", {}).get("conservative_history", ""),
                "neutral": final_state.get("risk_debate_state", {}).get("neutral_history", ""),
                "judge_decision": final_state.get("risk_debate_state", {}).get("judge_decision", ""),
            },
            "trader_plan": final_state.get("trader_investment_plan", ""),
            "final_trade_decision": final_state.get("final_trade_decision", ""),
        }

        # Embed all the processed structural risk parameters BACK into the trace history and save it identically
        final_state.update(result)
        # _log_state checks for "signal" key to persist structured fields; result uses "decision"
        if "signal" not in final_state and "decision" in result:
            final_state["signal"] = result["decision"]
        ta._log_state(run_timestamp if run_timestamp else trade_date, final_state)

        print(f"[Analysis {job_id}] Analysis complete, sending done event")
        jobs[job_id]["result"] = result
        jobs[job_id]["status"] = "done"

        # Dispatch Telegram push notification
        try:
            _send_telegram_alert(result)
        except Exception as e:
            print(f"[Analysis {job_id}] Telegram alert warning: {e}")

        # Auto-record shadow decision for later scoring
        try:
            shadow_dir = EVAL_RESULTS_DIR / "shadow" / ticker
            shadow_dir.mkdir(parents=True, exist_ok=True)

            # Get real entry price BEFORE building shadow entry
            _shadow_price = None
            try:
                _shadow_price = _get_price_on_date(ticker, trade_date)
            except Exception:
                pass

            # Detect regime at decision time (not retroactively at scoring time)
            _shadow_regime = "unknown"
            try:
                from tradingagents.backtesting.regime import detect_regime_context
                _shadow_regime = detect_regime_context(ticker, trade_date).get("regime", "unknown")
            except Exception:
                pass

            shadow_entry = {
                "ticker": ticker,
                "date": trade_date,
                "signal": decision_signal,
                "price": _shadow_price,
                "confidence": confidence,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "max_hold_days": max_hold_days,
                "position_size_pct": scored.get("position_size_pct") if scored else None,
                "reasoning": reasoning,
                "regime": _shadow_regime,
                "source": "live_analysis",
                "recorded_at": datetime.now().isoformat(),
                "scored": False,
            }
            with open(shadow_dir / "decisions.jsonl", "a") as _sf:
                _sf.write(json.dumps(shadow_entry, default=str) + "\n")
            _push_shadow_async(ticker)
            _push_history_async(ticker)
        except Exception as _shadow_err:
            print(f"[Analysis {job_id}] Shadow record failed (non-fatal): {_shadow_err}")

        # Background auto-scoring for all tickers with pending decisions
        _trigger_background_scoring(ticker)

        eq.put({"event": "decision", "signal": decision_signal})
        eq.put({"event": "done", "result": result})
        print(f"[Analysis {job_id}] Done event sent")

    except Exception as e:
        print(f"[Analysis {job_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        
        error_str = str(e)
        is_api_issue = any(k in error_str.lower() for k in ("rate limit", "timeout", "429", "502", "upstream", "api issue"))
        if is_api_issue or "all" in error_str.lower() and "failed" in error_str.lower():
            user_message = (
                "Analysis aborted: the AI provider APIs are overloaded or rate-limited. "
                "No partial results were saved. Please wait a few minutes and try again."
            )
        else:
            user_message = f"Analysis failed: {str(e)[:200]}"
        
        eq.put({"event": "error", "message": user_message})
        # Small delay to ensure error event is flushed before thread ends
        time.sleep(0.5)

# ── Scorecard API Endpoints ───────────────────────────────────────────

@app.get("/api/shadow/scorecard/{ticker}")
async def get_scorecard(ticker: str):
    """Get the full forward-test scorecard for a ticker."""
    try:
        from tradingagents.backtesting.scorecard import get_scorecard as _get_scorecard
        return _get_scorecard(ticker, str(EVAL_RESULTS_DIR))
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/shadow/score/{ticker}")
async def score_decisions(ticker: str):
    """Trigger scoring of pending decisions against actual future prices."""
    try:
        from tradingagents.backtesting.scorecard import score_pending_decisions
        result = score_pending_decisions(ticker, str(EVAL_RESULTS_DIR))
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/shadow/calibrate/{ticker}")
async def calibrate_ticker(ticker: str):
    """Run the 10-decision calibration study to compute overconfidence correction."""
    try:
        from tradingagents.backtesting.scorecard import run_calibration_study
        result = run_calibration_study(ticker, results_dir=str(EVAL_RESULTS_DIR))
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/shadow/walk-forward/{ticker}")
async def walk_forward_validate(ticker: str):
    """Run walk-forward validation on existing analysis logs."""
    try:
        from tradingagents.backtesting.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(ticker, str(EVAL_RESULTS_DIR))
        result = validator.validate()
        return result
    except Exception as e:
        return {"error": str(e)}


# ── API Endpoints ─────────────────────────────────────────────────────

@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health(tick: int = 0):
    """Health check + optional self-scheduling pulse tick (free-tier survival).

    Query param ``?tick=1`` turns this endpoint into a "fire pulses if due"
    trigger. Point an external uptime pinger (UptimeRobot, cron-job.org, etc.)
    at ``/api/health?tick=1`` every N minutes and you get:
        1. The Render free-tier idle timer reset (no cold start)
        2. Pulses fired on cadence, even if the in-process asyncio scheduler
           died during the 15-min idle window.

    Cadence is taken from the persisted pulse scheduler state; if the pulse
    scheduler is DISABLED in the UI, no pulses are fired regardless of tick=1.
    """
    ran: list = []
    if tick:
        try:
            # Only self-tick when the pulse scheduler is enabled in the UI.
            if _pulse_state.get("enabled"):
                interval_min = int(
                    _pulse_state.get("interval_minutes", _PULSE_INTERVAL_MINUTES)
                )
                last_run_iso = _pulse_state.get("last_run")
                now = datetime.now(timezone.utc)
                due = True
                if last_run_iso:
                    try:
                        last_run_dt = datetime.fromisoformat(last_run_iso)
                        if last_run_dt.tzinfo is None:
                            last_run_dt = last_run_dt.replace(tzinfo=timezone.utc)
                        due = (now - last_run_dt).total_seconds() >= interval_min * 60 - 5
                    except Exception:
                        due = True
                if due:
                    tickers = _pulse_state.get("tickers", ["BTC-USD"])
                    loop = asyncio.get_event_loop()
                    # Run each ticker in the default threadpool (don't block health check)
                    async def _fire_one(tkr: str):
                        return await loop.run_in_executor(None, _run_single_pulse, tkr)
                    results = await asyncio.gather(
                        *(_fire_one(t) for t in tickers),
                        return_exceptions=True,
                    )
                    for t, r in zip(tickers, results):
                        if isinstance(r, Exception):
                            ran.append({"ticker": t, "ok": False, "error": str(r)})
                        else:
                            try:
                                _append_pulse(t, r)
                                ran.append({
                                    "ticker": t, "ok": True,
                                    "signal": r.get("signal"),
                                    "confidence": r.get("confidence"),
                                })
                            except Exception as e:
                                ran.append({"ticker": t, "ok": False, "error": str(e)})
                    _pulse_state["last_run"] = now.isoformat()
                    _pulse_state["last_status"] = "ok (self-tick)"
                    _save_pulse_scheduler_state()
        except Exception as e:
            logger.warning(f"[Health] Self-tick failed: {e}")
            ran.append({"ok": False, "error": str(e)})

    # 4H auto-analysis self-tick (plan Part 2): fire the scheduled analysis
    # if its previous-boundary window is due and hasn't been claimed yet.
    # This is the failsafe that survives Render free-tier restarts where
    # the in-process asyncio loop died.
    analysis_ran: list = []
    if tick and _scheduler_state.get("enabled"):
        try:
            import pytz
            now_utc = datetime.now(pytz.utc)
            next_boundary = _next_4h_utc_boundary()
            prev_boundary = next_boundary - timedelta(hours=_SCHEDULER_INTERVAL_HOURS)
            physical_trigger = prev_boundary + timedelta(seconds=_DATA_DELAY_SECONDS)
            expected_label = prev_boundary.strftime("%Y-%m-%dT%H")

            if now_utc >= physical_trigger:
                # Atomic boundary claim
                launched = []
                with _BOUNDARY_CLAIM_LOCK:
                    if _scheduler_state.get("last_run") != expected_label:
                        _scheduler_state["last_run"] = expected_label
                        _scheduler_state["last_status"] = "ok (self-tick)"
                        _save_scheduler_state()
                        claim_won = True
                    else:
                        claim_won = False

                if claim_won:
                    logical_date = prev_boundary.strftime("%Y-%m-%d")
                    for ticker in _scheduler_state.get("tickers", _SCHEDULER_TICKERS):
                        if not _try_claim_ticker(ticker):
                            logger.info(f"[Health] Self-tick skipping {ticker} — analysis already in flight")
                            continue
                        job_id = str(uuid.uuid4())[:8]
                        eq = JobEventQueue()
                        jobs[job_id] = {
                            "ticker": ticker,
                            "date": logical_date,
                            "status": "running",
                            "queue": eq,
                            "result": None,
                            "error": None,
                            "scheduled": True,
                            "candle_time": expected_label,
                        }
                        def _run_and_release(jid=job_id, tkr=ticker, ld=logical_date, ll=expected_label):
                            try:
                                _run_analysis(jid, tkr, ld, True, ll)
                            finally:
                                _release_ticker(tkr)
                        thread = threading.Thread(target=_run_and_release, daemon=True)
                        thread.start()
                        launched.append({"ticker": ticker, "job_id": job_id})
                        logger.info(f"[Health] Self-tick launched 4H analysis {job_id} for {ticker} @ {expected_label}")
                    analysis_ran.append({"boundary": expected_label, "launched": launched})
        except Exception as e:
            logger.warning(f"[Health] 4H self-tick failed: {e}")
            analysis_ran.append({"ok": False, "error": str(e)})

    return {
        "status": "ok",
        "version": "1.0.0",
        "features": {
            "analysis": True,
            "backtest": True,
            "history": True,
            "streaming": True,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pulse": {
            "enabled": _pulse_state.get("enabled", False),
            "interval_minutes": _pulse_state.get(
                "interval_minutes", _PULSE_INTERVAL_MINUTES
            ),
            "last_run": _pulse_state.get("last_run"),
            "last_status": _pulse_state.get("last_status"),
            "tickers": _pulse_state.get("tickers", []),
        },
        "ticked": tick == 1,
        "ran": ran,
        "scheduler": {
            "enabled": _scheduler_state.get("enabled", False),
            "last_run": _scheduler_state.get("last_run"),
            "last_status": _scheduler_state.get("last_status"),
            "tickers": _scheduler_state.get("tickers", []),
        },
        "analysis_ran": analysis_ran,
    }


@app.post("/api/analyze")
async def start_analysis(req: AnalyzeRequest):
    ticker = req.ticker.upper()
    trade_date = req.date or date.today().strftime("%Y-%m-%d")

    job_id = str(uuid.uuid4())[:8]
    eq = JobEventQueue()

    # Generate unique local timestamp for manual runs: YYYY-MM-DD-HH-MM-AM/PM
    now_local = datetime.now()
    am_pm = now_local.strftime("%p")  # AM or PM
    run_timestamp = now_local.strftime(f"%Y-%m-%d-%I-%M-{am_pm}")

    jobs[job_id] = {
        "ticker": ticker,
        "date": run_timestamp,  # Use timestamped key for history
        "trade_date": trade_date,  # Original trade date for data lookup
        "status": "running",
        "queue": eq,
        "result": None,
        "error": None,
    }

    # Start analysis in background thread with force_refresh=True by default for live runs
    force_refresh = True if req.force_refresh is None else req.force_refresh
    thread = threading.Thread(target=_run_analysis, args=(job_id, ticker, trade_date, force_refresh, run_timestamp), daemon=True)
    thread.start()

    return {"job_id": job_id, "ticker": ticker, "date": run_timestamp}


@app.get("/api/stream/{job_id}")
async def stream_analysis(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    eq: JobEventQueue = jobs[job_id]["queue"]
    eq.set_loop(asyncio.get_event_loop())
    my_queue = eq._queue

    stream_start = time.time()

    async def event_generator():
        while True:
            if eq._queue is not my_queue:
                break
            try:
                event = await asyncio.wait_for(my_queue.get(), timeout=30)
                try:
                    event_data = json.dumps(event, default=str)
                    yield {"event": event.get("event", "message"), "data": event_data}
                except (TypeError, ValueError) as e:
                    logger.error(f"[SSE {job_id}] Failed to serialize event: {e}")
                    yield {
                        "event": "error",
                        "data": json.dumps({"event": "error", "message": f"Event serialization failed: {e}"}),
                    }
                    break
                if event.get("event") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                elapsed = int(time.time() - stream_start)
                yield {
                    "event": "heartbeat",
                    "data": json.dumps({"event": "heartbeat", "elapsed": elapsed}),
                }
            except Exception as e:
                logger.error(f"[SSE {job_id}] Unexpected error in event generator: {e}")
                yield {
                    "event": "error",
                    "data": json.dumps({"event": "error", "message": f"Stream error: {e}"}),
                }
                break

    return EventSourceResponse(event_generator())


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "ticker": job["ticker"],
        "date": job["date"],
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error"),
    }


# ── History Endpoints ─────────────────────────────────────────────────

@app.get("/api/history")
async def list_tickers():
    """List all tickers that have analysis logs."""
    if not EVAL_RESULTS_DIR.exists():
        return {"tickers": []}

    tickers = []
    for d in sorted(EVAL_RESULTS_DIR.iterdir()):
        if d.is_dir():
            logs_dir = d / "TradingAgentsStrategy_logs"
            if logs_dir.exists():
                json_files = list(logs_dir.glob("full_states_log_*.json"))
                tickers.append({
                    "ticker": d.name,
                    "analysis_count": len(json_files),
                    "latest_date": _extract_latest_date(json_files),
                })
    return {"tickers": tickers}


def _extract_latest_date(files) -> Optional[str]:
    dates = []
    for f in files:
        # Match both full_states_log_YYYY-MM-DD.json and full_states_log_YYYY-MM-DDTHH.json
        match = re.search(r"full_states_log_(\d{4}-\d{2}-\d{2}(?:T\d{2})?)\.json", f.name)
        if match:
            dates.append(match.group(1))
    return max(dates) if dates else None


@app.get("/api/history/{ticker}")
async def list_analyses(ticker: str):
    """List all analysis dates for a given ticker, including intraday timestamped files."""
    logs_dir = EVAL_RESULTS_DIR / ticker / "TradingAgentsStrategy_logs"
    if not logs_dir.exists():
        raise HTTPException(status_code=404, detail=f"No analyses found for {ticker}")

    analyses = []
    for f in sorted(logs_dir.glob("full_states_log_*.json"), reverse=True):
        # Match three formats:
        # 1. Legacy daily: YYYY-MM-DD
        # 2. Legacy hourly: YYYY-MM-DDTHH
        # 3. New manual format: YYYY-MM-DD-HH-MM-AM/PM
        legacy_match = re.search(r"full_states_log_(\d{4}-\d{2}-\d{2})(T(\d{2}))?(\.json)$", f.name)
        new_match = re.search(r"full_states_log_(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{2})-(AM|PM)\.json$", f.name)

        if new_match:
            # New format: YYYY-MM-DD-HH-MM-AM/PM
            analysis_date = new_match.group(1)
            hour_12 = int(new_match.group(2))
            minute = new_match.group(3)
            am_pm = new_match.group(4)
            # Convert to 24-hour for internal use
            hour_24 = hour_12 if am_pm == "AM" else (hour_12 % 12) + 12
            if hour_12 == 12 and am_pm == "AM":
                hour_24 = 0  # Midnight
            candle_time = f"{analysis_date}-{new_match.group(2)}-{minute}-{am_pm}"
            # Format for display: HH:MM AM/PM
            time_label = f"{hour_12}:{minute} {am_pm}"
            local_date = analysis_date
        elif legacy_match:
            # Legacy formats
            analysis_date = legacy_match.group(1)
            hour_str = legacy_match.group(3)  # e.g. '16' or None
            candle_time = f"{analysis_date}T{hour_str}" if hour_str else analysis_date
            time_label = None
            local_date = analysis_date
            if hour_str:
                # Emit a strict ISO 8601 UTC string to let the frontend localize it
                time_label = f"{analysis_date}T{hour_str}:00:00Z"
                try:
                    utc_dt = datetime.fromisoformat(time_label.replace("Z", "+00:00"))
                    local_date = utc_dt.astimezone().strftime("%Y-%m-%d")
                except Exception:
                    local_date = analysis_date
        else:
            continue

        # Extract signal from nested date key in log file
        try:
            data = json.loads(f.read_text())
            # Try multiple key formats: new format, candle_time, analysis_date
            date_data = data.get(candle_time) or data.get(analysis_date) or {}
            decision_text = date_data.get("final_trade_decision", "")
            signal = _extract_signal(decision_text)
            confidence = date_data.get("confidence")
        except Exception:
            signal = "UNKNOWN"
            confidence = None

        analyses.append({
            "date": analysis_date,
            "local_date": local_date,
            "candle_time": candle_time,
            "time": time_label,
            "signal": signal,
            "confidence": confidence,
            "file": f.name,
        })

    return {"ticker": ticker, "analyses": analyses}


def _extract_signal(text: str) -> str:
    """Extract trading signal from decision text."""
    text_upper = text.upper()
    for signal in ["SHORT", "COVER", "OVERWEIGHT", "UNDERWEIGHT", "BUY", "SELL", "HOLD"]:
        if signal in text_upper:
            return signal
    return "UNKNOWN"


def _ensemble_result_to_dict(consensus_result, ticker: str, date: str) -> dict:
    """Convert ConsensusResult to standard result dict format.
    
    Args:
        consensus_result: ConsensusResult from ensemble orchestrator
        ticker: Ticker symbol
        date: Analysis date
        
    Returns:
        Result dict compatible with existing system
    """
    # Build ensemble metadata for Telegram and history
    individual_signals = [
        {
            "signal": r.get("signal", "HOLD"),
            "confidence": r.get("confidence", 0.5),
        }
        for r in consensus_result.individual_results
    ]
    
    ensemble_metadata = {
        "runs": consensus_result.ensemble_metadata.get("runs", 3),
        "valid_runs": consensus_result.ensemble_metadata.get("valid_runs", 3),
        "retry_count": consensus_result.ensemble_metadata.get("retry_count", 0),
        "divergence_metrics": consensus_result.divergence_metrics,
        "individual_signals": individual_signals,
        "entry_price_snapshot": consensus_result.ensemble_metadata.get("entry_price_snapshot"),
        "stale_price_warning": consensus_result.ensemble_metadata.get("stale_price_warning", False),
    }
    
    # Determine conviction label from confidence
    confidence = consensus_result.confidence
    if confidence >= 0.80:
        conviction_label = "VERY HIGH"
    elif confidence >= 0.65:
        conviction_label = "HIGH"
    elif confidence >= 0.50:
        conviction_label = "MODERATE"
    else:
        conviction_label = "LOW"
    
    return {
        "ticker": ticker,
        "date": date,
        "decision": consensus_result.signal,
        "stop_loss_price": consensus_result.stop_loss_price,
        "take_profit_price": consensus_result.take_profit_price,
        "confidence": confidence,
        "max_hold_days": consensus_result.max_hold_days,
        "reasoning": consensus_result.reasoning,
        "position_size_pct": None,  # Will be computed by ConfidenceScorer
        "conviction_label": conviction_label,
        "gated": False,  # Will be computed by ConfidenceScorer
        "r_ratio": consensus_result.divergence_metrics.get("r_ratio_consensus"),
        "r_ratio_warning": False,
        "hold_period_scalar": None,
        "hedge_penalty_applied": 0,
        "ensemble_metadata": ensemble_metadata,
        # Reports will be from best individual result
        "market_report": consensus_result.individual_results[0].get("market_report", "") if consensus_result.individual_results else "",
        "sentiment_report": consensus_result.individual_results[0].get("sentiment_report", "") if consensus_result.individual_results else "",
        "news_report": consensus_result.individual_results[0].get("news_report", "") if consensus_result.individual_results else "",
        "fundamentals_report": consensus_result.individual_results[0].get("fundamentals_report", "") if consensus_result.individual_results else "",
        "final_trade_decision": consensus_result.reasoning,
    }

def _send_telegram_alert(result: dict):
    """Send Telegram alert for analysis completion.
    
    Guarantees alerts are sent for all analysis types (manual, scheduled, ensemble)
    with proper formatting and correct timestamps.
    """
    # Use module-level logger via sys.modules for proper test mocking
    import sys
    _logger = sys.modules[__name__].logger
    
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        _logger.warning("Telegram not configured: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing")
        return
    
    def _escape_html(text: str) -> str:
        """Escape HTML special characters for Telegram HTML parse mode."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    def _format_reasoning(decision_text: str) -> str:
        """Parse and format decision reasoning for Telegram display."""
        if not decision_text:
            return "<i>No reasoning provided</i>"
        
        # Try to parse as JSON
        try:
            # Strip markdown fences if present
            cleaned = decision_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                if len(lines) > 2:
                    cleaned = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
            
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                reasoning = parsed.get("reasoning", "")
                
                parts = []
                if reasoning:
                    max_len = 800
                    reason_text = reasoning[:max_len] + "..." if len(reasoning) > max_len else reasoning
                    parts.append(f"<b>Analysis:</b> {_escape_html(reason_text)}")
                
                return "\n".join(parts) if parts else f"<code>{_escape_html(decision_text[:500])}</code>"
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Not JSON - treat as plain text/markdown
        # Strip markdown code fences
        cleaned = decision_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if len(lines) > 2 and lines[-1].startswith("```"):
                cleaned = "\n".join(lines[1:-1])
            elif lines[0].startswith("```"):
                cleaned = "\n".join(lines[1:])
        
        # Truncate and escape
        max_len = 1000
        if len(cleaned) > max_len:
            cleaned = cleaned[:max_len] + "...\n\n<i>(truncated for Telegram)</i>"
        
        return _escape_html(cleaned)
    
    def _parse_timestamp(raw_date: str) -> str:
        """Parse various timestamp formats and return formatted local time."""
        if not raw_date:
            return "Unknown"
        
        try:
            from datetime import datetime
            import re
            
            # Format 1: New manual format YYYY-MM-DD-HH-MM-AM/PM
            new_format_match = re.match(r"(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{2})-(AM|PM)", raw_date)
            if new_format_match:
                date_part = new_format_match.group(1)
                hour_12 = int(new_format_match.group(2))
                minute = new_format_match.group(3)
                am_pm = new_format_match.group(4)
                return f"{date_part} at {hour_12}:{minute} {am_pm}"
            
            # Format 2: Scheduler format YYYY-MM-DDTHH
            if "T" in raw_date and ":" not in raw_date:
                from datetime import timezone
                normalized = f"{raw_date}:00:00"
                dt = datetime.fromisoformat(normalized.replace("Z", ""))
                dt = dt.replace(tzinfo=timezone.utc)
                # Always display in the user's configured timezone (EST/EDT),
                # NOT the server host timezone. Render / other cloud hosts run
                # in UTC which otherwise shows mismatched 'EDT' vs 'UTC'
                # across alerts and when a friend in another TZ opens the UI.
                local_dt = dt.astimezone(_USER_DISPLAY_TZ)
                return local_dt.strftime("%b %d, %Y at %I:%M %p %Z")
            
            # Format 3: ISO format YYYY-MM-DDTHH:MM:SS
            if "T" in raw_date:
                from datetime import timezone
                dt = datetime.fromisoformat(raw_date.replace("Z", "").replace("+00:00", ""))
                dt = dt.replace(tzinfo=timezone.utc)
                local_dt = dt.astimezone(_USER_DISPLAY_TZ)
                return local_dt.strftime("%b %d, %Y at %I:%M %p %Z")
            
            # Format 4: Simple date YYYY-MM-DD
            if re.match(r"\d{4}-\d{2}-\d{2}$", raw_date):
                dt = datetime.strptime(raw_date, "%Y-%m-%d")
                return dt.strftime("%b %d, %Y")
            
            return raw_date
        except Exception as e:
            _logger.debug(f"Failed to parse timestamp {raw_date}: {e}")
            return raw_date
    
    signal = result.get("decision", "UNKNOWN")
    signal_emoji = "🛑" if signal in ["SHORT", "SELL"] else "🟢" if signal in ["BUY", "COVER"] else "⚖️"
    conf = result.get("confidence") or 0
    
    # Ensure stop loss / take profit handle empty paths
    sl = result.get("stop_loss_price")
    tp = result.get("take_profit_price")
    sl_str = f"${float(sl):,.2f}" if sl else "N/A"
    tp_str = f"${float(tp):,.2f}" if tp else "N/A"
    sizing = result.get("position_size_pct") or 0
    
    # Parse timestamp with new format support
    raw_date = result.get('date', '')
    fmt_date = _parse_timestamp(raw_date)
    
    rr_value = result.get("r_ratio")
    rr_prefix = f"R:R Ratio: {rr_value:.2f}:1 | " if rr_value is not None else "R:R Ratio: N/A | "
    conviction = result.get("conviction_label", "MODERATE")
    
    # Ensemble metadata section
    ensemble_meta = result.get("ensemble_metadata", {})
    ensemble_section = ""
    if ensemble_meta:
        runs = ensemble_meta.get("runs", 1)
        valid = ensemble_meta.get("valid_runs", runs)
        retries = ensemble_meta.get("retry_count", 0)
        
        ensemble_section = f"\n🔀 <b>Ensemble:</b> {valid}/{runs} runs"
        if retries > 0:
            ensemble_section += f" (retries: {retries})"
        
        # Show divergence metrics
        div = ensemble_meta.get("divergence_metrics", {})
        if "confidence_range" in div:
            ensemble_section += f" | Range: {div['confidence_range']:.1%}"
        if "signal_agreement" in div:
            ensemble_section += f" | Agreement: {div['signal_agreement']:.0%}"
        
        # Show individual signals
        individual = ensemble_meta.get("individual_signals", [])
        if individual:
            sig_strs = []
            for sig in individual:
                emoji = "🟢" if sig["signal"] in ["BUY", "COVER"] else "🔴" if sig["signal"] in ["SHORT", "SELL"] else "⚖️"
                sig_strs.append(f"{emoji}{sig['signal']} {sig['confidence']:.0%}")
            ensemble_section += f"\n📊 <b>Individual:</b> {' | '.join(sig_strs)}"
        
        # Stale price warning
        if ensemble_meta.get("stale_price_warning"):
            ensemble_section += "\n⚠️ <i>Warning: Prices may be stale due to slow execution</i>"
    
    # Format reasoning section
    decision_text = result.get("final_trade_decision", "")
    reasoning_section = _format_reasoning(decision_text)

    # Calibration status
    _cal_correction = result.get("overconfidence_correction")
    _cal_scored = result.get("scored_count_used")
    if _cal_correction is not None and _cal_scored is not None:
        cal_line = f"\n📊 Calibration: {_cal_scored}/60 scored (correction: {_cal_correction:.0%})"
    else:
        cal_line = ""
    
    msg = (
        f"🚨 <b>Trading Agent Alert: {result.get('ticker')}</b> 🚨\n"
        f"🗓 <b>Time:</b> {fmt_date}"
        f"{ensemble_section}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>Decision:</b> {signal_emoji} {signal}\n"
        f"⚖️ <b>Confidence:</b> {conf * 100:.1f}%\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "💰 <b>Action Plan:</b>\n"
        f"• <b>Size:</b> {sizing * 100:.1f}%\n"
        f"• <b>Stop Loss:</b> {sl_str}\n"
        f"• <b>Take Profit:</b> {tp_str}\n"
        f"• <b>Hold:</b> {result.get('max_hold_days', 'N/A')} Period(s)\n\n"
        "📈 <b>Metrics:</b>\n"
        f"{rr_prefix}Conviction: {conviction}"
        f"{cal_line}\n\n"
        "📝 <b>Bot Reasoning:</b>\n"
        f"{reasoning_section}"
    )
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        _logger.info(f"Telegram alert sent for {result.get('ticker')} @ {raw_date}")
    except requests.exceptions.RequestException as e:
        _logger.error(f"Failed to send Telegram alert: {e}")


@app.get("/api/history/{ticker}/{analysis_date}")
async def get_analysis(ticker: str, analysis_date: str):
    """Return full log JSON for a specific analysis, augmented with risk parameters."""
    log_file = EVAL_RESULTS_DIR / ticker / "TradingAgentsStrategy_logs" / f"full_states_log_{analysis_date}.json"
    if not log_file.exists():
        raise HTTPException(status_code=404, detail=f"No analysis found for {ticker} on {analysis_date}")

    data = json.loads(log_file.read_text())

    # Extract the base date to parse correctly via fallback hierarchy
    # Handle both legacy format (YYYY-MM-DDTHH) and new format (YYYY-MM-DD-HH-MM-AM/PM)
    if "-" in analysis_date and "T" not in analysis_date:
        # New format with dashes - extract YYYY-MM-DD portion (first 10 chars)
        base_date = analysis_date[:10] if len(analysis_date) >= 10 else analysis_date
    else:
        # Legacy format with T separator
        base_date = analysis_date.split("T")[0]
    date_data = data.get(analysis_date) or data.get(base_date) or data

    # Reconstruct risk parameters that are normally only present during live run
    decision_text = date_data.get("final_trade_decision", "")
    
    # NEW FIX: Strongly prefer explicit parameters saved via the live Telegram injection
    decision_signal = date_data.get("decision")
    stop_loss_price = date_data.get("stop_loss_price")
    take_profit_price = date_data.get("take_profit_price")
    confidence = date_data.get("confidence")
    max_hold_days = date_data.get("max_hold_days")

    if not decision_signal:
        try:
            cleaned = decision_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                if len(lines) > 2:
                    cleaned = "\n".join(lines[1:-1]) if lines[-1].strip().startswith("```") else "\n".join(lines[1:])
            parsed = json.loads(cleaned)
            decision_signal = _extract_signal(parsed.get("signal", ""))
            stop_loss_price = stop_loss_price or parsed.get("stop_loss_price")
            take_profit_price = take_profit_price or parsed.get("take_profit_price")
            confidence = confidence if confidence is not None else parsed.get("confidence")
            max_hold_days = max_hold_days if max_hold_days is not None else parsed.get("max_hold_days")
        except Exception:
            decision_signal = _extract_signal(str(decision_text).upper())

    try:
        from tradingagents.graph.confidence import ConfidenceScorer
        from tradingagents.backtesting.regime import detect_regime_context
        from tradingagents.dataflows.asset_detection import is_crypto as _is_crypto_asset
        from tradingagents.backtesting.knowledge_store import BacktestKnowledgeStore

        regime_ctx = detect_regime_context(ticker, analysis_date)
        
        # Load actual knowledge store for calibration (BLOCKER FIX: was None)
        ks = BacktestKnowledgeStore(str(EVAL_RESULTS_DIR))
        
        # Apply crypto guards
        if _is_crypto_asset(ticker):
            if max_hold_days and max_hold_days > 7:
                max_hold_days = 7
            if decision_signal in ('SHORT', 'SELL') and regime_ctx.get('current_price') and stop_loss_price:
                max_crypto_stop = regime_ctx['current_price'] * 1.12
                if stop_loss_price > max_crypto_stop:
                    stop_loss_price = max_crypto_stop

        scored = ConfidenceScorer(results_dir=str(EVAL_RESULTS_DIR)).score(
            llm_confidence=confidence if confidence is not None else 0.50,
            ticker=ticker,
            signal=decision_signal,
            knowledge_store=ks,
            regime_ctx=regime_ctx,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            max_hold_days=max_hold_days if max_hold_days is not None else 3
        )

        date_data["decision"] = decision_signal
        date_data["stop_loss_price"] = stop_loss_price
        date_data["take_profit_price"] = take_profit_price
        date_data["confidence"] = scored.get("confidence", confidence if confidence is not None else 0.50)
        date_data["max_hold_days"] = max_hold_days
        date_data["position_size_pct"] = scored.get("position_size_pct")
        date_data["conviction_label"] = scored.get("conviction_label")
        date_data["r_ratio"] = scored.get("r_ratio")
        date_data["gated"] = scored.get("gated", False)
        date_data["r_ratio_warning"] = scored.get("r_ratio_warning", False)
        date_data["hold_period_scalar"] = scored.get("hold_period_scalar")
        date_data["hedge_penalty_applied"] = scored.get("hedge_penalty_applied")
    except Exception as e:
        print(f"Failed to rebuild risk parameters for history {analysis_date}: {e}")

    # Add formatted local time with AM/PM for display
    try:
        from datetime import timezone
        # Handle new format: YYYY-MM-DD-HH-MM-AM/PM
        new_format_match = re.match(r"(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{2})-(AM|PM)", analysis_date)
        if new_format_match:
            # Already in desired display format
            date_part = new_format_match.group(1)
            hour = new_format_match.group(2)
            minute = new_format_match.group(3)
            am_pm = new_format_match.group(4)
            # Convert to nicer display format
            date_formatted = f"{date_part} at {hour}:{minute} {am_pm}"
        else:
            # Legacy format with T separator — always render in user's
            # configured display TZ (see _USER_DISPLAY_TZ) so the UI and
            # Telegram alerts agree regardless of server host timezone.
            analysis_dt = datetime.fromisoformat(analysis_date.replace("Z", ""))
            analysis_dt = analysis_dt.replace(tzinfo=timezone.utc)
            local_dt = analysis_dt.astimezone(_USER_DISPLAY_TZ)
            date_formatted = local_dt.strftime("%b %d, %Y at %I:%M %p %Z")
    except Exception:
        date_formatted = analysis_date

    return {
        "ticker": ticker,
        "date": analysis_date,
        "date_formatted": date_formatted,
        "data": date_data,
    }


# ── Price Data Endpoint ──────────────────────────────────────────────

@app.get("/api/price/{ticker}")
async def get_price(ticker: str, days: int = 90, interval: str = "1d"):
    """Fetch OHLCV + SMA data for chart rendering."""
    try:
        stock = yf.Ticker(ticker)
        
        if interval == "4h":
            # For 4h, max period is 60d. 60d has enough 4h candles (360) for a 200 SMA
            hist = stock.history(period="60d", interval="4h")
        else:
            fetch_days = days + 200
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=fetch_days)
            hist = stock.history(start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="1d")

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

        # Compute SMAs on full dataset
        hist["SMA50"] = hist["Close"].rolling(window=50).mean()
        hist["SMA200"] = hist["Close"].rolling(window=200).mean()

        # Trim to requested chunks (each day has ~6 chunks of 4 hours)
        tail_candles = days * 6 if interval == "4h" else days
        hist = hist.tail(tail_candles)

        records = []
        for idx, row in hist.iterrows():
            # For 4h, datetime includes hour/min. Ensure frontend UI gracefully parses it
            dt_str = idx.strftime("%Y-%m-%dT%H:00:00Z") if interval == "4h" else idx.strftime("%Y-%m-%d")
            records.append({
                "date": dt_str,
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]),
                "sma50": round(row["SMA50"], 2) if not (row["SMA50"] != row["SMA50"]) else None,
                "sma200": round(row["SMA200"], 2) if not (row["SMA200"] != row["SMA200"]) else None,
            })

        return {
            "ticker": ticker,
            "days": days,
            "data": records,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Backtest Helper Functions ─────────────────────────────────────────

import math

def _sanitize_json(obj):
    """Replace inf/NaN float values with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    elif isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    return obj


from tradingagents.backtesting.engine import _generate_trade_dates, _get_price_on_date, _get_funding_on_date


def _extract_signal_from_text(text: str) -> str:
    """Extract trading signal from decision text."""
    text_upper = text.upper()
    for signal in ["SHORT", "COVER", "OVERWEIGHT", "UNDERWEIGHT", "BUY", "SELL", "HOLD"]:
        if signal in text_upper:
            return signal
    return "HOLD"


def _add_backtest_log(job_id: str, message: str):
    """Add a log message to the backtest job."""
    if job_id in backtest_jobs:
        if "logs" not in backtest_jobs[job_id]:
            backtest_jobs[job_id]["logs"] = []
        timestamp = datetime.now().isoformat()
        backtest_jobs[job_id]["logs"].append({"timestamp": timestamp, "message": message})
        # Keep only last 50 logs
        backtest_jobs[job_id]["logs"] = backtest_jobs[job_id]["logs"][-50:]
        print(f"[Backtest {job_id}] {message}")


def _run_backtest(job_id: str, ticker: str, start_date: str, end_date: str, mode: str, config: Dict[str, Any]):
    """Run backtest in background thread with detailed logging."""
    eq: JobEventQueue = backtest_jobs[job_id]["queue"]
    start_time = datetime.now()
    
    def log(msg: str):
        _add_backtest_log(job_id, msg)
    
    try:
        log(f"=== STARTING BACKTEST ===")
        log(f"Ticker: {ticker}, Mode: {mode}")
        log(f"Date range: {start_date} to {end_date}")
        
        # Emit starting event
        log("Step 1/5: Initializing backtest...")
        eq.put({
            "event": "status",
            "step": 1,
            "total_steps": 5,
            "status": "Initializing backtest...",
            "details": f"Configuring {ticker}"
        })
        
        # Get config values with crypto defaults
        initial_capital = config.get("initial_capital", 100_000.0)
        position_size_pct = config.get("position_size_pct", 0.25)
        frequency = config.get("frequency", "weekly")
        leverage = config.get("leverage", 1.0)
        maker_fee = config.get("maker_fee", 0.0002)
        taker_fee = config.get("taker_fee", 0.0005)
        use_funding = config.get("use_funding", True)
        position_sizing = config.get("position_sizing", "fixed")
        
        log(f"Config: capital=${initial_capital:,.0f}, leverage={leverage}x, fees={taker_fee*100:.3f}%, sizing={position_sizing}")
        
        # Emit generating dates event
        log("Step 2/5: Generating trade dates...")
        eq.put({
            "event": "status",
            "step": 2,
            "total_steps": 5,
            "status": "Generating trade dates...",
            "details": f"Frequency: {frequency}"
        })
        
        # Generate trade dates
        trade_dates = _generate_trade_dates(start_date, end_date, frequency, ticker=ticker)
        total_dates = len(trade_dates)
        
        if total_dates == 0:
            raise ValueError("No trading dates generated for the given range")
        
        log(f"Generated {total_dates} trading dates")
        
        # Check if analysis logs exist for replay mode
        if mode == "replay":
            logs_dir = EVAL_RESULTS_DIR / ticker / "TradingAgentsStrategy_logs"
            
            # Discover ALL available log dates for this ticker
            all_log_dates = sorted([
                f.stem.replace("full_states_log_", "")
                for f in logs_dir.glob("full_states_log_*.json")
            ]) if logs_dir.exists() else []
            
            if not all_log_dates:
                # No logs at all for this ticker — find which tickers DO have logs
                available_tickers = []
                for ticker_dir in EVAL_RESULTS_DIR.iterdir():
                    if ticker_dir.is_dir():
                        tk_logs = list((ticker_dir / "TradingAgentsStrategy_logs").glob("full_states_log_*.json")) if (ticker_dir / "TradingAgentsStrategy_logs").exists() else []
                        if tk_logs:
                            dates_for_tk = sorted([f.stem.replace("full_states_log_", "") for f in tk_logs])
                            available_tickers.append(f"{ticker_dir.name} ({dates_for_tk[0]} to {dates_for_tk[-1]}, {len(dates_for_tk)} days)")
                
                ticker_list = ", ".join(available_tickers) if available_tickers else "none"
                raise ValueError(
                    f"No analysis logs found for {ticker}. "
                    f"Quick Replay backtesting requires previously-run LLM analysis. "
                    f"Go to Home → Analyze to run analysis for {ticker} first, or use Full Simulation mode. "
                    f"Tickers with available logs: {ticker_list}"
                )
            
            # Check which requested dates have logs
            base_dates = set([d.split(' ')[0] for d in trade_dates])
            matching_dates = [bd for bd in base_dates if bd in all_log_dates]
            
            if not matching_dates:
                # None of the requested dates have logs — auto-adjust to available range
                original_start = start_date
                original_end = end_date
                log(f"No logs in requested range ({start_date} to {end_date}). Available logs: {all_log_dates[0]} to {all_log_dates[-1]}")
                log(f"Auto-adjusting date range to available analysis logs...")

                # Regenerate trade dates using the full available log range
                trade_dates = _generate_trade_dates(all_log_dates[0], all_log_dates[-1], frequency, ticker=ticker)
                total_dates = len(trade_dates)

                if total_dates == 0:
                    raise ValueError(f"Could not generate trade dates from available logs ({all_log_dates[0]} to {all_log_dates[-1]})")

                # Update the config to reflect actual dates used
                start_date = all_log_dates[0]
                end_date = all_log_dates[-1]
                
                eq.put({
                    "event": "warning",
                    "type": "date_range_adjusted",
                    "step": 2,
                    "total_steps": 5,
                    "status": "Date range auto-adjusted — no logs in requested range",
                    "requested": f"{original_start} to {original_end}",
                    "actual": f"{start_date} to {end_date}",
                    "details": f"Using available logs: {start_date} to {end_date} ({len(all_log_dates)} analysis days)"
                })
            
            # Recount after potential adjustment
            base_dates_final = set([d.split(' ')[0] for d in trade_dates])
            available_logs = sum(1 for bd in base_dates_final if bd in all_log_dates)
            log(f"Found {available_logs}/{len(base_dates_final)} analysis logs for {total_dates} tick intervals")
            log(f"Available analysis dates: {', '.join(all_log_dates)}")
        
        # Emit sample-size warnings (HIGH 5)
        if total_dates < 10:
            eq.put({
                "event": "warning",
                "type": "sample_size_unreliable",
                "count": total_dates,
                "message": f"Only {total_dates} trading periods — metrics will be statistically unreliable (SE(Sharpe) > 0.39). Consider a wider date range."
            })
        elif total_dates < 30:
            eq.put({
                "event": "warning",
                "type": "sample_size_limited",
                "count": total_dates,
                "message": f"{total_dates} trading periods — metrics have limited statistical significance (SE(Sharpe) ≈ 0.22–0.39). 30+ periods recommended."
            })

        # Update progress tracking
        backtest_jobs[job_id]["progress"] = {"current": 0, "total": total_dates}
        
        # Emit processing event
        log(f"Step 3/5: Processing {total_dates} trading periods...")
        eq.put({
            "event": "status",
            "step": 3,
            "total_steps": 5,
            "status": f"Processing {total_dates} trading periods...",
            "details": f"Mode: {mode}"
        })
        
        if mode == "replay":
            result = _replay_backtest(ticker, trade_dates, initial_capital, position_size_pct, eq, leverage, maker_fee, taker_fee, use_funding, position_sizing, log)
        elif mode == "hybrid":
            log("Step 3/5: Starting hybrid backtest (replay + simulation for missing dates)...")
            result = _hybrid_backtest(job_id, ticker, trade_dates, initial_capital, position_size_pct, frequency, eq, leverage, maker_fee, taker_fee, use_funding, position_sizing, log)
        else:
            log("Step 3/5: Starting full simulation with LLM pipeline...")
            result = _simulate_backtest(job_id, ticker, trade_dates, initial_capital, position_size_pct, frequency, eq, leverage, maker_fee, taker_fee, use_funding, position_sizing, log)
        
        # Check timeout
        elapsed = (datetime.now() - start_time).total_seconds()
        log(f"Processing complete in {elapsed:.1f}s")
        
        # Emit calculating metrics event
        log("Step 4/5: Calculating performance metrics...")
        eq.put({
            "event": "status",
            "step": 4,
            "total_steps": 5,
            "status": "Calculating performance metrics...",
            "details": "Computing returns, Sharpe ratio, drawdowns, and crypto-specific metrics"
        })
        
        # Save results with crypto config
        result["job_id"] = job_id
        result["config"] = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "mode": mode,
            "initial_capital": initial_capital,
            "position_size_pct": position_size_pct,
            "frequency": frequency,
            "leverage": leverage,
            "maker_fee": maker_fee,
            "taker_fee": taker_fee,
            "use_funding": use_funding,
            "position_sizing": position_sizing,
        }
        result["created_at"] = datetime.now().isoformat()
        
        # Emit coverage warning if < 70%
        cov = result.get("coverage", {})
        cov_pct = cov.get("coverage_pct", 100)
        if cov_pct < 70:
            eq.put({
                "event": "warning",
                "type": "low_coverage",
                "coverage_pct": cov_pct,
                "dates_requested": cov.get("dates_requested", 0),
                "dates_processed": cov.get("dates_processed", 0),
                "message": f"Only {cov_pct:.0f}% of requested dates were processed. "
                           f"Sharpe, Sortino, and Calmar ratios may be unreliable."
            })
            log(f"WARNING: Low coverage ({cov_pct:.0f}%) — metrics may be unreliable")
        
        # Persist to disk
        log("Step 5/5: Saving results to disk...")
        eq.put({
            "event": "status",
            "step": 5,
            "total_steps": 5,
            "status": "Saving results...",
            "details": "Persisting backtest data to disk"
        })
        
        backtest_dir = EVAL_RESULTS_DIR / "backtests" / ticker
        backtest_dir.mkdir(parents=True, exist_ok=True)
        result_file = backtest_dir / f"backtest_{job_id}.json"
        with open(result_file, "w") as f:
            json.dump(_sanitize_json(result), f, indent=2, default=str)
        
        backtest_jobs[job_id]["result"] = result
        backtest_jobs[job_id]["status"] = "completed"
        
        elapsed = (datetime.now() - start_time).total_seconds()
        log(f"=== BACKTEST COMPLETE in {elapsed:.1f}s ===")
        
        eq.put({"event": "complete", "result": _sanitize_json(result)})
        
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        log(f"=== BACKTEST FAILED after {elapsed:.1f}s ===")
        log(f"ERROR: {str(e)}")
        import traceback
        log(traceback.format_exc())
        backtest_jobs[job_id]["status"] = "failed"
        backtest_jobs[job_id]["error"] = str(e)
        eq.put({"event": "error", "message": str(e)})


def _replay_backtest(ticker: str, trade_dates: List[str], initial_capital: float, position_size_pct: float, eq: JobEventQueue, leverage: float = 1.0, maker_fee: float = 0.0002, taker_fee: float = 0.0005, use_funding: bool = True, position_sizing: str = "fixed", log_func = None) -> Dict[str, Any]:
    """Replay backtest using cached analysis logs with crypto features."""
    from tradingagents.backtesting.portfolio import Portfolio
    from tradingagents.backtesting.metrics import compute_metrics
    
    def log(msg: str):
        if log_func:
            log_func(msg)
    
    portfolio = Portfolio(
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        leverage=leverage,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        use_funding=use_funding,
        position_sizing=position_sizing,
    )
    decisions = []
    errors = []
    
    for i, trade_date in enumerate(trade_dates):
        eq.put({
            "event": "progress",
            "current": i + 1,
            "total": len(trade_dates),
            "date": trade_date,
        })
        
        base_date = trade_date.split(" ")[0]
        log_file = EVAL_RESULTS_DIR / ticker / "TradingAgentsStrategy_logs" / f"full_states_log_{base_date}.json"
        
        if not log_file.exists():
            errors.append({"date": trade_date, "error": f"No analysis log found for {base_date}"})
            log(f"  Warning: No analysis log for {base_date}")
            continue
        
        try:
            data = json.loads(log_file.read_text())
            date_data = data.get(base_date, data)
            decision_text = date_data.get("final_trade_decision", "")
            signal = _extract_signal_from_text(decision_text)

            # Extract structured risk params from stored signal if available
            stop_loss_price = date_data.get("stop_loss_price")
            take_profit_price = date_data.get("take_profit_price")
            max_hold_days = date_data.get("max_hold_days")
            log_source = date_data.get("source", "real_analysis")

            price = _get_price_on_date(ticker, trade_date)
            if price is None:
                errors.append({"date": trade_date, "error": "No price data available"})
                log(f"  Warning: No price data for {trade_date}")
                continue

            # BLOCKER 3: Validate stale stop/take prices from old analyses
            # Threshold: 15% for crypto, 25% for equity
            from tradingagents.dataflows.asset_detection import is_crypto as _is_crypto_asset
            _stale_threshold = 0.15 if _is_crypto_asset(ticker) else 0.25
            _DEFAULT_STOP_PCT = 0.07
            _DEFAULT_TAKE_PCT = 0.20
            if stop_loss_price is not None and price > 0:
                if abs(stop_loss_price - price) / price > _stale_threshold:
                    if signal in ("SHORT", "SELL"):
                        stop_loss_price = price * (1 + _DEFAULT_STOP_PCT)
                        take_profit_price = price * (1 - _DEFAULT_TAKE_PCT)
                    else:
                        stop_loss_price = price * (1 - _DEFAULT_STOP_PCT)
                        take_profit_price = price * (1 + _DEFAULT_TAKE_PCT)
                    log(f"  {trade_date}: Stale stop recalculated (>{_stale_threshold*100:.0f}% from current price)")

            funding_rate = _get_funding_on_date(ticker, trade_date)
            action = portfolio.process_signal(
                signal, price, trade_date,
                funding_rate=funding_rate,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                max_hold_days=max_hold_days,
            )
            log(f"  {trade_date}: {signal} @ ${price:,.2f} → {action}")
            
            decisions.append({
                "date": trade_date,
                "signal": signal,
                "price": price,
                "action": action,
                "portfolio_value": portfolio.portfolio_value(price),
                "position": portfolio.position_side.value,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "source": log_source,
            })
            
            eq.put({
                "event": "decision",
                "date": trade_date,
                "signal": signal,
                "price": price,
                "action": action,
                "portfolio_value": portfolio.portfolio_value(price),
                "source": log_source,
            })
            
        except Exception as e:
            errors.append({"date": trade_date, "error": str(e)})
            log(f"  Error processing {trade_date}: {str(e)}")
            continue
    
    if trade_dates:
        final_price = _get_price_on_date(ticker, trade_dates[-1])
        if final_price and portfolio.current_position:
            portfolio.force_close(final_price, trade_dates[-1], taker_fee)
            log(f"  Final close: {trade_dates[-1]} @ ${final_price:,.2f}")
    
    log(f"Processed {len(decisions)} dates with {len(errors)} errors")

    # Get portfolio stats
    portfolio_stats = portfolio.get_stats()

    # Compute benchmark (buy & hold) return for alpha (HIGH 7)
    benchmark_return_pct = None
    try:
        if trade_dates:
            start_price = _get_price_on_date(ticker, trade_dates[0])
            end_price = _get_price_on_date(ticker, trade_dates[-1])
            if start_price and end_price and start_price > 0:
                benchmark_return_pct = (end_price - start_price) / start_price * 100
    except Exception:
        pass

    from tradingagents.dataflows.asset_detection import is_crypto as _is_crypto_replay
    metrics = compute_metrics(
        equity_curve=portfolio.equity_curve,
        closed_positions=portfolio.closed_positions,
        initial_capital=initial_capital,
        total_fees=portfolio.total_fees_paid,
        total_funding=portfolio.total_funding_paid,
        liquidations=portfolio.liquidations,
        leverage=leverage,
        is_crypto=_is_crypto_replay(ticker),
        benchmark_return_pct=benchmark_return_pct,
    )
    
    dates_requested = len(trade_dates)
    dates_processed = len(decisions)
    coverage_pct = (dates_processed / dates_requested * 100) if dates_requested > 0 else 0

    return {
        "metrics": metrics,
        "portfolio_stats": portfolio_stats,
        "decisions": decisions,
        "equity_curve": portfolio.equity_curve,
        "trade_history": [
            {
                "date": t.date,
                "signal": t.signal,
                "price": t.price,
                "action": t.action_taken,
                "position": t.position_side,
                "portfolio_value": t.portfolio_value,
                "unrealized_pnl": t.unrealized_pnl,
                "realized_pnl": t.realized_pnl,
                "fees_paid": t.fees_paid,
                "funding_paid": t.funding_paid,
                "leverage": t.leverage,
            }
            for t in portfolio.trade_history
        ],
        "errors": errors,
        "coverage": {
            "dates_requested": dates_requested,
            "dates_processed": dates_processed,
            "coverage_pct": round(coverage_pct, 1),
        },
    }


def _hybrid_backtest(job_id: str, ticker: str, trade_dates: List[str], initial_capital: float, position_size_pct: float, frequency: str, eq: JobEventQueue, leverage: float = 1.0, maker_fee: float = 0.0002, taker_fee: float = 0.0005, use_funding: bool = True, position_sizing: str = "fixed", log_func = None) -> Dict[str, Any]:
    """Hybrid backtest: replay existing logs, simulate missing dates with live LLM."""
    from tradingagents.backtesting.portfolio import Portfolio
    from tradingagents.backtesting.metrics import compute_metrics
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.dataflows.asset_detection import is_crypto as _is_crypto_hybrid

    def log(msg: str):
        if log_func:
            log_func(msg)

    # Discover available replay logs
    logs_dir = EVAL_RESULTS_DIR / ticker / "TradingAgentsStrategy_logs"
    available_log_dates = set()
    if logs_dir.exists():
        available_log_dates = {
            f.stem.replace("full_states_log_", "")
            for f in logs_dir.glob("full_states_log_*.json")
        }

    # Count missing dates and estimate time
    missing_dates = []
    replay_dates = []
    for td in trade_dates:
        base_date = td.split(" ")[0]
        if base_date in available_log_dates:
            replay_dates.append(td)
        else:
            missing_dates.append(td)

    n_missing = len(missing_dates)
    n_replay = len(replay_dates)
    estimated_seconds = n_missing * 75

    log(f"Hybrid mode: {n_replay} replay dates, {n_missing} simulation dates")
    log(f"Estimated time for simulations: {estimated_seconds // 60}m {estimated_seconds % 60}s")

    # Emit time estimate warning if > 30 minutes
    if estimated_seconds > 1800:
        eq.put({
            "event": "warning",
            "type": "hybrid_time_estimate",
            "replay_count": n_replay,
            "simulation_count": n_missing,
            "estimated_minutes": round(estimated_seconds / 60),
            "message": f"Hybrid backtest will simulate {n_missing} dates (~{estimated_seconds // 60} min). "
                       f"{n_replay} dates have existing analysis logs for fast replay."
        })

    # Initialize portfolio
    portfolio = Portfolio(
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        leverage=leverage,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        use_funding=use_funding,
        position_sizing=position_sizing,
    )
    decisions = []
    errors = []
    replay_count = 0
    simulation_count = 0

    # Create a dedicated TradingAgentsGraph for simulation dates (thread-safe: one per job)
    graph = None
    if n_missing > 0:
        sim_config = DEFAULT_CONFIG.copy()
        sim_config["backtest_mode"] = True
        graph = TradingAgentsGraph(
            selected_analysts=["market", "social", "news", "fundamentals"],
            debug=False,
            config=sim_config,
        )
        if _is_crypto_hybrid(ticker):
            graph._rebuild_graph_for_asset(ticker, ["market", "social", "news", "fundamentals"])

    for i, trade_date in enumerate(trade_dates):
        eq.put({
            "event": "progress",
            "current": i + 1,
            "total": len(trade_dates),
            "date": trade_date,
        })

        base_date = trade_date.split(" ")[0]
        price = _get_price_on_date(ticker, trade_date)
        if price is None:
            errors.append({"date": trade_date, "error": "No price data available"})
            log(f"  Warning: No price data for {trade_date}")
            continue

        signal = None
        source = None
        stop_loss_price = None
        take_profit_price = None
        max_hold_days = None
        decision_text = ""

        # Try replay first
        log_file = logs_dir / f"full_states_log_{base_date}.json"
        if log_file.exists():
            try:
                data = json.loads(log_file.read_text())
                date_data = data.get(base_date, data)
                decision_text = date_data.get("final_trade_decision", "")
                signal = _extract_signal_from_text(decision_text)
                stop_loss_price = date_data.get("stop_loss_price")
                take_profit_price = date_data.get("take_profit_price")
                max_hold_days = date_data.get("max_hold_days")
                source = "replay"
                replay_count += 1

                # Validate stale stop/take prices
                from tradingagents.dataflows.asset_detection import is_crypto as _is_c
                _stale_threshold = 0.15 if _is_c(ticker) else 0.25
                _DEFAULT_STOP_PCT = 0.07
                _DEFAULT_TAKE_PCT = 0.20
                if stop_loss_price is not None and price > 0:
                    if abs(stop_loss_price - price) / price > _stale_threshold:
                        if signal in ("SHORT", "SELL"):
                            stop_loss_price = price * (1 + _DEFAULT_STOP_PCT)
                            take_profit_price = price * (1 - _DEFAULT_TAKE_PCT)
                        else:
                            stop_loss_price = price * (1 - _DEFAULT_STOP_PCT)
                            take_profit_price = price * (1 + _DEFAULT_TAKE_PCT)

            except Exception as e:
                log(f"  Replay failed for {base_date}: {e}, falling back to simulation")

        # Fall back to live LLM simulation for missing dates
        if signal is None and graph is not None:
            try:
                log(f"  Simulating {trade_date} via LLM pipeline...")
                final_state, proc_signal = graph.propagate(ticker, trade_date)
                signal = proc_signal.get("signal", "HOLD") if isinstance(proc_signal, dict) else str(proc_signal)
                stop_loss_price = final_state.get("stop_loss_price")
                take_profit_price = final_state.get("take_profit_price")
                max_hold_days = final_state.get("max_hold_days")
                decision_text = final_state.get("final_trade_decision", "")
                source = "simulation"
                simulation_count += 1

                # Save the simulation result as a log for future replay
                log_save_dir = logs_dir
                log_save_dir.mkdir(parents=True, exist_ok=True)
                log_save_file = log_save_dir / f"full_states_log_{base_date}.json"
                if not log_save_file.exists():
                    save_data = {
                        base_date: {
                            "company_of_interest": ticker,
                            "trade_date": base_date,
                            "final_trade_decision": decision_text,
                            "stop_loss_price": stop_loss_price,
                            "take_profit_price": take_profit_price,
                            "max_hold_days": max_hold_days,
                            "source": "hybrid_simulation",
                        }
                    }
                    log_save_file.write_text(json.dumps(save_data, indent=2, default=str))
                    log(f"    Saved simulation log for future replay")

            except Exception as e:
                errors.append({"date": trade_date, "error": f"Simulation failed: {e}"})
                log(f"  Simulation failed for {trade_date}: {e}")
                continue

        if signal is None:
            errors.append({"date": trade_date, "error": "No signal from replay or simulation"})
            continue

        # Process signal through portfolio
        funding_rate = _get_funding_on_date(ticker, trade_date)
        action = portfolio.process_signal(
            signal, price, trade_date,
            funding_rate=funding_rate,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            max_hold_days=max_hold_days,
        )
        log(f"  {trade_date}: [{source}] {signal} @ ${price:,.2f} → {action}")

        decision = {
            "date": trade_date,
            "signal": signal,
            "price": price,
            "action": action,
            "portfolio_value": portfolio.portfolio_value(price),
            "position": portfolio.position_side.value,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "source": source,
        }
        if source == "simulation":
            decision["look_ahead_caveat"] = True
        decisions.append(decision)

        eq.put({
            "event": "decision",
            "date": trade_date,
            "signal": signal,
            "price": price,
            "action": action,
            "portfolio_value": portfolio.portfolio_value(price),
            "source": source,
        })

    # Force close any open position
    if trade_dates:
        final_price = _get_price_on_date(ticker, trade_dates[-1])
        if final_price and portfolio.current_position:
            portfolio.force_close(final_price, trade_dates[-1], taker_fee)
            log(f"  Final close: {trade_dates[-1]} @ ${final_price:,.2f}")

    log(f"Hybrid complete: {replay_count} replayed, {simulation_count} simulated, {len(errors)} errors")

    # Compute metrics
    portfolio_stats = portfolio.get_stats()
    benchmark_return_pct = None
    try:
        if trade_dates:
            start_price = _get_price_on_date(ticker, trade_dates[0])
            end_price = _get_price_on_date(ticker, trade_dates[-1])
            if start_price and end_price and start_price > 0:
                benchmark_return_pct = (end_price - start_price) / start_price * 100
    except Exception:
        pass

    metrics = compute_metrics(
        equity_curve=portfolio.equity_curve,
        closed_positions=portfolio.closed_positions,
        initial_capital=initial_capital,
        total_fees=portfolio.total_fees_paid,
        total_funding=portfolio.total_funding_paid,
        liquidations=portfolio.liquidations,
        leverage=leverage,
        is_crypto=_is_crypto_hybrid(ticker),
        benchmark_return_pct=benchmark_return_pct,
    )

    return {
        "metrics": metrics,
        "portfolio_stats": portfolio_stats,
        "decisions": decisions,
        "equity_curve": portfolio.equity_curve,
        "trade_history": [
            {
                "date": t.date,
                "signal": t.signal,
                "price": t.price,
                "action": t.action_taken,
                "position": t.position_side,
                "portfolio_value": t.portfolio_value,
                "unrealized_pnl": t.unrealized_pnl,
                "realized_pnl": t.realized_pnl,
                "fees_paid": t.fees_paid,
                "funding_paid": t.funding_paid,
                "leverage": t.leverage,
            }
            for t in portfolio.trade_history
        ],
        "errors": errors,
        "hybrid_stats": {
            "replay_count": replay_count,
            "simulation_count": simulation_count,
            "total_dates": len(trade_dates),
        },
        "coverage": {
            "dates_requested": len(trade_dates),
            "dates_processed": len(decisions),
            "coverage_pct": round(len(decisions) / len(trade_dates) * 100, 1) if trade_dates else 0,
        },
    }


def _simulate_backtest(job_id: str, ticker: str, trade_dates: List[str], initial_capital: float, position_size_pct: float, frequency: str, eq: JobEventQueue, leverage: float = 1.0, maker_fee: float = 0.0002, taker_fee: float = 0.0005, use_funding: bool = True, position_sizing: str = "fixed", log_func = None) -> Dict[str, Any]:
    """Full simulation backtest using LLM agent pipeline with crypto features."""
    from tradingagents.backtesting.engine import BacktestEngine
    from tradingagents.default_config import DEFAULT_CONFIG
    
    def log(msg: str):
        if log_func:
            log_func(msg)
    
    config = DEFAULT_CONFIG.copy()
    
    # Create engine with crypto parameters
    log(f"Creating BacktestEngine for {ticker}")
    engine = BacktestEngine(
        ticker=ticker,
        start_date=trade_dates[0],
        end_date=trade_dates[-1],
        config=config,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        trading_frequency=frequency,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
    )
    
    # Update engine portfolio with crypto settings
    engine.portfolio.leverage = leverage
    engine.portfolio.maker_fee = maker_fee
    engine.portfolio.taker_fee = taker_fee
    engine.portfolio.use_funding = use_funding
    engine.portfolio.position_sizing = position_sizing
    
    log(f"Engine configured: leverage={leverage}x, fees={taker_fee*100:.3f}%")
    
    # Patch engine to emit progress events
    original_process = engine.portfolio.process_signal
    
    def patched_process(signal: str, price: float, date: str, **kwargs) -> str:
        result = original_process(signal, price, date, **kwargs)
        current_idx = trade_dates.index(date) if date in trade_dates else 0
        eq.put({
            "event": "progress",
            "current": current_idx + 1,
            "total": len(trade_dates),
            "date": date,
        })
        eq.put({
            "event": "decision",
            "date": date,
            "signal": signal,
            "price": price,
            "action": result,
            "portfolio_value": engine.portfolio.portfolio_value(price),
        })
        log(f"  {date}: {signal} @ ${price:,.2f} → {result}")
        return result
    
    engine.portfolio.process_signal = patched_process
    
    # Run the backtest
    log("Starting engine.run() - this may take several minutes...")
    raw_result = engine.run()
    log("Engine run complete")
    
    # Re-compute metrics with crypto parameters (engine.run() already uses is_crypto)
    # engine.run() now returns metrics with correct is_crypto, just use them
    from tradingagents.backtesting.metrics import compute_metrics
    from tradingagents.dataflows.asset_detection import is_crypto as _is_crypto_sim
    _benchmark_pct = raw_result.get("metrics", {}).get("benchmark_return_pct")
    crypto_metrics = compute_metrics(
        equity_curve=engine.portfolio.equity_curve,
        closed_positions=engine.portfolio.closed_positions,
        initial_capital=initial_capital,
        total_fees=engine.portfolio.total_fees_paid,
        total_funding=engine.portfolio.total_funding_paid,
        liquidations=engine.portfolio.liquidations,
        leverage=leverage,
        is_crypto=_is_crypto_sim(ticker),
        benchmark_return_pct=_benchmark_pct,
    )
    
    # Build trade_history from engine's portfolio (engine.run() doesn't include this)
    trade_history = [
        {
            "date": t.date,
            "signal": t.signal,
            "price": t.price,
            "action": t.action_taken,
            "position": t.position_side,
            "portfolio_value": t.portfolio_value,
            "unrealized_pnl": t.unrealized_pnl,
            "realized_pnl": t.realized_pnl,
            "fees_paid": t.fees_paid,
            "funding_paid": t.funding_paid,
            "leverage": t.leverage,
        }
        for t in engine.portfolio.trade_history
    ]
    
    n_decisions = len(raw_result.get("decisions", []))
    result = {
        "metrics": crypto_metrics,
        "portfolio_stats": engine.portfolio.get_stats(),
        "decisions": raw_result.get("decisions", []),
        "equity_curve": engine.portfolio.equity_curve,
        "trade_history": trade_history,
        "errors": raw_result.get("errors", []),
        "coverage": {
            "dates_requested": len(trade_dates),
            "dates_processed": n_decisions if n_decisions > 0 else len(trade_history),
            "coverage_pct": round((n_decisions if n_decisions > 0 else len(trade_history)) / len(trade_dates) * 100, 1) if trade_dates else 0,
        },
    }
    
    log(f"Simulation result: {len(trade_history)} trade log entries, {crypto_metrics.get('total_trades', 0)} closed trades")
    
    return result


# ── Backtest API Endpoints ────────────────────────────────────────────

@app.post("/api/backtest")
async def start_backtest(req: BacktestRequest):
    """Start a new backtest job."""
    ticker = req.ticker.upper()
    
    # Validate date range
    try:
        start = datetime.strptime(req.start_date, "%Y-%m-%d")
        end = datetime.strptime(req.end_date, "%Y-%m-%d")
        if end < start:
            raise HTTPException(status_code=400, detail="End date must be after start date")
        # Max 2 years
        if (end - start).days > 730:
            raise HTTPException(status_code=400, detail="Date range cannot exceed 2 years")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # HIGH 10: Block 4h frequency in full-simulation mode
    req_frequency = (req.config or {}).get("frequency", "weekly")
    if req_frequency == "4h" and req.mode not in ("replay", "hybrid"):
        raise HTTPException(
            status_code=400,
            detail="4h frequency is only supported in replay/hybrid mode. Use weekly/daily for simulation."
        )
    
    job_id = str(uuid.uuid4())[:8]
    eq = JobEventQueue()
    
    backtest_jobs[job_id] = {
        "job_id": job_id,
        "ticker": ticker,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "mode": req.mode,
        "status": "running",
        "queue": eq,
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
    }
    
    # Get config with defaults
    config = req.config or {}
    config.setdefault("initial_capital", 100_000.0)
    config.setdefault("position_size_pct", 0.25)
    config.setdefault("frequency", "weekly")
    
    # Start backtest in background thread
    thread = threading.Thread(
        target=_run_backtest,
        args=(job_id, ticker, req.start_date, req.end_date, req.mode, config),
        daemon=True
    )
    thread.start()
    
    return {
        "job_id": job_id,
        "ticker": ticker,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "mode": req.mode,
        "status": "running",
    }


@app.get("/api/backtest/stream/{job_id}")
async def stream_backtest(job_id: str):
    """SSE stream for backtest progress."""
    if job_id not in backtest_jobs:
        raise HTTPException(status_code=404, detail="Backtest job not found")
    
    eq: JobEventQueue = backtest_jobs[job_id]["queue"]
    eq.set_loop(asyncio.get_event_loop())
    
    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(eq.get(), timeout=120)
                safe_event = _sanitize_json(event)
                yield {"event": safe_event.get("event", "message"), "data": json.dumps(safe_event, default=str)}
                if event.get("event") in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": "{}"}
    
    return EventSourceResponse(event_generator())


@app.get("/api/backtest/active")
async def list_active_backtests():
    """List currently running backtest jobs."""
    active = [
        {
            "job_id": job_id,
            "ticker": job["ticker"],
            "start_date": job["start_date"],
            "end_date": job["end_date"],
            "mode": job["mode"],
            "status": job["status"],
            "created_at": job["created_at"],
            "progress": job.get("progress", {}),
            "logs": job.get("logs", [])[-10:],  # last 10 logs
        }
        for job_id, job in backtest_jobs.items()
        if job["status"] == "running"
    ]
    return {"active": active}

@app.get("/api/backtests/recent")
async def get_recent_backtests(limit: int = 20):
    """Get recent backtests with full metrics for Backtest History page."""
    results = []
    seen_backtests = set()  # Track unique ticker+date combinations
    
    # Search disk for saved backtests
    backtest_dir = EVAL_RESULTS_DIR / "backtests"
    if not backtest_dir.exists():
        return results
    
    # Collect all backtest files with timestamps
    all_files = []
    for ticker_dir in backtest_dir.iterdir():
        if not ticker_dir.is_dir():
            continue
        for result_file in ticker_dir.glob("backtest_*.json"):
            all_files.append(result_file)
    
    # Sort by modification time (most recent first)
    all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    for result_file in all_files:
        try:
            with open(result_file) as f:
                data = _sanitize_json(json.load(f))
                metrics = data.get("metrics", {})
                config = data.get("config", {})
                
                ticker = config.get("ticker", "UNKNOWN")
                start_date = config.get("start_date", "")
                end_date = config.get("end_date", "")
                
                # Create unique key for deduplication
                backtest_key = f"{ticker}_{start_date}_{end_date}"
                
                # Skip if we've already seen this backtest
                if backtest_key in seen_backtests:
                    continue
                
                seen_backtests.add(backtest_key)
                
                results.append({
                    "id": data.get("job_id", result_file.stem),
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "frequency": config.get("frequency", "daily"),
                    "mode": config.get("mode", "replay"),
                    "created_at": datetime.fromtimestamp(result_file.stat().st_mtime).isoformat(),
                    "total_return_pct": metrics.get("total_return_pct", 0),
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
                    "total_trades": metrics.get("total_trades", 0),
                    "win_rate_pct": metrics.get("win_rate_pct", 0),
                })
                
                # Stop once we have enough unique backtests
                if len(results) >= limit:
                    break
                    
        except Exception as e:
            logger.error(f"Error loading backtest {result_file}: {e}")
            continue
    
    return results

@app.get("/api/backtests/lessons/{ticker}")
async def get_backtest_lessons(ticker: str):
    """Get backtest-derived lessons for a ticker."""
    try:
        from tradingagents.backtesting.knowledge_store import BacktestKnowledgeStore
        store = BacktestKnowledgeStore(str(EVAL_RESULTS_DIR))
        # Load latest saved lessons (no re-generation to keep it fast)
        lessons = store.feedback_generator.load_latest_lessons(ticker.upper())
        if not lessons:
            # Generate fresh if none exist
            lessons = store.feedback_generator.generate_lessons(ticker.upper(), min_trades=3)
            if lessons:
                store.feedback_generator.save_lessons(ticker.upper(), lessons)
        return {"ticker": ticker.upper(), "lessons": lessons, "count": len(lessons)}
    except Exception as e:
        logger.error(f"Error getting lessons for {ticker}: {e}")
        return {"ticker": ticker.upper(), "lessons": [], "count": 0}


@app.get("/api/backtest/results")
async def list_backtest_results(ticker: Optional[str] = None, limit: int = 50):
    """List historical backtest results."""
    results = []
    
    # Search disk for saved backtests
    backtest_dir = EVAL_RESULTS_DIR / "backtests"
    if not backtest_dir.exists():
        return {"results": []}
    
    for ticker_dir in backtest_dir.iterdir():
        if not ticker_dir.is_dir():
            continue
        if ticker and ticker_dir.name.upper() != ticker.upper():
            continue
        
        for result_file in sorted(ticker_dir.glob("backtest_*.json"), reverse=True):
            try:
                with open(result_file) as f:
                    data = _sanitize_json(json.load(f))
                    results.append({
                        "job_id": data.get("job_id"),
                        "ticker": data.get("config", {}).get("ticker"),
                        "start_date": data.get("config", {}).get("start_date"),
                        "end_date": data.get("config", {}).get("end_date"),
                        "mode": data.get("config", {}).get("mode"),
                        "total_return_pct": data.get("metrics", {}).get("total_return_pct"),
                        "sharpe_ratio": data.get("metrics", {}).get("sharpe_ratio"),
                        "created_at": data.get("created_at"),
                    })
            except Exception:
                continue
            
            if len(results) >= limit:
                break
        
        if len(results) >= limit:
            break
    
    return {"results": results}


@app.get("/api/backtest/{job_id}")
async def get_backtest_result(job_id: str):
    """Get backtest result by job ID."""
    # Check in-memory first
    if job_id in backtest_jobs:
        job = backtest_jobs[job_id]
        return {
            "job_id": job_id,
            "status": job["status"],
            "result": _sanitize_json(job.get("result")),
            "error": job.get("error"),
        }
    
    # Check disk for completed jobs
    for ticker_dir in EVAL_RESULTS_DIR.glob("backtests/*"):
        result_file = ticker_dir / f"backtest_{job_id}.json"
        if result_file.exists():
            with open(result_file) as f:
                return {"job_id": job_id, "status": "completed", "result": _sanitize_json(json.load(f))}
    
    raise HTTPException(status_code=404, detail="Backtest job not found")


# ── Shadow Mode MVP ───────────────────────────────────────────────────

SHADOW_DIR = EVAL_RESULTS_DIR / "shadow"


def _push_shadow_async(ticker: str) -> None:
    """Best-effort fire-and-forget push of shadow decisions to GitHub Gist.

    Free-tier survival for analysis-based backtests — mirrors the pulse
    gist sync path. No-op when GITHUB_TOKEN / PULSE_GIST_ID aren't set.
    """
    try:
        from tradingagents.pulse import gist_sync
        if gist_sync.is_enabled():
            threading.Thread(
                target=gist_sync.push_shadow,
                args=(SHADOW_DIR, ticker),
                daemon=True,
            ).start()
    except Exception as _e:
        print(f"[GistSync] Shadow push dispatch failed (non-fatal): {_e}")


def _push_history_async(ticker: str) -> None:
    """Best-effort fire-and-forget push of full analysis logs to the
    HISTORY gist so the History page survives Render free-tier wipes.
    Uses a separate gist (HISTORY_GIST_ID) to isolate large payloads
    from the hot-path pulse gist.
    """
    try:
        from tradingagents.pulse import gist_sync
        if gist_sync.is_history_enabled():
            threading.Thread(
                target=gist_sync.push_history,
                args=(EVAL_RESULTS_DIR, ticker),
                daemon=True,
            ).start()
    except Exception as _e:
        print(f"[GistSync] History push dispatch failed (non-fatal): {_e}")


class ShadowDecision(BaseModel):
    ticker: str
    date: str  # yyyy-mm-dd
    signal: str  # BUY, SELL, SHORT, HOLD, etc.
    price: float
    confidence: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: Optional[str] = None
    source: str = "live_analysis"  # live_analysis | manual


@app.post("/api/shadow/record")
async def shadow_record(decision: ShadowDecision):
    """Record a shadow (paper-trade) decision without executing."""
    ticker = decision.ticker.upper()
    ticker_dir = SHADOW_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "ticker": ticker,
        "date": decision.date,
        "signal": decision.signal.upper(),
        "price": decision.price,
        "confidence": decision.confidence,
        "stop_loss": decision.stop_loss,
        "take_profit": decision.take_profit,
        "reasoning": decision.reasoning,
        "source": decision.source,
        "recorded_at": datetime.now().isoformat(),
    }

    # Append to JSONL file (one line per decision, fast append)
    log_file = ticker_dir / "decisions.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    _push_shadow_async(ticker)

    return {"status": "recorded", "ticker": ticker, "date": decision.date, "signal": entry["signal"]}


@app.get("/api/shadow/decisions/{ticker}")
async def shadow_decisions(ticker: str, limit: int = Query(default=100, le=500)):
    """Retrieve shadow decisions for a ticker."""
    ticker = ticker.upper()
    log_file = SHADOW_DIR / ticker / "decisions.jsonl"
    if not log_file.exists():
        return {"ticker": ticker, "decisions": []}

    decisions = []
    for line in log_file.read_text().strip().split("\n"):
        if line:
            try:
                decisions.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Most recent first
    decisions.reverse()
    return {"ticker": ticker, "decisions": decisions[:limit]}


@app.get("/api/shadow/score/{ticker}")
async def shadow_score(ticker: str):
    """Score shadow decisions against actual market outcomes.

    For each past decision, fetches the actual price N days later
    and computes whether the signal would have been profitable.
    """
    ticker = ticker.upper()
    log_file = SHADOW_DIR / ticker / "decisions.jsonl"
    if not log_file.exists():
        raise HTTPException(status_code=404, detail=f"No shadow decisions for {ticker}")

    decisions = []
    for line in log_file.read_text().strip().split("\n"):
        if line:
            try:
                decisions.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not decisions:
        raise HTTPException(status_code=404, detail=f"No shadow decisions for {ticker}")

    scored = []
    correct = 0
    total_scored = 0

    for d in decisions:
        signal = d.get("signal", "HOLD")
        entry_price = d.get("price", 0)
        decision_date = d.get("date", "")
        if signal == "HOLD" or entry_price <= 0:
            continue

        # Get price 7 days later for outcome
        try:
            outcome_date = (datetime.strptime(decision_date, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
            outcome_price = _get_price_on_date(ticker, outcome_date)
            if outcome_price is None:
                scored.append({**d, "outcome": "pending", "outcome_price": None, "pnl_pct": None})
                continue

            if signal in ("BUY", "OVERWEIGHT"):
                pnl_pct = (outcome_price - entry_price) / entry_price * 100
            elif signal in ("SHORT", "SELL"):
                pnl_pct = (entry_price - outcome_price) / entry_price * 100
            else:
                pnl_pct = 0.0

            is_correct = pnl_pct > 0
            if is_correct:
                correct += 1
            total_scored += 1

            scored.append({
                **d,
                "outcome": "correct" if is_correct else "incorrect",
                "outcome_price": outcome_price,
                "outcome_date": outcome_date,
                "pnl_pct": round(pnl_pct, 2),
            })
        except Exception:
            scored.append({**d, "outcome": "error", "outcome_price": None, "pnl_pct": None})

    accuracy = (correct / total_scored * 100) if total_scored > 0 else None

    return {
        "ticker": ticker,
        "total_decisions": len(decisions),
        "scored": total_scored,
        "correct": correct,
        "accuracy_pct": round(accuracy, 1) if accuracy is not None else None,
        "decisions": scored[:100],
    }


# ── Quant Pulse — Deterministic Short-Term Signal Engine ──────────────

PULSE_DIR = EVAL_RESULTS_DIR / "pulse"
PULSE_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline locks: cover full read-compute-write cycle per ticker
_pulse_pipeline_locks: Dict[str, asyncio.Lock] = {}

def _get_pulse_lock(ticker: str) -> asyncio.Lock:
    if ticker not in _pulse_pipeline_locks:
        _pulse_pipeline_locks[ticker] = asyncio.Lock()
    return _pulse_pipeline_locks[ticker]

# Pulse scheduler state (independent from the 4H scheduler).
# v3 default: 5-minute cadence (see config/pulse_scoring.yaml).
_PULSE_INTERVAL_MINUTES = 5
_PULSE_STATE_FILE = EVAL_RESULTS_DIR / ".pulse_scheduler_state.json"

def _load_pulse_scheduler_state() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "enabled": False,
        "task": None,
        "last_run": None,
        "last_status": None,
        "interval_minutes": _PULSE_INTERVAL_MINUTES,
        "tickers": ["BTC-USD"],
    }
    try:
        if _PULSE_STATE_FILE.exists():
            saved = json.loads(_PULSE_STATE_FILE.read_text())
            defaults["enabled"] = saved.get("enabled", False)
            defaults["last_run"] = saved.get("last_run")
            defaults["last_status"] = saved.get("last_status")
            defaults["tickers"] = saved.get("tickers", ["BTC-USD"])
            defaults["interval_minutes"] = saved.get("interval_minutes", _PULSE_INTERVAL_MINUTES)
    except Exception:
        pass
    return defaults

def _save_pulse_scheduler_state():
    try:
        _PULSE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PULSE_STATE_FILE.write_text(json.dumps({
            "enabled": _pulse_state.get("enabled", False),
            "last_run": _pulse_state.get("last_run"),
            "last_status": _pulse_state.get("last_status"),
            "tickers": _pulse_state.get("tickers", ["BTC-USD"]),
            "interval_minutes": _pulse_state.get("interval_minutes", _PULSE_INTERVAL_MINUTES),
        }, indent=2))
    except Exception as e:
        logger.warning(f"Failed to persist pulse scheduler state: {e}")

_pulse_state: Dict[str, Any] = _load_pulse_scheduler_state()


def _run_single_pulse(ticker: str, results_dir: str = None) -> dict:
    """Execute a single pulse cycle: build report → score → return v3 entry.

    v3 pipeline:
        1. build_pulse_report (with partial_bar_flags)
        2. load TSMOM cache (refreshed separately on 1h cadence)
        3. detect regime from 1h history (cached)
        4. fetch book imbalance + liq cluster (optional, best-effort)
        5. look up previous signal for persistence multiplier
        6. score_pulse with all v3 inputs

    NOT async — runs synchronously. Locking is done by the caller.
    """
    from tradingagents.agents.quant_pulse_data import build_pulse_report
    from tradingagents.dataflows.hyperliquid_client import HyperliquidClient
    from tradingagents.pulse.config import get_config
    from tradingagents.pulse.tsmom import load_tsmom
    from tradingagents.pulse.regime import detect_regime
    from tradingagents.pulse.support_resistance import compute_support_resistance
    from tradingagents.pulse.pulse_assembly import PulseInputs, score_pulse_from_inputs
    import numpy as _np

    cfg = get_config()
    rd = results_dir or str(EVAL_RESULTS_DIR)
    report = build_pulse_report(ticker, results_dir=rd)
    base_asset = ticker.replace("-USD", "").replace("USDT", "").upper()
    hl = HyperliquidClient()

    # --- TSMOM (best-effort; None if unavailable) ---------------------
    tsmom_direction = None
    tsmom_strength = None
    tsmom_cached = load_tsmom(ticker, rd)
    if tsmom_cached is not None and not tsmom_cached.insufficient_history:
        tsmom_direction = int(tsmom_cached.direction)
        tsmom_strength = float(tsmom_cached.strength)

    # --- Regime (from 1h history; reuse 1h fetch) --------------------
    regime_mode = "mixed"
    regime_snapshot = None
    df_1h = None   # reused below for S/R + z_4h computation
    try:
        # Fetch enough 1h history for regime (500 bars is plenty)
        from datetime import timedelta as _td
        start_dt = datetime.now(timezone.utc) - _td(hours=720)
        df_1h = hl.get_ohlcv(
            base_asset, "1h",
            start=start_dt.strftime("%Y-%m-%d"),
            end=(datetime.now(timezone.utc) + _td(days=1)).strftime("%Y-%m-%d"),
            max_age_override=1800,
        )
        if df_1h is not None and len(df_1h) >= 30:
            regime = detect_regime(df_1h)
            regime_mode = regime.mode
            regime_snapshot = regime.to_dict()
    except Exception as e:
        logger.warning(f"[Pulse] regime detection failed for {ticker}: {e}")

    # --- Book imbalance (best-effort; None if unavailable) -----------
    book_imbalance = None
    try:
        cache_sec = int(cfg.get("confluence", "book_imbalance", "cache_seconds", default=60))
        book_imbalance = hl.compute_book_imbalance(base_asset, levels=1, max_age_override=cache_sec)
    except Exception:
        book_imbalance = None

    # --- Liq cluster (best-effort; may be None if HL doesn't tag liq) -
    liq_score = None
    try:
        window_min = int(cfg.get("confluence", "liquidation", "window_minutes", default=15))
        liq_score = hl.liquidation_cluster_score(
            base_asset, window_minutes=window_min, max_age_override=60,
        )
    except Exception:
        liq_score = None

    # Realized vol windows for liq override (30m recent, 30m prior)
    rv_recent = None
    rv_prior = None
    try:
        candles_1m = report.get("timeframes", {}).get("1m", {})
        # We don't have 1m df in the report — fetch ourselves (cached).
        start_dt = datetime.now(timezone.utc) - timedelta(minutes=70)
        df_1m = hl.get_ohlcv(
            base_asset, "1m",
            start=start_dt.strftime("%Y-%m-%d"),
            end=(datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d"),
            max_age_override=60,
        )
        if df_1m is not None and len(df_1m) >= 60:
            import numpy as _np
            closes = df_1m["close"].astype(float).values[-60:]
            log_rets = _np.diff(_np.log(_np.clip(closes, 1e-12, None)))
            rv_recent = float(_np.std(log_rets[-30:], ddof=1)) if len(log_rets) >= 30 else None
            rv_prior = float(_np.std(log_rets[:30], ddof=1)) if len(log_rets) >= 30 else None
    except Exception:
        pass

    # --- EMA liquidity gate -----------------------------------------
    ema_liquidity_ok = True
    try:
        min_vol = float(cfg.get("confluence", "ema_liquidity_gate", "min_24h_volume_usd", default=5e7))
        day_vol = float(report.get("day_volume") or 0)
        if day_vol > 0:
            ema_liquidity_ok = day_vol >= min_vol
    except Exception:
        pass

    # --- Previous signal for persistence multiplier ------------------
    prev_signal = None
    try:
        last = _read_last_pulse_entry(ticker)
        if last is not None:
            prev_signal = last.get("signal")
    except Exception:
        pass

    # --- Support / Resistance (pivots + L2 book clusters) ------------
    support = None
    resistance = None
    sr_source = "none"
    try:
        # 1h candles were already fetched for regime; reuse via client cache.
        # Fetch 4h candles (cheap, cached) for longer-term pivots.
        atr_1h_for_sr = (report.get("timeframes") or {}).get("1h", {}).get("atr")
        df_4h_sr = hl.get_ohlcv(
            base_asset, "4h",
            start=(datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d"),
            end=(datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d"),
            max_age_override=3600,
        )
        # Small 5m slice for the liquidity-sweep spoof filter
        df_5m_sr = hl.get_ohlcv(
            base_asset, "5m",
            start=(datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%d"),
            end=(datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d"),
            max_age_override=60,
        )
        # L2 snapshot (best-effort)
        try:
            l2_snap = hl.get_l2_snapshot(
                base_asset,
                max_age_override=int(cfg.get("confluence", "book_imbalance", "cache_seconds", default=60)),
            )
        except Exception:
            l2_snap = None

        sr = compute_support_resistance(
            spot_price=report.get("spot_price"),
            df_1h=df_1h,
            df_4h=df_4h_sr,
            atr_1h=atr_1h_for_sr,
            l2_snapshot=l2_snap,
            df_5m=df_5m_sr,
            cluster_atr_mul=float(cfg.get("confluence", "sr_proximity", "cluster_atr_mul", default=0.15)),
            pivot_left=int(cfg.get("confluence", "sr_proximity", "pivot_left", default=3)),
            pivot_right=int(cfg.get("confluence", "sr_proximity", "pivot_right", default=3)),
            band_pct=float(cfg.get("confluence", "sr_proximity", "book_band_pct", default=0.02)),
            min_book_notional_usd=float(cfg.get("confluence", "sr_proximity", "min_book_notional_usd", default=500_000)),
            recency_half_life_hours=float(cfg.get("confluence", "sr_proximity", "recency_half_life_hours", default=24.0)),
        )
        support = sr.support
        resistance = sr.resistance
        sr_source = sr.source
    except Exception as e:
        logger.warning(f"[Pulse] S/R compute failed for {ticker}: {e}")

    # --- 4h return z-score (for parabolic soft-gate) -----------------
    z_4h_return = None
    try:
        if df_1h is not None and len(df_1h) >= 100:
            closes = df_1h["close"].astype(float).values
            # 4h log return = sum of last 4 hourly log-returns
            log_rets_1h = _np.diff(_np.log(_np.clip(closes, 1e-12, None)))
            if len(log_rets_1h) >= 96:
                ret_4h_series = _np.convolve(log_rets_1h, _np.ones(4), mode="valid")
                last = float(ret_4h_series[-1])
                # Reference distribution: last ~90 observations (exclude current)
                ref = ret_4h_series[-91:-1]
                sd = float(_np.std(ref, ddof=1)) if len(ref) > 1 else 0.0
                mu = float(_np.mean(ref)) if len(ref) > 0 else 0.0
                if sd > 1e-12:
                    z_4h_return = (last - mu) / sd
    except Exception:
        z_4h_return = None

    # --- Score via unified PulseInputs -------------------------------
    signal_threshold_cfg = float(
        cfg.get("confluence", "signal_threshold", default=0.22)
    )
    inputs = PulseInputs(
        report=report,
        signal_threshold=signal_threshold_cfg,
        backtest_mode=False,
        tsmom_direction=tsmom_direction,
        tsmom_strength=tsmom_strength,
        regime_mode=regime_mode,
        realized_vol_recent=rv_recent,
        realized_vol_prior=rv_prior,
        liquidation_score=liq_score,
        book_imbalance=book_imbalance,
        prev_signal=prev_signal,
        ema_liquidity_ok=ema_liquidity_ok,
        support=support,
        resistance=resistance,
        sr_source=sr_source,
        z_4h_return=z_4h_return,
        cfg=cfg,
    )
    result = score_pulse_from_inputs(inputs)

    # v3 pulse entry schema ------------------------------------------
    atr_1h = (report.get("timeframes") or {}).get("1h", {}).get("atr")

    def _build_entry(scored: dict, *, config_name: str, ensemble_tick_id: str) -> dict:
        """Stamp a scored result into the on-disk schema.

        Pure: same shape regardless of variant. ``config_name`` /
        ``ensemble_tick_id`` (R.2) are the new fields — everything else
        mirrors the legacy layout so downstream readers don't break.
        """
        return {
            "ts": report.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "engine_version": cfg.engine_version,
            "config_hash": cfg.hash_short(),
            "config_name": config_name,
            "ensemble_tick_id": ensemble_tick_id,
            "signal": scored["signal"],
            "confidence": scored["confidence"],
            "normalized_score": scored["normalized_score"],
            "raw_normalized_score": scored.get("raw_normalized_score"),
            "price": report.get("spot_price"),
            "atr_1h_at_pulse": atr_1h,
            "partial_bar_flags": report.get("partial_bar_flags", {}),
            "stop_loss": scored.get("stop_loss"),
            "take_profit": scored.get("take_profit"),
            "hold_minutes": scored.get("hold_minutes"),
            "timeframe_bias": scored.get("timeframe_bias"),
            "funding_rate": report.get("funding_rate"),
            "premium_pct": report.get("premium_pct"),
            "reasoning": scored.get("reasoning", ""),
            "breakdown": scored.get("breakdown", {}),
            "volatility_flag": scored.get("volatility_flag", False),
            "signal_threshold": scored.get("signal_threshold"),
            "persistence_mul": scored.get("persistence_mul"),
            "override_reason": scored.get("override_reason"),
            "tsmom_direction": tsmom_direction,
            "tsmom_strength": tsmom_strength,
            "tsmom_gated_out": scored.get("tsmom_gated_out", False),
            "tsmom_gate_reason": scored.get("tsmom_gate_reason"),
            "tsmom_gate_mode": scored.get("tsmom_gate_mode"),
            "regime_mode": regime_mode,
            "regime_snapshot": regime_snapshot,
            "book_imbalance": book_imbalance,
            "liquidation_score": liq_score,
            "day_volume_usd": report.get("day_volume"),
            "support": support,
            "resistance": resistance,
            "sr_source": sr_source,
            "sr_near_side": scored.get("sr_near_side"),
            "z_4h_return": z_4h_return,
            "scored": False,
        }

    # ── R.2 Ensemble scatter ─────────────────────────────────────────
    # Score the same PulseInputs under every non-baseline variant and
    # persist to per-config streams. Baseline result we already have
    # above — keep it as the legacy return value so every existing
    # consumer is unaffected. Failures in a variant are logged by
    # score_ensemble and silently dropped so one bad overlay can't
    # take the live loop down.
    from tradingagents.pulse.ensemble import (
        generate_ensemble_tick_id, list_variant_names, score_ensemble,
    )
    tick_id = generate_ensemble_tick_id()
    try:
        variant_names = [n for n in list_variant_names() if n != "baseline"]
        variant_results = score_ensemble(
            inputs,
            variant_names=variant_names,
            active_regime=cfg.active_regime,
            venue=cfg.venue,
            data_source=cfg.data_source,
            ensemble_tick_id=tick_id,
        )
        for vname, vres in variant_results.items():
            try:
                ventry = _build_entry(vres, config_name=vname, ensemble_tick_id=tick_id)
                _append_variant_pulse(ticker, vname, ventry)
            except Exception as e:
                logger.warning(f"[Ensemble] write failed for {ticker}/{vname}: {e}")
    except Exception as e:
        logger.warning(f"[Ensemble] scatter failed for {ticker}: {e}")

    # Baseline entry — stamped with the same tick_id so the verifier
    # can join variants to the baseline tick even though the legacy
    # pulse.jsonl path has no ``config_name`` field historically.
    pulse_entry = _build_entry(result, config_name="baseline", ensemble_tick_id=tick_id)
    # Also mirror the baseline entry to configs/baseline/pulse.jsonl so
    # the ensemble verifier / metrics aggregator can read all variants
    # uniformly without special-casing the legacy path.
    try:
        _append_variant_pulse(ticker, "baseline", pulse_entry)
    except Exception as e:
        logger.warning(f"[Ensemble] baseline mirror write failed for {ticker}: {e}")
    return pulse_entry


# ── R.5 Champion indirection ─────────────────────────────────────────
#
# Every downstream reader of pulse.jsonl MUST route through
# ``_champion_pulse_path`` so that when an operator promotes a variant
# (via POST /api/pulse/ensemble/champion), the "live" pulse stream
# surfaced to UI + risk consumers swaps atomically. Without this
# indirection the ensemble is decorative (the debate's blocking point
# against v1 of the plan).
#
# Default champion = "baseline"; a missing champion.json falls through
# to the legacy pulse.jsonl so upgrades are backward-compatible.

def _read_champion_name(ticker: str) -> str:
    """Return the currently-championed config name for ``ticker``.

    Reads ``PULSE_DIR/<ticker>/champion.json`` — a small JSON doc of
    shape ``{"config": "<name>", "set_at": "<iso>"}``. Defaults to
    ``"baseline"`` when the file is absent or malformed.
    """
    path = PULSE_DIR / ticker / "champion.json"
    if not path.exists():
        return "baseline"
    try:
        data = json.loads(path.read_text())
        name = str(data.get("config") or "baseline")
        return name or "baseline"
    except Exception:
        return "baseline"


def _write_champion_name(ticker: str, name: str) -> dict:
    """Atomic write of ``champion.json`` — returns the new doc."""
    d = PULSE_DIR / ticker
    d.mkdir(parents=True, exist_ok=True)
    doc = {"config": name, "set_at": datetime.now(timezone.utc).isoformat()}
    tmp = d / "champion.json.tmp"
    tmp.write_text(json.dumps(doc))
    tmp.replace(d / "champion.json")
    return doc


def _champion_pulse_path(ticker: str) -> Path:
    """Return the JSONL path that should be treated as "the" pulse
    stream for this ticker. Falls back to the legacy single-stream
    path if the champion's variant directory doesn't yet exist (i.e.
    we haven't written any pulses for that variant yet)."""
    champ = _read_champion_name(ticker)
    variant_path = PULSE_DIR / ticker / "configs" / champ / "pulse.jsonl"
    if variant_path.exists():
        return variant_path
    return PULSE_DIR / ticker / "pulse.jsonl"


def _read_last_pulse_entry(ticker: str) -> Optional[dict]:
    """Return the most recent pulse entry (or None). Tolerates corrupt lines."""
    pulse_path = _champion_pulse_path(ticker)
    if not pulse_path.exists():
        return None
    try:
        last = None
        with open(pulse_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
        return last
    except Exception:
        return None


def _append_variant_pulse(ticker: str, variant_name: str, entry: dict) -> None:
    """Append a variant's pulse entry to its per-config JSONL stream.

    Layout: ``<EVAL_RESULTS>/pulse/<TICKER>/configs/<variant>/pulse.jsonl``.
    Mirrors ``_append_pulse`` semantics (small-write atomic append; large
    entries go through tmp+rename). Kept structurally close to the
    legacy writer so any future atomicity fix to one path applies to
    both.
    """
    if not variant_name:
        raise ValueError("variant_name is required")
    vdir = PULSE_DIR / ticker / "configs" / variant_name
    vdir.mkdir(parents=True, exist_ok=True)
    vpath = vdir / "pulse.jsonl"
    line = json.dumps(entry, default=str) + "\n"
    if len(line.encode()) >= 4096:
        existing = vpath.read_text() if vpath.exists() else ""
        final_tmp = vpath.with_suffix(".jsonl.tmp")
        final_tmp.write_text(existing + line)
        os.replace(final_tmp, vpath)
        return
    with open(vpath, "a") as f:
        f.write(line)


def _append_pulse(ticker: str, entry: dict):
    """Append a pulse entry to the JSONL file.

    Append is naturally atomic on POSIX for small writes < PIPE_BUF (4 KB).
    Pulse entries are ~1-2 KB so safe. For multi-line rewrites we use
    _score_pending_pulses's atomic tmp+rename.

    Free-tier survival: if GITHUB_TOKEN + PULSE_GIST_ID are set, every append
    triggers an async push to a GitHub Gist (see `tradingagents.pulse.gist_sync`).
    This is the only way pulse history survives Render free-tier filesystem wipes.
    """
    pulse_dir = PULSE_DIR / ticker
    pulse_dir.mkdir(parents=True, exist_ok=True)
    pulse_path = pulse_dir / "pulse.jsonl"
    line = json.dumps(entry, default=str) + "\n"
    if len(line.encode()) >= 4096:
        # Unlikely but possible for huge reasoning; use lock-like approach
        tmp = pulse_path.with_suffix(".jsonl.append.tmp")
        with open(tmp, "w") as f:
            f.write(line)
        # Read-modify-write path as fallback
        existing = pulse_path.read_text() if pulse_path.exists() else ""
        tmp.unlink()
        final_tmp = pulse_path.with_suffix(".jsonl.tmp")
        final_tmp.write_text(existing + line)
        os.replace(final_tmp, pulse_path)
        return
    with open(pulse_path, "a") as f:
        f.write(line)

    # Free-tier persistence: best-effort push to GitHub Gist.
    # Fire-and-forget in a thread so we never block the caller.
    try:
        from tradingagents.pulse import gist_sync
        if gist_sync.is_enabled():
            threading.Thread(
                target=gist_sync.push_ticker,
                args=(PULSE_DIR, ticker),
                daemon=True,
            ).start()
    except Exception:
        pass


def _read_pulses(ticker: str, limit: int = 50) -> List[dict]:
    """Read the last N pulses from JSONL. Routes through the champion
    indirection so promoting a variant immediately swaps what the UI +
    ``/api/pulse/latest`` surface (R.5)."""
    pulse_path = _champion_pulse_path(ticker)
    if not pulse_path.exists():
        return []
    entries = []
    with open(pulse_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries[-limit:]


def _read_pulse_at(ticker: str, ts: str) -> Optional[dict]:
    """Return the pulse entry with exact or nearest-match timestamp.

    Match strategy:
      1. Exact string match on ``ts``.
      2. Nearest by UTC timestamp within 60 seconds.
    Returns None if no match. R.5: routes via ``_champion_pulse_path``.
    """
    pulse_path = _champion_pulse_path(ticker)
    if not pulse_path.exists():
        return None
    target_dt: Optional[datetime] = None
    try:
        target_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=timezone.utc)
    except Exception:
        target_dt = None

    best: Optional[dict] = None
    best_delta: float = float("inf")
    with open(pulse_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_ts = entry.get("ts", "")
            if entry_ts == ts:
                return entry
            if target_dt is not None:
                try:
                    e_dt = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
                    if e_dt.tzinfo is None:
                        e_dt = e_dt.replace(tzinfo=timezone.utc)
                    delta = abs((e_dt - target_dt).total_seconds())
                    if delta < best_delta:
                        best_delta = delta
                        best = entry
                except Exception:
                    continue
    if best is not None and best_delta <= 60.0:
        return best
    return None


async def _tick_ticker(ticker: str, skip_count: dict) -> None:
    """Run one pulse for one ticker with skip-on-overlap lock semantics."""
    lock = _get_pulse_lock(ticker)
    if lock.locked():
        skip_count["skipped"] = skip_count.get("skipped", 0) + 1
        logger.warning(
            f"[Pulse] Skipped {ticker} — previous run still active"
        )
        return
    async with lock:
        try:
            entry = await asyncio.get_event_loop().run_in_executor(
                None, _run_single_pulse, ticker
            )
            _append_pulse(ticker, entry)
            logger.info(
                f"[Pulse] {ticker}: {entry['signal']} "
                f"(conf={entry['confidence']:.2f}) "
                f"TSMOM={entry.get('tsmom_direction')} "
                f"regime={entry.get('regime_mode')}"
            )
            # Fire alerts (best-effort, non-blocking from the run-loop's perspective)
            try:
                from tradingagents.pulse.alerts import dispatch_alert_if_eligible
                await asyncio.get_event_loop().run_in_executor(
                    None, dispatch_alert_if_eligible, entry, ticker
                )
            except Exception as alert_exc:
                logger.warning(f"[Pulse.alerts] {ticker}: {alert_exc}")
        except Exception as e:
            logger.error(f"[Pulse] Error for {ticker}: {e}")


async def _pulse_scheduler():
    """Run pulse analysis at configured interval for all pulse tickers.

    v3: parallel per-ticker via asyncio.gather, per-ticker jitter to spread
    API load, skip-on-overlap so slow tickers never cause pile-up.
    """
    logger.info("[Pulse] Scheduler started (v3: parallel + skip-on-overlap)")
    skip_count = {"skipped": 0}
    while _pulse_state["enabled"]:
        try:
            interval = _pulse_state.get("interval_minutes", _PULSE_INTERVAL_MINUTES)
            tickers = _pulse_state.get("tickers", ["BTC-USD"])

            async def _jittered(tkr: str) -> None:
                offset = (hash(tkr) & 0xFFFF) % max(1, min(15, int(interval * 60 / 10)))
                if offset > 0:
                    await asyncio.sleep(offset)
                await _tick_ticker(tkr, skip_count)

            await asyncio.gather(
                *(_jittered(t) for t in tickers),
                return_exceptions=True,
            )

            _pulse_state["last_run"] = datetime.now(timezone.utc).isoformat()
            _pulse_state["last_status"] = (
                f"ok (skipped={skip_count['skipped']})"
                if skip_count["skipped"] else "ok"
            )
            _save_pulse_scheduler_state()

            await asyncio.sleep(interval * 60)

        except asyncio.CancelledError:
            logger.info("[Pulse] Scheduler cancelled")
            break
        except Exception as e:
            logger.error(f"[Pulse] Scheduler error: {e}")
            _pulse_state["last_status"] = f"error: {e}"
            await asyncio.sleep(60)

    logger.info("[Pulse] Scheduler stopped")


async def _tsmom_refresh_loop():
    """Refresh TSMOM cache every hour for all configured tickers.

    Separated from the 5-min pulse scheduler: TSMOM only needs 1h cadence.
    """
    from tradingagents.pulse.tsmom import refresh_tsmom
    from tradingagents.pulse.config import get_config as _get_cfg
    logger.info("[TSMOM] Refresh loop started")
    while _pulse_state["enabled"]:
        try:
            cfg = _get_cfg()
            universe = cfg.get("tsmom", "universe", default=[]) or []
            lookbacks = cfg.get("tsmom", "lookbacks_hours", default=[504, 1512, 6048])
            target_vol = float(cfg.get("tsmom", "target_annualized_vol", default=0.20))

            for ticker in universe:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        refresh_tsmom,
                        ticker, None, tuple(lookbacks), target_vol, 500,
                        str(EVAL_RESULTS_DIR),
                    )
                except Exception as e:
                    logger.warning(f"[TSMOM] refresh failed for {ticker}: {e}")

            rebalance_min = int(cfg.get("tsmom", "rebalance_minutes", default=60))
            await asyncio.sleep(rebalance_min * 60)
        except asyncio.CancelledError:
            logger.info("[TSMOM] Refresh loop cancelled")
            break
        except Exception as e:
            logger.error(f"[TSMOM] Loop error: {e}")
            await asyncio.sleep(120)


async def _ensemble_verifier_loop():
    """Run the pulse verifier every 5 minutes for all configured tickers.

    The verifier is CPU-cheap — it only does 1m-OHLC lookups and
    arithmetic — so we gather across tickers without the
    skip-on-overlap gymnastics the pulse scheduler needs.
    """
    from scripts.pulse_verifier import process_ticker
    from tradingagents.dataflows.hyperliquid_client import HyperliquidClient

    logger.info("[Verifier] Loop started")
    hl = None
    while _pulse_state.get("enabled"):
        try:
            if hl is None:
                try:
                    hl = HyperliquidClient()
                except Exception as e:
                    logger.warning(f"[Verifier] HL client init failed: {e}")
            tickers = _pulse_state.get("tickers", ["BTC-USD"])
            loop = asyncio.get_event_loop()
            for t in tickers:
                try:
                    outcomes = await loop.run_in_executor(
                        None,
                        lambda tkr=t: process_ticker(
                            tkr, pulse_dir=PULSE_DIR, hl_client=hl,
                        ),
                    )
                    if outcomes:
                        logger.info(f"[Verifier] {t}: resolved {len(outcomes)} outcome(s)")
                        # R.4 — refresh per-config metrics.json so the
                        # UI + champion selector see fresh aggregates.
                        try:
                            from tradingagents.pulse.ensemble_metrics import refresh_all
                            await loop.run_in_executor(
                                None,
                                lambda tkr=t: refresh_all(tkr, pulse_dir=PULSE_DIR),
                            )
                        except Exception as e:
                            logger.warning(f"[Metrics] refresh failed for {t}: {e}")
                except Exception as e:
                    logger.warning(f"[Verifier] {t}: {e}")
        except Exception as e:
            logger.error(f"[Verifier] Loop error: {e}")
        await asyncio.sleep(300)


# ── Pulse API Endpoints ──────────────────────────────────────────────

@app.get("/api/pulse/{ticker}")
async def get_pulses(ticker: str, limit: int = 50):
    """Return the last N pulse signals for a ticker."""
    pulses = _read_pulses(ticker.upper(), limit)
    return {"ticker": ticker.upper(), "pulses": pulses, "count": len(pulses)}


@app.get("/api/pulse/latest/{ticker}")
async def get_latest_pulse(ticker: str):
    """Return the most recent pulse signal."""
    pulses = _read_pulses(ticker.upper(), 1)
    if not pulses:
        return {"ticker": ticker.upper(), "pulse": None}
    return {"ticker": ticker.upper(), "pulse": pulses[-1]}


# ── Pulse Explain Chart ──────────────────────────────────────────────

# Window sizes (bars) per TF for the explain endpoint. Centered on signal ts
# with the majority BEFORE the entry so patterns have context; small tail
# AFTER so state-FSM can see breakouts.
_EXPLAIN_WINDOWS = {
    "5m":  {"before": 36, "after": 12, "interval_sec": 300},
    "15m": {"before": 36, "after": 12, "interval_sec": 900},
    "1h":  {"before": 54, "after": 18, "interval_sec": 3600},
    "4h":  {"before": 30, "after": 10, "interval_sec": 14400},
}


def _floor_ts_5min(ts: str) -> int:
    """Return ts floored to 5-min boundary (unix seconds). Deterministic cache key."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() // 300 * 300)
    except Exception:
        return 0


def _fetch_explain_candles(ticker: str, ts: str) -> Dict[str, Any]:
    """Fetch candle windows for all TFs around the given signal ts.

    Returns dict of TF → list[dict{ts,o,h,l,c,v}] plus raw DataFrames (not
    serialized) for detector use.
    """
    from tradingagents.dataflows.hyperliquid_client import HyperliquidClient
    base_asset = ticker.replace("-USD", "").replace("USDT", "").upper()
    hl = HyperliquidClient()

    try:
        target_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=timezone.utc)
    except Exception:
        target_dt = datetime.now(timezone.utc)

    candles_serial: Dict[str, List[dict]] = {}
    candles_df: Dict[str, Any] = {}

    for tf, win in _EXPLAIN_WINDOWS.items():
        interval_sec = win["interval_sec"]
        start_dt = target_dt - timedelta(seconds=interval_sec * win["before"])
        end_dt = target_dt + timedelta(seconds=interval_sec * win["after"])
        start_str = start_dt.strftime("%Y-%m-%d")
        # end date inclusive, add 1 day padding to ensure tail bars present
        end_str = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            df = hl.get_ohlcv(base_asset, interval=tf, start=start_str, end=end_str)
        except Exception as e:
            logger.warning(f"[explain] fetch failed {ticker} {tf}: {e}")
            continue
        if df is None or df.empty:
            continue

        # Trim to the requested window by timestamp
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
        df = df.sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            continue

        candles_df[tf] = df
        candles_serial[tf] = [
            {
                "ts": int(row.timestamp.timestamp()),
                "o": float(row.open),
                "h": float(row.high),
                "l": float(row.low),
                "c": float(row.close),
                "v": float(row.volume),
            }
            for row in df.itertuples(index=False)
        ]

    return {"serial": candles_serial, "df": candles_df}


def _pulse_attribution_for(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 2 Commit O — thin wrapper around attribution module.

    Kept as a module-level helper so the explain endpoint stays readable
    and the attribution lookup can be patched in tests.
    """
    from tradingagents.backtesting.attribution import per_decision_attribution
    try:
        return per_decision_attribution(entry)
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("attribution computation failed: %s", e)
        return {"top_positive": [], "top_negative": [], "persistence_mul": None,
                "total_abs_contribution": 0.0}


@app.get("/api/pulse/explain/{ticker}/{ts}")
async def get_pulse_explain(ticker: str, ts: str):
    """Return chart-ready explain payload for a single pulse signal.

    Includes candles (5m/15m/1h/4h), pulse entry data (SL/TP/S/R/TSMOM),
    and detected chart patterns with full geometry. Display-only — patterns
    do NOT feed back into signal logic.
    """
    ticker = ticker.upper()
    # URL-decode the ts path param (frontend double-encodes ':' etc.)
    try:
        from urllib.parse import unquote
        ts = unquote(ts)
    except Exception:
        pass

    entry = _read_pulse_at(ticker, ts)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No pulse entry for {ticker} at {ts}",
        )

    # Fetch candle windows
    fetched = await asyncio.get_event_loop().run_in_executor(
        None, _fetch_explain_candles, ticker, entry.get("ts", ts)
    )
    candles_serial: Dict[str, List[dict]] = fetched["serial"]
    candles_df: Dict[str, Any] = fetched["df"]

    # Detect patterns
    from tradingagents.patterns import detect_all
    matches, detector_errors = await asyncio.get_event_loop().run_in_executor(
        None, detect_all, candles_df, ("1h", "4h")
    )

    # Apply regime alignment (WCT request)
    liq_score = float(entry.get("liquidation_score") or 0.0)
    for m in matches:
        if liq_score > 0.5:
            if (m.bias == "bearish" and entry.get("signal") in ("SHORT", "SELL")) or \
               (m.bias == "bullish" and entry.get("signal") in ("BUY", "COVER")):
                m.regime_aligned = True

    chart_patterns = [m.to_dict() for m in matches]

    # Candlestick patterns from persisted indicator_detail (if present)
    candlestick_patterns: List[dict] = []
    for tf, detail in (entry.get("indicator_detail") or {}).items():
        for patt in (detail.get("patterns") or []):
            candlestick_patterns.append({"tf": tf, "name": patt})

    # Breakdown top-3 by absolute weight
    breakdown = entry.get("breakdown") or {}
    top3 = sorted(
        [(k, float(v)) for k, v in breakdown.items() if isinstance(v, (int, float))],
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:3]

    return {
        "ticker": ticker,
        "entry": {
            "ts": entry.get("ts"),
            "price": entry.get("price"),
            "signal": entry.get("signal"),
            "confidence": entry.get("confidence"),
            "normalized_score": entry.get("normalized_score"),
        },
        "levels": {
            "stop_loss": entry.get("stop_loss"),
            "take_profit": entry.get("take_profit"),
            "support": entry.get("support"),
            "resistance": entry.get("resistance"),
            "sr_source": entry.get("sr_source"),
            "sr_near_side": entry.get("sr_near_side"),
        },
        "tsmom": {
            "direction": entry.get("tsmom_direction"),
            "strength": entry.get("tsmom_strength"),
            "gated_out": entry.get("tsmom_gated_out"),
        },
        "timeframe_bias": entry.get("timeframe_bias"),
        "regime_mode": entry.get("regime_mode"),
        "breakdown_top3": [{"key": k, "weight": w} for k, w in top3],
        "breakdown": breakdown,
        # Stage 2 Commit O — per-decision feature attribution.
        "attribution": _pulse_attribution_for(entry),
        "candles": candles_serial,
        "chart_patterns": chart_patterns,
        "candlestick_patterns": candlestick_patterns,
        "reasoning_prose": entry.get("reasoning", ""),
        "detector_errors": detector_errors,
        "pattern_detection_degraded": len(detector_errors) >= 3,
    }


# ── R.5 Ensemble API endpoints ───────────────────────────────────────

@app.get("/api/pulse/ensemble/{ticker}")
async def get_ensemble_latest(ticker: str):
    """Latest pulse from every variant stream, plus agreement score.

    Agreement score = fraction of variants that agreed with the
    champion's signal on the most recent shared ``ensemble_tick_id``.
    If variants fired at different tick ids (shouldn't happen in
    production but can on startup) we report the champion's id only.
    """
    ticker = ticker.upper()
    from tradingagents.pulse.ensemble_metrics import list_configs
    configs = list_configs(ticker, pulse_dir=PULSE_DIR) or ["baseline"]
    latest: Dict[str, Optional[dict]] = {}
    for cfg in configs:
        p = PULSE_DIR / ticker / "configs" / cfg / "pulse.jsonl"
        if not p.exists():
            latest[cfg] = None
            continue
        try:
            last = None
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        continue
            latest[cfg] = last
        except Exception as e:
            logger.warning(f"[Ensemble] read failed {ticker}/{cfg}: {e}")
            latest[cfg] = None

    champion = _read_champion_name(ticker)
    champ_entry = latest.get(champion) or {}
    champ_tick = champ_entry.get("ensemble_tick_id")
    champ_signal = champ_entry.get("signal")
    # Agreement: fraction of non-null variants whose most-recent entry
    # both shares the champion's tick_id AND emitted the same signal.
    agreeing = 0
    compared = 0
    for cfg, entry in latest.items():
        if entry is None:
            continue
        compared += 1
        if entry.get("ensemble_tick_id") == champ_tick and entry.get("signal") == champ_signal:
            agreeing += 1
    return {
        "ticker": ticker,
        "champion": champion,
        "champion_tick_id": champ_tick,
        "champion_signal": champ_signal,
        "agreement_score": round(agreeing / compared, 4) if compared else None,
        "n_variants": compared,
        "variants": latest,
    }


@app.get("/api/pulse/ensemble/{ticker}/metrics")
async def get_ensemble_metrics(ticker: str):
    """Return the per-config rolling metrics.json for every variant."""
    ticker = ticker.upper()
    from tradingagents.pulse.ensemble_metrics import list_configs
    out: Dict[str, Any] = {}
    for cfg in list_configs(ticker, pulse_dir=PULSE_DIR):
        mpath = PULSE_DIR / ticker / "configs" / cfg / "metrics.json"
        if mpath.exists():
            try:
                out[cfg] = json.loads(mpath.read_text())
            except Exception as e:
                logger.warning(f"[Ensemble] metrics read failed {ticker}/{cfg}: {e}")
                out[cfg] = None
        else:
            out[cfg] = None
    return {"ticker": ticker, "champion": _read_champion_name(ticker), "metrics": out}


@app.get("/api/pulse/ensemble/{ticker}/disagreements")
async def get_ensemble_disagreements(ticker: str, limit: int = 50):
    """Ticks where configs produced ≥2 distinct signals.

    Reads the most recent pulse from every variant, groups by
    ``ensemble_tick_id``, and returns those groups that have more
    than one unique signal value. High-signal diagnostic for tuning.
    """
    ticker = ticker.upper()
    from tradingagents.pulse.ensemble_metrics import list_configs
    by_tick: Dict[str, Dict[str, dict]] = {}
    for cfg in list_configs(ticker, pulse_dir=PULSE_DIR):
        p = PULSE_DIR / ticker / "configs" / cfg / "pulse.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines()[-2000:]:  # cap work
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = entry.get("ensemble_tick_id")
            if not tid:
                continue
            by_tick.setdefault(tid, {})[cfg] = entry

    disagreements: List[dict] = []
    for tid, group in by_tick.items():
        signals = {e.get("signal") for e in group.values()}
        if len(signals) >= 2:
            disagreements.append({
                "ensemble_tick_id": tid,
                "ts": next(iter(group.values())).get("ts"),
                "signals": {cfg: e.get("signal") for cfg, e in group.items()},
                "confidences": {cfg: e.get("confidence") for cfg, e in group.items()},
            })
    # Newest first.
    disagreements.sort(key=lambda d: d.get("ts") or "", reverse=True)
    return {
        "ticker": ticker,
        "count": len(disagreements),
        "disagreements": disagreements[:limit],
    }


class _ChampionRequest(BaseModel):
    config: str


@app.post("/api/pulse/ensemble/{ticker}/champion")
async def set_ensemble_champion(ticker: str, req: _ChampionRequest):
    """Propose-only champion swap. Updates ``champion.json`` on disk —
    this only swaps which variant's stream the live pulse API surfaces.
    Actual order routing is still gated by ``EXECUTE_TRADES`` (Stage 2
    Commit M.2)."""
    ticker = ticker.upper()
    from tradingagents.pulse.config import list_variant_names
    valid = set(list_variant_names())
    if req.config not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown variant {req.config!r}; valid: {sorted(valid)}",
        )
    doc = _write_champion_name(ticker, req.config)
    return {"ticker": ticker, **doc}


@app.get("/api/pulse/ensemble/{ticker}/champion")
async def get_ensemble_champion(ticker: str):
    ticker = ticker.upper()
    return {"ticker": ticker, "config": _read_champion_name(ticker)}


@app.post("/api/pulse/run/{ticker}")
async def run_pulse(ticker: str):
    """Manually trigger a single pulse analysis."""
    ticker = ticker.upper()
    lock = _get_pulse_lock(ticker)
    async with lock:
        try:
            entry = await asyncio.get_event_loop().run_in_executor(
                None, _run_single_pulse, ticker
            )
            _append_pulse(ticker, entry)
            return {"ticker": ticker, "pulse": entry}
        except Exception as e:
            logger.error(f"[Pulse] Manual run failed for {ticker}: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pulse/scorecard/{ticker}")
async def get_pulse_scorecard(ticker: str, engine_version: Optional[str] = None):
    """Return forward-return hit rates + 4-fill-model summaries for scored pulses.

    Query args:
        engine_version: if provided, only pulses tagged with that engine_version
            are included. v3 entries have "engine_version"; legacy entries do
            not. By default we include everything but break out the counts.
    """
    ticker = ticker.upper()
    pulse_path = _champion_pulse_path(ticker)  # R.5 indirection
    if not pulse_path.exists():
        return {"ticker": ticker, "total": 0, "scored": 0, "hit_rates": {}}

    all_pulses = _read_pulses(ticker, limit=10000)
    if engine_version:
        all_pulses = [
            p for p in all_pulses if p.get("engine_version") == engine_version
        ]
    scored = [p for p in all_pulses if p.get("scored")]
    total = len(all_pulses)
    n_scored = len(scored)

    if n_scored == 0:
        return {
            "ticker": ticker,
            "total": total,
            "scored": 0,
            "hit_rates": {},
            "fill_summary": {},
            "engine_versions": sorted({p.get("engine_version") for p in all_pulses if p.get("engine_version")}),
        }

    # Hit rates per horizon
    hit_rates: dict = {}
    for horizon in ["+5m", "+15m", "+1h"]:
        key = f"hit_{horizon}"
        hits = sum(1 for p in scored if p.get(key))
        hit_rates[horizon] = {
            "overall": round(hits / n_scored, 4) if n_scored > 0 else 0,
        }
        for sig in ["BUY", "SHORT"]:
            sig_scored = [p for p in scored if p.get("signal") == sig]
            sig_hits = sum(1 for p in sig_scored if p.get(key))
            hit_rates[horizon][sig] = (
                round(sig_hits / len(sig_scored), 4)
                if sig_scored else 0
            )

    # Fill-model summary: mean net return per model per horizon (v3 entries only)
    fill_summary: dict = {}
    for horizon in ["+5m", "+15m", "+1h"]:
        fills_key = f"fills_{horizon}"
        by_model: dict = {"best": [], "realistic": [], "maker_rejected": [], "maker_adverse": []}
        for p in scored:
            fills = p.get(fills_key) or {}
            if not isinstance(fills, dict):
                continue
            for m, vals in fills.items():
                if not isinstance(vals, dict) or m not in by_model:
                    continue
                nr = vals.get("net_return")
                if isinstance(nr, (int, float)):
                    by_model[m].append(float(nr))
        fill_summary[horizon] = {
            m: (
                {
                    "count": len(vs),
                    "mean_net_bps": round(sum(vs) / len(vs) * 10_000, 2) if vs else 0.0,
                    "win_rate": round(sum(1 for v in vs if v > 0) / len(vs), 4) if vs else 0.0,
                }
                if vs else {"count": 0, "mean_net_bps": 0.0, "win_rate": 0.0}
            )
            for m, vs in by_model.items()
        }

    return {
        "ticker": ticker,
        "total": total,
        "scored": n_scored,
        "hit_rates": hit_rates,
        "fill_summary": fill_summary,
        "engine_versions": sorted({p.get("engine_version") for p in all_pulses if p.get("engine_version")}),
    }


@app.get("/api/pulse/regime/current/{ticker}")
async def get_current_directional_regime(ticker: str):
    """Return the latest directional regime classification (Stage 2 G).

    Fetches ~120 days of daily OHLC via yfinance and runs the directional
    classifier with the standard latency budget. Log-only — does NOT
    switch the active regime profile; the UI uses this purely for the
    "Currently detected regime" card and the mismatch callout.
    """
    from tradingagents.pulse.regime_directional import classify_directional

    ticker = ticker.upper()
    try:
        df = yf.Ticker(ticker).history(period="180d", interval="1d")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OHLC fetch failed: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No OHLC data for {ticker}")

    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    df = df[["open", "high", "low", "close"]].dropna()

    result = classify_directional(df, ticker=ticker, log=True)
    return {"ticker": ticker, **result.to_dict()}


@app.get("/api/pulse/scheduler/status")
async def pulse_scheduler_status():
    """Return pulse scheduler state."""
    return {
        "enabled": _pulse_state.get("enabled", False),
        "tickers": _pulse_state.get("tickers", ["BTC-USD"]),
        "interval_minutes": _pulse_state.get("interval_minutes", _PULSE_INTERVAL_MINUTES),
        "last_run": _pulse_state.get("last_run"),
        "last_status": _pulse_state.get("last_status"),
    }


@app.post("/api/pulse/scheduler/toggle")
async def toggle_pulse_scheduler():
    """Enable or disable the pulse scheduler."""
    currently_enabled = _pulse_state.get("enabled", False)

    if currently_enabled:
        _pulse_state["enabled"] = False
        task = _pulse_state.get("task")
        if task:
            task.cancel()
        _pulse_state["task"] = None
        tsmom_task = _pulse_state.get("tsmom_task")
        if tsmom_task:
            tsmom_task.cancel()
        _pulse_state["tsmom_task"] = None
        _save_pulse_scheduler_state()
        return {"enabled": False, "message": "Pulse scheduler disabled"}
    else:
        _pulse_state["enabled"] = True
        _pulse_state["task"] = asyncio.create_task(_pulse_scheduler())
        _pulse_state["tsmom_task"] = asyncio.create_task(_tsmom_refresh_loop())
        _save_pulse_scheduler_state()
        return {"enabled": True, "message": "Pulse scheduler enabled (v3)"}


# ── Pulse Scorecard Scoring Job ───────────────────────────────────────

async def _score_pending_pulses():
    """Score unscored pulses older than 1h against actual forward returns.

    Uses candle OPEN at target timestamp (not close) per debate finding.
    """
    from tradingagents.dataflows.hyperliquid_client import HyperliquidClient, PULSE_CACHE_TTL

    hl = HyperliquidClient()
    now_utc = datetime.now(timezone.utc)
    one_hour_ago = now_utc - timedelta(hours=1)

    for ticker_dir in PULSE_DIR.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        pulse_path = ticker_dir / "pulse.jsonl"
        if not pulse_path.exists():
            continue

        base_asset = ticker.replace("-USD", "").replace("USDT", "").upper()
        all_entries = []
        modified = False

        with open(pulse_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        all_entries.append({"_raw": line})

        for entry in all_entries:
            if entry.get("scored") or entry.get("_raw"):
                continue
            if entry.get("signal") == "NEUTRAL":
                entry["scored"] = True
                modified = True
                continue

            ts_str = entry.get("ts")
            if not ts_str:
                continue
            try:
                pulse_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue

            if pulse_ts > one_hour_ago:
                continue  # too recent to score

            entry_price = entry.get("price")
            if not entry_price or entry_price <= 0:
                continue

            signal = entry.get("signal")
            direction = 1 if signal == "BUY" else -1

            # Fetch 1m candles covering [pulse_ts - 1m, pulse_ts + 65m]
            try:
                start_str = (pulse_ts - timedelta(minutes=2)).strftime("%Y-%m-%d")
                end_str = (pulse_ts + timedelta(minutes=70)).strftime("%Y-%m-%d")
                candles = hl.get_ohlcv(
                    base_asset, "1m", start=start_str, end=end_str,
                    max_age_override=3600,
                )
            except Exception:
                continue

            if candles.empty:
                continue

            # v3 scoring: ATR-sqrt-time thresholds + 4 fill models.
            from tradingagents.pulse.config import get_config as _get_cfg
            from tradingagents.pulse.fills import simple_fill_returns
            cfg = _get_cfg()

            horizons_min = cfg.get("forward_return", "horizons_minutes", default=[5, 15, 60]) or [5, 15, 60]
            atr_mul = float(cfg.get("forward_return", "atr_multiplier", default=0.5))
            exec_cost_bps = float(cfg.get("forward_return", "exec_cost_bps", default=5))
            fallback_bps = cfg.get("forward_return", "fallback_fixed_bps", default=[5, 10, 15]) or [5, 10, 15]
            atr_at_pulse = entry.get("atr_1h_at_pulse")
            threshold_source = "atr_sqrt_time" if atr_at_pulse else "fallback"
            entry["threshold_source"] = threshold_source

            for idx, h_min in enumerate(horizons_min):
                horizon = f"+{h_min}m" if h_min != 60 else "+1h"
                target_ts = pulse_ts + timedelta(minutes=int(h_min))
                mask = candles["timestamp"] >= pd.Timestamp(target_ts.replace(tzinfo=None))
                if not mask.any():
                    continue
                target_candle = candles[mask].iloc[0]
                fwd_price = float(target_candle["open"])

                # Raw return (direction-signed)
                raw_return = (fwd_price - entry_price) / entry_price * direction

                # Threshold: ATR-sqrt-time if available, else fallback fixed bps
                if atr_at_pulse and atr_at_pulse > 0 and entry_price > 0:
                    thr = atr_mul * float(atr_at_pulse) * math.sqrt(h_min / 60.0) / entry_price
                else:
                    fb_bps = fallback_bps[idx] if idx < len(fallback_bps) else fallback_bps[-1]
                    thr = float(fb_bps) / 10_000.0

                net_return = raw_return - exec_cost_bps / 10_000.0
                hit = net_return >= thr
                entry[f"hit_{horizon}"] = bool(hit)
                entry[f"return_{horizon}"] = round(float(net_return), 6)
                entry[f"return_raw_{horizon}"] = round(float(raw_return), 6)
                entry[f"threshold_{horizon}"] = round(float(thr), 6)

                # 4 fill models — compute once per horizon.
                try:
                    # Price at +10s and +30s relative to signal (for maker models)
                    p10_mask = candles["timestamp"] >= pd.Timestamp(
                        (pulse_ts + timedelta(seconds=10)).replace(tzinfo=None)
                    )
                    p30_mask = candles["timestamp"] >= pd.Timestamp(
                        (pulse_ts + timedelta(seconds=30)).replace(tzinfo=None)
                    )
                    p10 = float(candles[p10_mask].iloc[0]["open"]) if p10_mask.any() else None
                    p30 = float(candles[p30_mask].iloc[0]["open"]) if p30_mask.any() else None
                    fills = simple_fill_returns(
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=fwd_price,
                        spread_bps=float(cfg.get("fill_models", "slippage_bps", default=5)) * 0.4,
                        slippage_bps=float(cfg.get("fill_models", "slippage_bps", default=5)),
                        price_at_10s=p10,
                        price_at_30s=p30,
                        maker_reject_move_bps=float(cfg.get("fill_models", "maker_reject_move_bps", default=100)),
                        impact_coefficient=float(cfg.get("fill_models", "impact_coefficient", default=10)),
                    )
                    fill_key = f"fills_{horizon}"
                    entry[fill_key] = {
                        model: {
                            "gross_return": round(fr.gross_return, 6),
                            "cost_bps": round(fr.cost_bps, 2),
                            "net_return": round(fr.net_return, 6),
                            "notes": fr.notes,
                        }
                        for model, fr in fills.items()
                    }
                except Exception as _e:
                    entry[f"fills_{horizon}_error"] = str(_e)[:200]

            entry["scored"] = True
            entry["scored_at"] = now_utc.isoformat()
            modified = True

        if modified:
            # Atomic rewrite: tmp + os.replace (same directory on POSIX = atomic)
            tmp_path = pulse_path.with_suffix(".jsonl.tmp")
            with open(tmp_path, "w") as f:
                for entry in all_entries:
                    raw = entry.pop("_raw", None)
                    if raw:
                        f.write(raw + "\n")
                    else:
                        f.write(json.dumps(entry, default=str) + "\n")
            os.replace(tmp_path, pulse_path)


# ── Pulse Backtest API ────────────────────────────────────────────────

class PulseBacktestRequest(BaseModel):
    start_date: str
    end_date: str
    interval_minutes: int = 15
    threshold: float = 0.25


@app.post("/api/pulse/backtest/{ticker}")
async def start_pulse_backtest(ticker: str, req: PulseBacktestRequest):
    """Start a historical pulse replay backtest. Returns SSE stream."""
    ticker = ticker.upper()

    # Validate crypto ticker
    if not ticker.endswith("-USD") and not ticker.endswith("USDT"):
        raise HTTPException(status_code=400, detail="Pulse backtest is crypto-only (ticker must end with -USD)")

    # Validate dates
    try:
        start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(req.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format (use YYYY-MM-DD)")

    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end_date must be after start_date")

    days = (end_dt - start_dt).days
    if days > 90:
        raise HTTPException(status_code=400, detail="Max backtest range is 90 days")

    job_id = str(uuid.uuid4())[:8]

    async def run_backtest():
        try:
            from tradingagents.backtesting.pulse_backtest import PulseBacktestEngine
            engine = PulseBacktestEngine(
                ticker=ticker,
                start_date=req.start_date,
                end_date=req.end_date,
                pulse_interval_minutes=req.interval_minutes,
                signal_threshold=req.threshold,
                results_dir=str(EVAL_RESULTS_DIR),
            )

            yield {
                "event": "progress",
                "data": json.dumps({
                    "phase": "starting",
                    "message": f"Starting pulse backtest for {ticker} ({req.start_date} to {req.end_date})",
                }),
            }

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, engine.run)

            yield {
                "event": "result",
                "data": json.dumps(result, default=str),
            }
            yield {
                "event": "done",
                "data": json.dumps({"status": "completed"}),
            }
        except Exception as e:
            logger.error(f"[Pulse Backtest] Error: {e}")
            import traceback
            traceback.print_exc()
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(run_backtest())


# ── Auto-Tune Endpoints (Stage 2 Phase A) ────────────────────────────
#
# The auto-tune orchestrator (``tradingagents.backtesting.autotune``)
# runs a walk-forward Latin-Hypercube sweep over a curated 6-parameter
# space and produces a ``TuneReport`` with a verdict (PROPOSE /
# PROVISIONAL / REJECT). Endpoints here:
#
#   POST /api/pulse/autotune/{ticker}           start job, SSE progress
#   GET  /api/pulse/autotune/jobs/{job_id}      poll status / resume
#   GET  /api/pulse/autotune/artifacts          list recent tune artifacts
#   GET  /api/pulse/autotune/artifacts/{name}   fetch a specific artifact
#   POST /api/pulse/autotune/apply              apply a PROPOSE-verdict artifact
#
# The apply endpoint NEVER auto-applies — it requires an explicit user
# request + re-checks verdict gates defensively before writing the YAML
# (preserves the propose-only contract from the plan even if an artifact
# file was tampered with).

class PulseAutoTuneRequest(BaseModel):
    """Shape of POST /api/pulse/autotune/{ticker} body.

    Defaults match the user-confirmed Balanced profile:
      * 60-day window → fast iteration (labeled PROVISIONAL if trades<400)
      * 3 folds       → minimum for PBO / cross-validation
      * 30 configs    → 6-dim LHS at N=30 gets good coverage
    """
    start_date: str
    end_date: str
    n_folds: int = 3
    n_configs: int = 30
    active_regime: str = "base"
    pulse_interval_minutes: int = 15
    seed: int = 42
    # Stage 2 Commit I — provenance tag stamped onto the resulting
    # artifact. "manual" is the Stage 1 default; "auto-drift" is set by
    # the weekly drift monitor when it triggers a re-tune.
    source: str = "manual"


class PulseAutoTuneApplyRequest(BaseModel):
    artifact_path: str
    # Optional sanity check: must match the hash in the artifact to
    # prevent "apply an old artifact" after config has drifted.
    expected_current_config_hash: Optional[str] = None


#: In-process registry of auto-tune jobs. Survives page refresh since
#: the server thread holds it, but not a full process restart — for
#: that we rely on the checkpoint JSONL (on-disk) and the artifact
#: (on-disk). State fields: status, progress (last emitted event),
#: result (TuneReport dict once done), error.
_AUTOTUNE_JOBS: Dict[str, Dict[str, Any]] = {}
_AUTOTUNE_JOB_LOCK = threading.Lock()
# Stage 2 F.2 — inflight guard keyed on (ticker, active_regime). A second
# POST for the same (ticker, regime) while a job is running returns 409
# with the existing job_id, so the apply path never races two artifacts
# for the same profile.
_AUTOTUNE_INFLIGHT: Dict[Tuple[str, str], str] = {}
# Stage 2 Commit I — 48h cooldown per (ticker, regime) for auto-drift
# triggers only. Manual POSTs are never rate-limited. Value = epoch
# seconds of the last auto-drift trigger accepted.
_AUTOTUNE_DRIFT_COOLDOWN: Dict[Tuple[str, str], float] = {}
_AUTOTUNE_DRIFT_COOLDOWN_SEC = 48 * 3600


def _autotune_spec_from_request(ticker: str, req: PulseAutoTuneRequest):
    """Build a :class:`TuneSpec` — validates inputs, raises HTTPException."""
    from tradingagents.backtesting.autotune import TuneSpec

    try:
        start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(req.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format (use YYYY-MM-DD)")
    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end_date must be after start_date")

    days = (end_dt - start_dt).days
    if days < 14:
        raise HTTPException(status_code=400,
                            detail="Window too short (≥14 days required)")
    # Cap at 2 years — anything longer is usually a mistake and
    # murders the CI/rate-limit budget.
    if days > 730:
        raise HTTPException(status_code=400, detail="Window too large (max 730 days)")

    # Stage 2 Commit G — 'range_bound' is the crypto-specific replacement
    # for 'sideways'; 'sideways' kept for back-compat with existing YAML.
    valid_regimes = {"base", "bull", "bear", "sideways", "range_bound", "ambiguous"}
    if req.active_regime not in valid_regimes:
        raise HTTPException(
            status_code=400,
            detail=f"active_regime must be one of {sorted(valid_regimes)}",
        )

    if not (2 <= req.n_folds <= 10):
        raise HTTPException(status_code=400, detail="n_folds must be in [2, 10]")
    if not (4 <= req.n_configs <= 100):
        raise HTTPException(status_code=400, detail="n_configs must be in [4, 100]")

    return TuneSpec(
        ticker=ticker,
        start_date=req.start_date,
        end_date=req.end_date,
        active_regime=req.active_regime,
        n_folds=req.n_folds,
        n_configs=req.n_configs,
        pulse_interval_minutes=req.pulse_interval_minutes,
        seed=req.seed,
    )


@app.post("/api/pulse/autotune/apply")
async def apply_pulse_autotune(req: PulseAutoTuneApplyRequest):
    """Apply a PROPOSE-verdict artifact to ``config/pulse_scoring.yaml``.

    (Moved to the top of the autotune route group so that
    ``POST /api/pulse/autotune/apply`` isn't matched as
    ``{ticker}="apply"`` — FastAPI resolves routes in declaration order.)
    See full docstring on the implementation below.
    """
    return await _apply_pulse_autotune_impl(req)


@app.post("/api/pulse/autotune/{ticker}")
async def start_pulse_autotune(ticker: str, req: PulseAutoTuneRequest):
    """Launch a walk-forward auto-tune sweep — returns SSE stream.

    The actual orchestration runs in the default threadpool executor
    via ``contextvars.copy_context().run(...)`` so any ContextVar
    overrides set by the orchestrator propagate into the backtest
    engine (critical for config isolation — see the ContextVar docstring
    in ``tradingagents/pulse/config.py``).

    The server tracks job state in ``_AUTOTUNE_JOBS`` so a poll endpoint
    (``GET /api/pulse/autotune/jobs/{job_id}``) can recover if the SSE
    connection drops before the final ``result`` event is emitted.
    """
    ticker = ticker.upper()
    if not ticker.endswith("-USD") and not ticker.endswith("USDT"):
        raise HTTPException(
            status_code=400,
            detail="Pulse auto-tune is crypto-only (ticker must end with -USD)",
        )

    spec = _autotune_spec_from_request(ticker, req)

    # Commit I — 48h cooldown: drift-triggered re-tunes for the same
    # (ticker, regime) are rate-limited so a flapping alert doesn't
    # spawn back-to-back jobs. Manual POSTs bypass the cooldown.
    cooldown_key = (ticker, req.active_regime)
    if req.source == "auto-drift":
        last = _AUTOTUNE_DRIFT_COOLDOWN.get(cooldown_key, 0.0)
        remaining = _AUTOTUNE_DRIFT_COOLDOWN_SEC - (time.time() - last)
        if remaining > 0:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "auto_drift_cooldown",
                    "ticker": ticker,
                    "active_regime": req.active_regime,
                    "remaining_sec": int(remaining),
                },
            )

    # F.2 — acquire the (ticker, regime) inflight slot BEFORE allocating
    # a job_id. If another tune is running for the same pair, return 409
    # with that job's id so the caller can just resume its SSE stream.
    inflight_key = (ticker, req.active_regime)
    with _AUTOTUNE_JOB_LOCK:
        existing = _AUTOTUNE_INFLIGHT.get(inflight_key)
        if existing is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "autotune_inflight",
                    "job_id": existing,
                    "ticker": ticker,
                    "active_regime": req.active_regime,
                },
            )
        job_id = str(uuid.uuid4())[:12]
        _AUTOTUNE_INFLIGHT[inflight_key] = job_id
        if req.source == "auto-drift":
            _AUTOTUNE_DRIFT_COOLDOWN[cooldown_key] = time.time()

    queue: asyncio.Queue = asyncio.Queue(maxsize=1024)

    with _AUTOTUNE_JOB_LOCK:
        _AUTOTUNE_JOBS[job_id] = {
            "job_id": job_id,
            "ticker": ticker,
            "spec": spec,
            "status": "running",
            "progress": None,
            "result": None,
            "error": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    loop = asyncio.get_event_loop()

    def progress_cb(event: dict) -> None:
        """Invoked by the orchestrator on each backtest completion.

        Must be thread-safe relative to the asyncio event loop — we
        schedule the enqueue via ``call_soon_threadsafe`` since this
        runs inside the executor thread.
        """
        with _AUTOTUNE_JOB_LOCK:
            _AUTOTUNE_JOBS[job_id]["progress"] = event
        try:
            loop.call_soon_threadsafe(queue.put_nowait, {"event": "progress", "data": event})
        except Exception:
            pass  # queue full or loop closed — drop the event

    def run_job() -> None:
        from tradingagents.backtesting.autotune import AutoTuner
        try:
            tuner = AutoTuner(spec=spec, progress_cb=progress_cb, job_id=job_id)
            report = tuner.run()
            # Commit I — stamp provenance (manual vs auto-drift) onto the
            # persisted artifact so the Proposals UI can badge it.
            try:
                ap = Path(report.artifact_path)
                if ap.exists():
                    data = json.loads(ap.read_text())
                    data["source"] = req.source
                    ap.write_text(json.dumps(data, indent=2, default=str))
            except Exception as e:
                logger.warning("artifact source stamp failed: %s", e)
            payload = {
                "job_id": job_id,
                "verdict": report.verdict,
                "reasons": report.reasons,
                "current_config_hash": report.current_config_hash,
                "proposed_config": report.proposed_config,
                "proposed_config_hash": report.proposed_config_hash,
                "diff": report.diff,
                "metrics": report.metrics,
                "per_fold": report.per_fold,
                "artifact_path": report.artifact_path,
                "ran_at": report.ran_at,
                "source": req.source,
            }
            with _AUTOTUNE_JOB_LOCK:
                _AUTOTUNE_JOBS[job_id]["status"] = "done"
                _AUTOTUNE_JOBS[job_id]["result"] = payload
            loop.call_soon_threadsafe(queue.put_nowait, {"event": "result", "data": payload})
            loop.call_soon_threadsafe(queue.put_nowait, {"event": "done", "data": {"status": "completed"}})
        except Exception as e:
            logger.error(f"[AutoTune job={job_id}] failed: {e}")
            import traceback
            traceback.print_exc()
            with _AUTOTUNE_JOB_LOCK:
                _AUTOTUNE_JOBS[job_id]["status"] = "error"
                _AUTOTUNE_JOBS[job_id]["error"] = str(e)
            loop.call_soon_threadsafe(queue.put_nowait, {"event": "error", "data": {"error": str(e)}})
        finally:
            # F.2 — release inflight slot on terminal state (done or error).
            # The handler can't do this because it returns as soon as SSE
            # opens; only the executor task sees the real end.
            with _AUTOTUNE_JOB_LOCK:
                if _AUTOTUNE_INFLIGHT.get(inflight_key) == job_id:
                    _AUTOTUNE_INFLIGHT.pop(inflight_key, None)

    # Launch via copy_context().run so the ContextVar override system
    # the engine uses is properly propagated. We do NOT ``await`` the
    # future — the job should run concurrently with the SSE drain loop,
    # otherwise progress events wouldn't reach the client until after
    # the whole job completed (defeats the point of streaming).
    ctx = contextvars.copy_context()
    job_future = loop.run_in_executor(None, ctx.run, run_job)
    # Retaining the future prevents GC (which would cancel the task)
    # and lets us observe exceptions in the drain loop.
    with _AUTOTUNE_JOB_LOCK:
        _AUTOTUNE_JOBS[job_id]["_future"] = job_future

    async def sse_stream():
        """Drain the progress queue until a terminal event is seen."""
        try:
            while True:
                msg = await asyncio.wait_for(queue.get(), timeout=3600)
                yield {"event": msg["event"], "data": json.dumps(msg["data"], default=str)}
                if msg["event"] in ("done", "error"):
                    return
        except asyncio.TimeoutError:
            yield {"event": "error", "data": json.dumps({"error": "timeout"})}

    return EventSourceResponse(sse_stream())


@app.get("/api/pulse/autotune/jobs/{job_id}")
async def get_pulse_autotune_job(job_id: str):
    """Return the current snapshot of a job — for post-refresh resume.

    Callers that lost the SSE connection can poll this endpoint; the
    response includes the latest progress event and, if complete, the
    full result payload. 404 when the job is unknown (process restart
    or purged).
    """
    with _AUTOTUNE_JOB_LOCK:
        job = _AUTOTUNE_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job_id not found")
        # Return a copy — spec dataclass isn't directly JSON-serializable.
        from dataclasses import asdict
        return {
            "job_id": job["job_id"],
            "ticker": job["ticker"],
            "spec": asdict(job["spec"]),
            "status": job["status"],
            "progress": job["progress"],
            "result": job["result"],
            "error": job["error"],
            "started_at": job["started_at"],
        }


@app.get("/api/pulse/autotune/artifacts")
async def list_pulse_autotune_artifacts(limit: int = 50):
    """Return the last N auto-tune artifact filenames + summary metadata.

    Used by the UI's "recent proposals" list. Artifacts are immutable
    once written (the apply endpoint never rewrites them) so cheap to
    scan — we only read the top-level keys for the summary.
    """
    artifact_dir = Path("results/autotune")
    if not artifact_dir.exists():
        return {"artifacts": []}
    entries: List[Dict[str, Any]] = []
    for f in sorted(artifact_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        entries.append({
            "artifact": f.name,
            "path": str(f),
            "ran_at": payload.get("ran_at"),
            "ticker": payload.get("spec", {}).get("ticker"),
            "active_regime": payload.get("spec", {}).get("active_regime"),
            "source": payload.get("source", "manual"),
            "verdict": payload.get("verdict"),
            "current_config_hash": payload.get("current_config_hash"),
            "proposed_config_hash": payload.get("proposed_config_hash"),
            "n_changes": len(payload.get("diff", [])),
        })
    return {"artifacts": entries}


@app.get("/api/pulse/autotune/artifacts/{name}")
async def get_pulse_autotune_artifact(name: str):
    """Fetch the full payload of a specific artifact by filename.

    Path traversal guard: we only accept a bare filename ending in
    ``.json`` and restrict lookup to ``results/autotune/``.
    """
    if "/" in name or ".." in name or not name.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid artifact name")
    path = Path("results/autotune") / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corrupt artifact: {e}")


async def _apply_pulse_autotune_impl(req: PulseAutoTuneApplyRequest):
    """Apply a PROPOSE-verdict artifact to ``config/pulse_scoring.yaml``.

    Safety layers (defense in depth):

      1. Artifact must exist under ``results/autotune/``.
      2. Artifact ``verdict`` must be ``PROPOSE`` exactly. PROVISIONAL
         artifacts are 409-rejected with a message pointing to the
         sample-size requirement.
      3. Optional ``expected_current_config_hash`` — if supplied, must
         match the artifact's ``current_config_hash``. Prevents applying
         a stale artifact on top of a newer base.
      4. The proposed configuration is re-validated through
         ``_validate()`` in the config module before writing.
      5. For ``active_regime != "base"``, the proposal is written into
         ``regime_profiles.<regime>`` only — base is untouched.
      6. Atomic write (tmp + rename) via ``write_config_atomic``.
      7. Calibration metadata (``calibrated_at``, ``deflated_sharpe``,
         ``pbo``, ``oos_sharpe``) is stamped.
    """
    from tradingagents.pulse.config import (
        DEFAULT_CONFIG_PATH, _validate, deep_merge, write_config_atomic,
    )

    # Path-traversal guard: treat the user-supplied path as relative to
    # the artifact directory and refuse anything outside.
    raw_path = Path(req.artifact_path)
    if raw_path.is_absolute():
        artifact_path = raw_path
    else:
        artifact_path = Path(req.artifact_path)
    if ".." in str(artifact_path):
        raise HTTPException(status_code=400, detail="Invalid artifact_path")
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")

    try:
        payload = json.loads(artifact_path.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corrupt artifact: {e}")

    verdict = payload.get("verdict")
    if verdict != "PROPOSE":
        # Distinguish PROVISIONAL (sample-size gate) from REJECT (failed
        # other gates) so the UI can show a useful error.
        detail = (
            "PROVISIONAL verdict — widen the tuning window (or lower "
            "n_configs) until OOS trade count meets the per-regime "
            "minimum (see MIN_TRADES_OOS_BY_REGIME)."
            if verdict == "PROVISIONAL"
            else f"Cannot apply artifact with verdict={verdict!r} — must be PROPOSE"
        )
        raise HTTPException(status_code=409, detail=detail)

    if req.expected_current_config_hash and \
            payload.get("current_config_hash") != req.expected_current_config_hash:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Config has drifted since this artifact was produced "
                f"(artifact base_hash={payload.get('current_config_hash')!r} vs "
                f"expected {req.expected_current_config_hash!r}). Re-run the tune."
            ),
        )

    proposed_flat = payload.get("proposed_config") or {}
    if not proposed_flat:
        raise HTTPException(status_code=400, detail="Artifact has empty proposed_config")

    # Flatten dotted-path candidate → nested overlay.
    overlay: Dict[str, Any] = {}
    for path, value in proposed_flat.items():
        keys = str(path).split(".")
        node = overlay
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    # Re-read current YAML to compose the write payload.
    import yaml as _yaml
    try:
        current_data = _yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()) or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot read live config: {e}")

    active_regime = payload.get("spec", {}).get("active_regime", "base")
    if active_regime in (None, "", "base"):
        # Merge directly into base.
        new_data = deep_merge(current_data, overlay)
    else:
        # Merge into ``regime_profiles.<regime>`` only; base untouched.
        profiles = dict(current_data.get("regime_profiles") or {})
        profile_now = profiles.get(active_regime) or {}
        profiles[active_regime] = deep_merge(profile_now, overlay)
        new_data = dict(current_data)
        new_data["regime_profiles"] = profiles

    # Stamp calibration metadata so the scorecard UI surfaces provenance.
    metrics = payload.get("metrics") or {}
    new_data["calibrated_at"] = datetime.now(timezone.utc).isoformat()
    new_data["calibration_window"] = {
        "start_date": payload.get("spec", {}).get("start_date"),
        "end_date": payload.get("spec", {}).get("end_date"),
        "n_trades_oos": metrics.get("oos_n_trades_total"),
        "n_folds_used": metrics.get("n_folds_used"),
        "n_effective": metrics.get("n_eff"),
        "engine_version": int(new_data.get("engine_version", 3)),
        "active_regime": active_regime,
        "artifact": artifact_path.name,
    }
    new_data["deflated_sharpe"] = metrics.get("deflated_oos_sharpe")
    new_data["oos_sharpe"] = metrics.get("oos_sharpe_point")
    new_data["pbo"] = metrics.get("pbo")
    new_data["n_eff"] = metrics.get("n_eff")

    # Final validation before the atomic write. The module's ``_validate``
    # raises ValueError on any out-of-range param — catch and surface
    # as 400 so a UI copy-paste bug never corrupts the live YAML.
    try:
        _validate(new_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Proposed config invalid: {e}")

    try:
        cfg_after = write_config_atomic(new_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Write failed: {e}")

    logger.info(
        f"[AutoTune.apply] Applied artifact={artifact_path.name} "
        f"regime={active_regime} new_hash={cfg_after.hash_short()}"
    )
    return {
        "applied": True,
        "new_config_hash": cfg_after.content_hash,
        "active_regime": active_regime,
        "calibrated_at": new_data["calibrated_at"],
        "calibration_window": new_data["calibration_window"],
    }


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
