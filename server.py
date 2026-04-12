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
import json
import logging
import os
import re
import threading
import time
import uuid
import requests
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────
app = FastAPI(title="TradingAgents API", version="1.0.0")

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
_DATA_DELAY_SECONDS = 300         # 5-min offset so data APIs have the closed candle

_scheduler_state: Dict[str, Any] = {
    "enabled": False,
    "task": None,           # asyncio.Task handle
    "last_run": None,       # ISO string of last logical candle time run
    "last_status": None,    # 'ok' | 'error'
    "next_run": None,       # ISO string of next scheduled logical candle time
    "tickers": _SCHEDULER_TICKERS,
}


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
    logger.info("[Scheduler] 4H auto-analysis scheduler started")
    while _scheduler_state["enabled"]:
        try:
            boundary_utc = _next_4h_utc_boundary()
            # Physical trigger = boundary + data delay
            physical_trigger = boundary_utc + timedelta(seconds=_DATA_DELAY_SECONDS)
            # Sleep until that moment
            now_utc = datetime.now(pytz.utc)
            sleep_secs = (physical_trigger - now_utc).total_seconds()
            if sleep_secs > 0:
                # Logical candle label (e.g. '2026-04-08T16')
                logical_label = boundary_utc.strftime("%Y-%m-%dT%H")
                logical_date = boundary_utc.strftime("%Y-%m-%d")
                _scheduler_state["next_run"] = boundary_utc.isoformat()
                logger.info(
                    f"[Scheduler] Next run: {logical_label} UTC (sleeping {sleep_secs:.0f}s until data ready)"
                )
                await asyncio.sleep(sleep_secs)

            if not _scheduler_state["enabled"]:
                break

            # Recalculate logical boundary (may shift slightly if sleep overran)
            logical_label = boundary_utc.strftime("%Y-%m-%dT%H")
            logical_date = boundary_utc.strftime("%Y-%m-%d")

            # Check if ensemble mode will be used
            from tradingagents.default_config import DEFAULT_CONFIG
            from tradingagents.graph.ensemble_orchestrator import should_use_ensemble
            current_provider = DEFAULT_CONFIG.get("llm_provider", "deepseek")
            ensemble_active = should_use_ensemble(DEFAULT_CONFIG, current_provider)
            
            if ensemble_active:
                logger.info(f"[Scheduler] Running ENSEMBLE auto-analysis (3x parallel) for {_SCHEDULER_TICKERS} @ {logical_label}")
            else:
                logger.info(f"[Scheduler] Running auto-analysis for {_SCHEDULER_TICKERS} @ {logical_label} (provider: {current_provider})")
            
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
                    "scheduled": True,
                    "candle_time": logical_label,
                }
                thread = threading.Thread(
                    target=_run_analysis,
                    args=(job_id, ticker, logical_date, True, logical_label),
                    daemon=True,
                )
                thread.start()
                logger.info(f"[Scheduler] Launched job {job_id} for {ticker} @ {logical_label}")

            _scheduler_state["last_run"] = logical_label
            _scheduler_state["last_status"] = "ok"
            _scheduler_state["error_count"] = 0  # Reset on success

        except asyncio.CancelledError:
            logger.info("[Scheduler] Task cancelled")
            break
        except Exception as e:
            error_count = _scheduler_state.get("error_count", 0) + 1
            _scheduler_state["error_count"] = error_count
            logger.error(f"[Scheduler] Error ({error_count}/3): {e}")
            _scheduler_state["last_status"] = f"error: {e}"
            
            # BLOCKER FIX: Circuit breaker - disable after 3 consecutive errors
            if error_count >= 3:
                _scheduler_state["enabled"] = False
                logger.error("[Scheduler] DISABLED after 3 consecutive errors")
                break
            
            await asyncio.sleep(60)  # brief wait before retry

    logger.info("[Scheduler] Stopped")


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

@app.on_event("startup")
async def _start_background_tasks():
    asyncio.create_task(_evict_old_backtest_jobs())


# ── Scheduler API Endpoints ───────────────────────────────────────────

@app.get("/api/scheduler/status")
async def scheduler_status():
    """Return current scheduler state including next run time (local + UTC)."""
    import pytz
    try:
        boundary_utc = _next_4h_utc_boundary()
        # Convert to local time for display
        local_tz = datetime.now().astimezone().tzinfo
        boundary_local = boundary_utc.astimezone(local_tz)
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
        logger.info("[Scheduler] Disabled via API")
        return {"enabled": False, "message": "Scheduler disabled"}
    else:
        # Enable: start the background task (singleton guard)
        task = _scheduler_state.get("task")
        if task and not task.done():
            return {"enabled": True, "message": "Already running"}
        _scheduler_state["enabled"] = True
        _scheduler_state["task"] = asyncio.create_task(_auto_analysis_scheduler())
        logger.info("[Scheduler] Enabled via API")
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


@app.get("/api/model/config")
async def get_model_config():
    """Get current model configuration including ensemble settings."""
    from tradingagents.default_config import DEFAULT_CONFIG
    
    return {
        "provider": DEFAULT_CONFIG.get("llm_provider", "openrouter"),
        "model": DEFAULT_CONFIG.get("deep_think_llm", "qwen/qwen3.6-plus"),
        "ensemble_enabled": DEFAULT_CONFIG.get("enable_ensemble", True),
        "ensemble_runs": DEFAULT_CONFIG.get("ensemble_runs", 3),
        "ensemble_providers": DEFAULT_CONFIG.get("ensemble_enabled_providers", ["openrouter"]),
        "single_run_providers": DEFAULT_CONFIG.get("ensemble_disabled_providers", ["deepseek"]),
        "fallback_model": DEFAULT_CONFIG.get("openrouter_fallback_model", "anthropic/claude-3.5-sonnet"),
    }


@app.post("/api/model/config")
async def set_model_config(req: ModelConfigRequest):
    """Update model configuration for session (non-persistent).
    
    Note: Ensemble is auto-disabled for providers in ensemble_disabled_providers.
    """
    from tradingagents.default_config import DEFAULT_CONFIG
    
    # Update configuration
    DEFAULT_CONFIG["llm_provider"] = req.provider
    DEFAULT_CONFIG["deep_think_llm"] = req.model
    # For third-party providers, quick_think must also use the selected model
    # (the OpenAI default gpt-5-mini would be routed through the provider and billed)
    DEFAULT_CONFIG["quick_think_llm"] = req.model
    
    # Check if provider supports ensemble
    disabled_providers = DEFAULT_CONFIG.get("ensemble_disabled_providers", ["deepseek"])
    enabled_providers = DEFAULT_CONFIG.get("ensemble_enabled_providers", ["openrouter"])
    
    if req.provider.lower() in disabled_providers:
        DEFAULT_CONFIG["enable_ensemble"] = False
        ensemble_active = False
        message = f"Ensemble disabled (not supported for {req.provider})"
    elif req.provider.lower() in enabled_providers:
        DEFAULT_CONFIG["enable_ensemble"] = req.parallel_mode
        ensemble_active = req.parallel_mode
        message = f"Ensemble {'enabled' if req.parallel_mode else 'disabled'} for {req.provider}"
    else:
        # Default to user preference
        DEFAULT_CONFIG["enable_ensemble"] = req.parallel_mode
        ensemble_active = req.parallel_mode
        message = f"Configuration updated"
    
    return {
        "status": "ok",
        "provider": req.provider,
        "model": req.model,
        "ensemble_active": ensemble_active,
        "message": message,
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
    """Thread-safe event queue for SSE streaming with buffering."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._buffer: list = []  # Buffer events before loop is set
        self._lock = threading.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        with self._lock:
            self._loop = loop
            # Flush buffered events
            for event in self._buffer:
                self._unsafe_put(event)
            self._buffer.clear()

    def _unsafe_put(self, event: dict):
        """Put event into queue - must be called with lock held or after loop is set."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def put(self, event: dict):
        with self._lock:
            if self._loop and not self._loop.is_closed():
                self._unsafe_put(event)
            else:
                # Buffer events until loop is set
                self._buffer.append(event)

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


def _detect_step_from_chunk(chunk: dict, seen_steps: set) -> Optional[dict]:
    """Detect which step just completed based on what fields are present in the chunk."""
    # Check for report fields to determine which analyst just finished
    report_fields = [
        ("market_report", "market", "Market Analyst", 1),
        ("sentiment_report", "social", "Social Media Analyst", 2),
        ("news_report", "news", "News Analyst", 3),
        ("fundamentals_report", "fundamentals", "Fundamentals Analyst", 4),
    ]
    
    for report_key, step_key, label, step_num in report_fields:
        if chunk.get(report_key) and step_key not in seen_steps:
            return {"key": step_key, "label": label, "step": step_num}
    
    # Check for debate states
    investment_debate = chunk.get("investment_debate_state", {})
    if investment_debate.get("bull_history") and investment_debate.get("bear_history") and "bull_bear" not in seen_steps:
        return {"key": "bull_bear", "label": "Bull vs Bear Debate", "step": 5}
    
    # Check for risk debate
    risk_debate = chunk.get("risk_debate_state", {})
    if risk_debate.get("aggressive_history") and "risk_debate" not in seen_steps:
        return {"key": "risk_debate", "label": "Risk Debate", "step": 8}
    
    # Check for final decision
    if chunk.get("final_trade_decision") and "portfolio_manager" not in seen_steps:
        return {"key": "portfolio_manager", "label": "Portfolio Manager", "step": 9}
    
    # Check for trader plan
    if chunk.get("trader_investment_plan") and "trader" not in seen_steps:
        return {"key": "trader", "label": "Trader", "step": 7}
    
    # Check for investment plan (research manager)
    if chunk.get("investment_plan") and "research_manager" not in seen_steps:
        return {"key": "research_manager", "label": "Research Manager", "step": 6}
    
    return None


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
            n_runs = config.get("ensemble_runs", 3)
            print(f"[Analysis {job_id}] Ensemble mode enabled for {current_provider} ({n_runs} members)")
            eq.put({"event": "agent_start", "agent": "Ensemble Analysis", "step": 0, "total": 9})
            
            # Emit progress: data gathering phase
            eq.put({"event": "agent_start", "agent": "Market Analyst", "step": 1, "total": 9})
            eq.put({"event": "agent_start", "agent": "Social Media Analyst", "step": 2, "total": 9})
            eq.put({"event": "agent_start", "agent": "News Analyst", "step": 3, "total": 9})
            eq.put({"event": "agent_start", "agent": "Fundamentals Analyst", "step": 4, "total": 9})
            
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
                message = "All ensemble runs failed (provider returned no valid outputs)."
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = message
                eq.put({"event": "error", "message": message})
                return
            
            # Convert ConsensusResult to result dict
            result = _ensemble_result_to_dict(consensus_result, ticker, run_timestamp or trade_date)
            
            # Log and complete
            print(f"[Analysis {job_id}] Ensemble complete: {consensus_result.signal} "
                  f"(conf={consensus_result.confidence:.2f}, "
                  f"agreement={consensus_result.divergence_metrics.get('signal_agreement', 0):.0%})")
            
            jobs[job_id]["result"] = result
            jobs[job_id]["status"] = "done"
            
            # Dispatch Telegram
            try:
                _send_telegram_alert(result)
            except Exception as e:
                print(f"[Analysis {job_id}] Telegram alert warning: {e}")
            
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
            "r_ratio": scored.get("r_ratio"),
            "r_ratio_warning": scored.get("r_ratio_warning", False),
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
                "reasoning": reasoning,
                "regime": _shadow_regime,
                "source": "live_analysis",
                "recorded_at": datetime.now().isoformat(),
                "scored": False,
            }
            with open(shadow_dir / "decisions.jsonl", "a") as _sf:
                _sf.write(json.dumps(shadow_entry, default=str) + "\n")
        except Exception as _shadow_err:
            print(f"[Analysis {job_id}] Shadow record failed (non-fatal): {_shadow_err}")

        eq.put({"event": "decision", "signal": decision_signal})
        eq.put({"event": "done", "result": result})
        print(f"[Analysis {job_id}] Done event sent")

    except Exception as e:
        print(f"[Analysis {job_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        
        # Parse error for better user message
        error_str = str(e)
        is_rate_limit = "429" in error_str or "rate-limit" in error_str.lower()
        if is_rate_limit:
            user_message = "Rate limited by provider. The free model has strict limits. Try again shortly or switch to a non-free model."
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

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "features": {
            "analysis": True,
            "backtest": True,
            "history": True,
            "streaming": True,
        },
        "timestamp": datetime.now().isoformat(),
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

    stream_start = time.time()

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(eq.get(), timeout=30)
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
                # Extract structured fields
                reasoning = parsed.get("reasoning", "")
                signal = parsed.get("signal", "")
                confidence = parsed.get("confidence")
                max_hold = parsed.get("max_hold_days")
                
                parts = []
                if reasoning:
                    # Truncate long reasoning to fit Telegram limits
                    max_len = 800
                    reason_text = reasoning[:max_len] + "..." if len(reasoning) > max_len else reasoning
                    parts.append(f"<b>Analysis:</b> {_escape_html(reason_text)}")
                
                if signal:
                    parts.append(f"<b>Signal:</b> {signal}")
                if confidence is not None:
                    parts.append(f"<b>Confidence:</b> {confidence * 100:.1f}%")
                if max_hold:
                    parts.append(f"<b>Max Hold:</b> {max_hold} days")
                
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
                local_dt = dt.astimezone()
                return local_dt.strftime("%b %d, %Y at %I:%M %p %Z")
            
            # Format 3: ISO format YYYY-MM-DDTHH:MM:SS
            if "T" in raw_date:
                from datetime import timezone
                dt = datetime.fromisoformat(raw_date.replace("Z", "").replace("+00:00", ""))
                dt = dt.replace(tzinfo=timezone.utc)
                local_dt = dt.astimezone()
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
    conf = result.get("confidence", 0)
    
    # Ensure stop loss / take profit handle empty paths
    sl = result.get("stop_loss_price")
    tp = result.get("take_profit_price")
    sl_str = f"${float(sl):,.2f}" if sl else "N/A"
    tp_str = f"${float(tp):,.2f}" if tp else "N/A"
    sizing = result.get("position_size_pct", 0)
    
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
        f"{rr_prefix}Conviction: {conviction}\n\n"
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
            parsed = json.loads(decision_text)
            decision_signal = _extract_signal(parsed.get("signal", ""))
            stop_loss_price = parsed.get("stop_loss_price")
            take_profit_price = parsed.get("take_profit_price")
            confidence = parsed.get("confidence")
            max_hold_days = parsed.get("max_hold_days")
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
        date_data["confidence"] = confidence if confidence is not None else 0.50
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
            # Legacy format with T separator
            analysis_dt = datetime.fromisoformat(analysis_date.replace("Z", ""))
            analysis_dt = analysis_dt.replace(tzinfo=timezone.utc)
            local_dt = analysis_dt.astimezone()
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


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
