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
"""

import asyncio
import json
import logging
import os
import re
import threading
import uuid
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


class AnalyzeRequest(BaseModel):
    ticker: str
    date: Optional[str] = None


class BacktestRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    mode: str = "replay"  # "replay" or "simulation"
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


def _run_analysis(job_id: str, ticker: str, trade_date: str):
    """Run the TradingAgents analysis in a background thread."""
    eq: JobEventQueue = jobs[job_id]["queue"]
    
    # Immediately emit a starting event
    eq.put({"event": "agent_start", "agent": "Starting Analysis", "step": 0, "total": 9})
    
    import sys
    print(f"[Analysis {job_id}] Starting analysis for {ticker} on {trade_date}", flush=True)
    sys.stdout.flush()

    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "deepseek"
        config["deep_think_llm"] = "deepseek-reasoner"
        config["quick_think_llm"] = "deepseek-chat"
        config["max_debate_rounds"] = 1
        config["data_vendors"] = {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
        }

        ta = TradingAgentsGraph(debug=True, config=config)

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
                    # ToolNode stores tools in different ways depending on version
                    if hasattr(tool_node, 'tools'):
                        tool_names = [t.name for t in tool_node.tools]
                    elif hasattr(tool_node, '_tools'):
                        tool_names = [t.name for t in tool_node._tools]
                    else:
                        tool_names = ["unknown"]
                    print(f"[Analysis {job_id}] Tools for {node_name}: {tool_names}")
                except Exception as e:
                    print(f"[Analysis {job_id}] Could not list tools for {node_name}: {e}")

        print(f"[Analysis {job_id}] Starting analysis for {ticker} on {trade_date}")
        
        seen_steps = set()
        reports_cache = {}  # Track reports to avoid duplicates
        trace = []

        for chunk in ta.graph.stream(init_state, **args):
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

            trace.append(chunk)

        # Extract final state
        final_state = trace[-1] if trace else {}
        ta.curr_state = final_state
        ta._log_state(trade_date, final_state)

        # Process signal
        from tradingagents.graph.signal_processing import SignalProcessor
        decision = ta.process_signal(final_state.get("final_trade_decision", "HOLD"))

        # Build result
        result = {
            "ticker": ticker,
            "date": trade_date,
            "decision": decision,
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

        print(f"[Analysis {job_id}] Analysis complete, sending done event")
        jobs[job_id]["result"] = result
        jobs[job_id]["status"] = "done"

        eq.put({"event": "decision", "signal": decision})
        eq.put({"event": "done", "result": result})
        print(f"[Analysis {job_id}] Done event sent")

    except Exception as e:
        print(f"[Analysis {job_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        eq.put({"event": "error", "message": str(e)})


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

    jobs[job_id] = {
        "ticker": ticker,
        "date": trade_date,
        "status": "running",
        "queue": eq,
        "result": None,
        "error": None,
    }

    # Start analysis in background thread
    thread = threading.Thread(target=_run_analysis, args=(job_id, ticker, trade_date), daemon=True)
    thread.start()

    return {"job_id": job_id, "ticker": ticker, "date": trade_date}


@app.get("/api/stream/{job_id}")
async def stream_analysis(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    eq: JobEventQueue = jobs[job_id]["queue"]
    eq.set_loop(asyncio.get_event_loop())

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(eq.get(), timeout=120)
                yield {"event": event.get("event", "message"), "data": json.dumps(event)}
                if event.get("event") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": "{}"}

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
        match = re.search(r"full_states_log_(\d{4}-\d{2}-\d{2})\.json", f.name)
        if match:
            dates.append(match.group(1))
    return max(dates) if dates else None


@app.get("/api/history/{ticker}")
async def list_analyses(ticker: str):
    """List all analysis dates for a given ticker."""
    logs_dir = EVAL_RESULTS_DIR / ticker / "TradingAgentsStrategy_logs"
    if not logs_dir.exists():
        raise HTTPException(status_code=404, detail=f"No analyses found for {ticker}")

    analyses = []
    for f in sorted(logs_dir.glob("full_states_log_*.json"), reverse=True):
        match = re.search(r"full_states_log_(\d{4}-\d{2}-\d{2})\.json", f.name)
        if match:
            analysis_date = match.group(1)
            # Try to extract decision from log
            try:
                data = json.loads(f.read_text())
                date_data = data.get(analysis_date, {})
                decision_text = date_data.get("final_trade_decision", "")
                # Extract signal from decision text
                signal = _extract_signal(decision_text)
            except Exception:
                signal = "UNKNOWN"

            analyses.append({
                "date": analysis_date,
                "signal": signal,
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


@app.get("/api/history/{ticker}/{analysis_date}")
async def get_analysis(ticker: str, analysis_date: str):
    """Return full log JSON for a specific analysis."""
    log_file = EVAL_RESULTS_DIR / ticker / "TradingAgentsStrategy_logs" / f"full_states_log_{analysis_date}.json"
    if not log_file.exists():
        raise HTTPException(status_code=404, detail=f"No analysis found for {ticker} on {analysis_date}")

    data = json.loads(log_file.read_text())
    date_data = data.get(analysis_date, data)

    return {
        "ticker": ticker,
        "date": analysis_date,
        "data": date_data,
    }


# ── Price Data Endpoint ──────────────────────────────────────────────

@app.get("/api/price/{ticker}")
async def get_price(ticker: str, days: int = 90):
    """Fetch OHLCV + SMA data for chart rendering."""
    try:
        # Fetch extra data for SMA computation
        fetch_days = days + 200
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=fetch_days)

        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

        # Compute SMAs on full dataset
        hist["SMA50"] = hist["Close"].rolling(window=50).mean()
        hist["SMA200"] = hist["Close"].rolling(window=200).mean()

        # Trim to requested days
        hist = hist.tail(days)

        records = []
        for idx, row in hist.iterrows():
            records.append({
                "date": idx.strftime("%Y-%m-%d"),
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
                    "event": "status",
                    "step": 2,
                    "total_steps": 5,
                    "status": "Date range auto-adjusted",
                    "details": f"Using available logs: {start_date} to {end_date} ({len(all_log_dates)} analysis days)"
                })
            
            # Recount after potential adjustment
            base_dates_final = set([d.split(' ')[0] for d in trade_dates])
            available_logs = sum(1 for bd in base_dates_final if bd in all_log_dates)
            log(f"Found {available_logs}/{len(base_dates_final)} analysis logs for {total_dates} tick intervals")
            log(f"Available analysis dates: {', '.join(all_log_dates)}")
        
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
            
            price = _get_price_on_date(ticker, trade_date)
            if price is None:
                errors.append({"date": trade_date, "error": "No price data available"})
                log(f"  Warning: No price data for {trade_date}")
                continue
            
            funding_rate = _get_funding_on_date(ticker, trade_date)
            action = portfolio.process_signal(signal, price, trade_date, funding_rate=funding_rate)
            log(f"  {trade_date}: {signal} @ ${price:,.2f} → {action}")
            
            decisions.append({
                "date": trade_date,
                "signal": signal,
                "price": price,
                "action": action,
                "portfolio_value": portfolio.portfolio_value(price),
                "position": portfolio.position_side.value,
            })
            
            eq.put({
                "event": "decision",
                "date": trade_date,
                "signal": signal,
                "price": price,
                "action": action,
                "portfolio_value": portfolio.portfolio_value(price),
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
    
    metrics = compute_metrics(
        equity_curve=portfolio.equity_curve,
        closed_positions=portfolio.closed_positions,
        initial_capital=initial_capital,
        total_fees=portfolio.total_fees_paid,
        total_funding=portfolio.total_funding_paid,
        liquidations=portfolio.liquidations,
        leverage=leverage,
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
    
    # Re-compute metrics with crypto parameters (engine.run() uses basic compute_metrics)
    from tradingagents.backtesting.metrics import compute_metrics
    crypto_metrics = compute_metrics(
        equity_curve=engine.portfolio.equity_curve,
        closed_positions=engine.portfolio.closed_positions,
        initial_capital=initial_capital,
        total_fees=engine.portfolio.total_fees_paid,
        total_funding=engine.portfolio.total_funding_paid,
        liquidations=engine.portfolio.liquidations,
        leverage=leverage,
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
    
    result = {
        "metrics": crypto_metrics,
        "portfolio_stats": engine.portfolio.get_stats(),
        "decisions": raw_result.get("decisions", []),
        "equity_curve": engine.portfolio.equity_curve,
        "trade_history": trade_history,
        "errors": raw_result.get("errors", []),
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


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
