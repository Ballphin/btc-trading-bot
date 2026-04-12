"""Ensemble analysis orchestrator for parallel LLM analysis.

Runs multiple parallel TradingAgentsGraph instances, computes consensus,
detects divergence, and manages retries with fallback model diversity.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from tradingagents.graph.consensus_engine import ConsensusEngine, ConsensusResult
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger(__name__)


@dataclass
class EnsembleMemberResult:
    """Result from a single ensemble member run."""
    member_id: int
    signal: str
    confidence: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    max_hold_days: int
    reasoning: str
    error: Optional[str] = None


class EnsembleAnalysisOrchestrator:
    """Orchestrates parallel ensemble analysis with consensus computation.
    
    BLOCKER FIXES:
    - Timeout protection (300s per run)
    - Price snapshot at start for consistency
    - Staleness detection (>30s warning)
    - Fallback model on retry for diversity
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        provider: str,
        model: str,
        parallel_runs: int = 3,
    ):
        """Initialize the ensemble orchestrator.
        
        Args:
            config: Application configuration dictionary
            provider: LLM provider name
            model: Model identifier
            parallel_runs: Number of parallel ensemble members
        """
        self.config = config
        self.provider = provider
        self.model = model
        self.parallel_runs = parallel_runs
        self.max_retries = config.get("ensemble_max_retries", 2)
        # Free-tier models need longer; default 600s
        self.timeout_per_run = config.get("ensemble_timeout_per_run", 600)
        self.max_ensemble_time = config.get("ensemble_max_total_time", 30)  # 30 seconds
        
        # Temperature variation for diversity
        base_temp = config.get("llm_temperature", 0.4)
        self.temperatures = [
            base_temp - 0.05,
            base_temp,
            base_temp + 0.05,
        ]
        
        self.consensus_engine = ConsensusEngine()
        
    async def analyze(
        self,
        ticker: str,
        trade_date: str,
        selected_analysts: Optional[List[str]] = None,
    ) -> ConsensusResult:
        """Run parallel ensemble analysis and return consensus result.
        
        Args:
            ticker: Ticker symbol to analyze
            trade_date: Trade date string
            selected_analysts: Optional list of analyst types
            
        Returns:
            ConsensusResult with averaged parameters and metadata
        """
        # BLOCKER FIX: Snapshot price at start for consistency
        entry_price = self._get_current_price(ticker)
        ensemble_start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            logger.info(f"Ensemble attempt {attempt + 1}/{self.max_retries + 1} for {ticker}")
            
            # BLOCKER FIX: Use fallback model on retry for diversity
            current_model = self.model
            if attempt > 0:
                fallback = self.config.get("openrouter_fallback_model")
                if fallback:
                    current_model = fallback
                    logger.info(f"Retry {attempt}: Switching to fallback model {current_model}")
            
            # Run ensemble members — sequential for free-tier (rate limits),
            # parallel for paid models
            is_free_model = ":free" in self.model.lower()
            
            # Progress event helper
            eq = getattr(self, "_event_queue", None)
            def _emit_member_done(member_id, signal, error=None):
                if eq:
                    # Steps 5-7 map to ensemble members completing (after analysts 1-4)
                    step = min(5 + member_id, 8)
                    label = f"Ensemble Member {member_id + 1}"
                    if error:
                        label += f" (failed)"
                    else:
                        label += f" → {signal}"
                    eq.put({"event": "agent_start", "agent": label, "step": step, "total": 9})
            
            if is_free_model:
                # Sequential execution to stay within rate limits (20 req/min)
                results = []
                for i in range(self.parallel_runs):
                    result = await self._run_with_timeout(
                        member_id=i,
                        ticker=ticker,
                        trade_date=trade_date,
                        entry_price=entry_price,
                        temperature=self.temperatures[i % len(self.temperatures)],
                        selected_analysts=selected_analysts,
                        model_override=current_model if attempt > 0 else None,
                    )
                    results.append(result)
                    _emit_member_done(i, result.signal, result.error)
            else:
                # Parallel execution for paid models with higher rate limits
                tasks = [
                    self._run_with_timeout(
                        member_id=i,
                        ticker=ticker,
                        trade_date=trade_date,
                        entry_price=entry_price,
                        temperature=self.temperatures[i % len(self.temperatures)],
                        selected_analysts=selected_analysts,
                        model_override=current_model if attempt > 0 else None,
                    )
                    for i in range(self.parallel_runs)
                ]
                results = await asyncio.gather(*tasks)
                for i, r in enumerate(results):
                    _emit_member_done(i, r.signal, r.error)
            
            # Filter out timeouts/errors
            valid_results = [r for r in results if r.error is None]
            error_results = [r for r in results if r.error is not None]
            
            if error_results:
                logger.warning(f"{len(error_results)} ensemble members failed: "
                             f"{[r.error for r in error_results]}")
            
            min_required = min(2, self.parallel_runs)
            if len(valid_results) < min_required:
                logger.warning(f"Only {len(valid_results)} valid results, need at least {min_required} for consensus")
                if attempt < self.max_retries:
                    continue
                else:
                    logger.error("Max retries reached, returning partial consensus")
            
            # Convert to dict format for consensus engine
            result_dicts = [
                {
                    "signal": r.signal,
                    "confidence": r.confidence,
                    "stop_loss_price": r.stop_loss_price,
                    "take_profit_price": r.take_profit_price,
                    "max_hold_days": r.max_hold_days,
                    "reasoning": r.reasoning,
                }
                for r in valid_results
            ]
            
            # Compute consensus
            consensus = self.consensus_engine.compute_consensus(
                results=result_dicts,
                entry_price=entry_price,
                ticker=ticker,
            )
            
            # BLOCKER FIX: Check staleness
            elapsed = time.time() - ensemble_start_time
            if elapsed > self.max_ensemble_time:
                consensus.ensemble_metadata["stale_price_warning"] = True
                logger.warning(f"Ensemble took {elapsed:.1f}s, prices may be stale")
            
            # Record retry count
            consensus.ensemble_metadata["retry_count"] = attempt
            
            # Check divergence
            if attempt < self.max_retries:
                should_retry = self.consensus_engine.should_rerun(
                    result_dicts,
                    confidence_range_threshold=self.config.get(
                        "ensemble_divergence_range_threshold", 0.20
                    ),
                )
                if should_retry:
                    logger.info("Divergence detected, triggering retry...")
                    continue
            
            return consensus
        
        # Should never reach here, but return last consensus just in case
        return consensus
    
    async def _run_with_timeout(
        self,
        member_id: int,
        ticker: str,
        trade_date: str,
        entry_price: float,
        temperature: float,
        selected_analysts: Optional[List[str]],
        model_override: Optional[str] = None,
    ) -> EnsembleMemberResult:
        """Run single ensemble member with timeout protection.
        
        BLOCKER FIX: Wraps execution in asyncio.wait_for() with 300s timeout.
        
        Args:
            member_id: Unique ID for this ensemble member
            ticker: Ticker symbol
            trade_date: Trade date
            entry_price: Snapshot entry price
            temperature: LLM temperature for this run
            selected_analysts: List of analyst types
            model_override: Optional different model for retry diversity
            
        Returns:
            EnsembleMemberResult with signal or error
        """
        try:
            return await asyncio.wait_for(
                self._run_single(
                    member_id=member_id,
                    ticker=ticker,
                    trade_date=trade_date,
                    entry_price=entry_price,
                    temperature=temperature,
                    selected_analysts=selected_analysts,
                    model_override=model_override,
                ),
                timeout=self.timeout_per_run,
            )
        except asyncio.TimeoutError:
            logger.error(f"Ensemble member {member_id} timed out after {self.timeout_per_run}s")
            return EnsembleMemberResult(
                member_id=member_id,
                signal="HOLD",
                confidence=0.5,
                stop_loss_price=None,
                take_profit_price=None,
                max_hold_days=3,
                reasoning="",
                error=f"timeout_after_{self.timeout_per_run}s",
            )
        except Exception as e:
            logger.error(f"Ensemble member {member_id} failed: {e}")
            return EnsembleMemberResult(
                member_id=member_id,
                signal="HOLD",
                confidence=0.5,
                stop_loss_price=None,
                take_profit_price=None,
                max_hold_days=3,
                reasoning="",
                error=str(e),
            )
    
    async def _run_single(
        self,
        member_id: int,
        ticker: str,
        trade_date: str,
        entry_price: float,
        temperature: float,
        selected_analysts: Optional[List[str]],
        model_override: Optional[str] = None,
    ) -> EnsembleMemberResult:
        """Execute single ensemble member analysis.
        
        BLOCKER FIX: Runs blocking graph.propagate() in thread pool to not block async loop.
        
        Args:
            member_id: Unique ID for this ensemble member
            ticker: Ticker symbol
            trade_date: Trade date
            entry_price: Snapshot entry price (for validation)
            temperature: LLM temperature for this run
            selected_analysts: List of analyst types
            model_override: Optional different model for retry diversity
            
        Returns:
            EnsembleMemberResult with signal data
        """
        import concurrent.futures
        
        # Create graph instance with modified temperature
        member_config = self.config.copy()
        member_config["llm_temperature"] = temperature
        
        # Use override model if provided (for retry diversity)
        if model_override:
            member_config["deep_think_llm"] = model_override
        
        # Create graph
        analysts = selected_analysts or ["market", "social", "news", "fundamentals"]
        graph = TradingAgentsGraph(
            selected_analysts=analysts,
            config=member_config,
        )
        
        # Rebuild graph with crypto tools if needed
        from tradingagents.dataflows.asset_detection import is_crypto
        if is_crypto(ticker):
            graph._rebuild_graph_for_asset(ticker, analysts)
        
        # BLOCKER FIX: Run blocking propagate in thread pool with retry for 429s
        logger.info(f"Ensemble member {member_id}: starting graph.propagate for {ticker}")
        print(f"Ensemble member {member_id}: starting graph.propagate for {ticker}", flush=True)
        loop = asyncio.get_event_loop()
        
        max_propagate_retries = 5
        for retry in range(max_propagate_retries):
            try:
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    final_state, signal_word = await loop.run_in_executor(
                        pool, graph.propagate, ticker, trade_date
                    )
                break  # Success
            except Exception as e:
                if "429" in str(e) and retry < max_propagate_retries - 1:
                    wait = 30 * (retry + 1)  # 30s, 60s, 90s, 120s
                    print(f"Ensemble member {member_id}: rate limited, waiting {wait}s (retry {retry + 1}/{max_propagate_retries})", flush=True)
                    await asyncio.sleep(wait)
                    # Rebuild graph for fresh state
                    graph = TradingAgentsGraph(
                        selected_analysts=analysts,
                        config=member_config,
                    )
                    if is_crypto(ticker):
                        graph._rebuild_graph_for_asset(ticker, analysts)
                else:
                    raise
        
        logger.info(f"Ensemble member {member_id}: propagate complete, signal={signal_word}")
        print(f"Ensemble member {member_id}: propagate complete, signal={signal_word}", flush=True)
        
        # Extract structured signal if available
        processed_signal = graph.process_signal(final_state.get("final_trade_decision", ""))
        
        if isinstance(processed_signal, dict):
            return EnsembleMemberResult(
                member_id=member_id,
                signal=processed_signal.get("signal", signal_word),
                confidence=processed_signal.get("confidence", 0.5),
                stop_loss_price=processed_signal.get("stop_loss_price"),
                take_profit_price=processed_signal.get("take_profit_price"),
                max_hold_days=processed_signal.get("max_hold_days", 3),
                reasoning=processed_signal.get("reasoning", ""),
            )
        else:
            # Fallback to extracting from text
            return EnsembleMemberResult(
                member_id=member_id,
                signal=signal_word,
                confidence=0.5,  # Default when no structured signal
                stop_loss_price=None,
                take_profit_price=None,
                max_hold_days=3,
                reasoning=final_state.get("final_trade_decision", ""),
            )
    
    def _get_current_price(self, ticker: str) -> float:
        """Get current price for ticker for ensemble snapshot.
        
        Uses Hyperliquid live spot for crypto (zero-cache), yfinance for equities.
        Falls back to yfinance if Hyperliquid fails.
        """
        from tradingagents.dataflows.asset_detection import is_crypto
        
        # Try Hyperliquid first for crypto (live mid price, not yesterday's close)
        if is_crypto(ticker):
            try:
                from tradingagents.dataflows.hyperliquid_client import HyperliquidClient
                hl = HyperliquidClient()
                base = ticker.replace("-USD", "").replace("USDT", "").upper()
                price = hl.get_spot_price(base, max_age_override=0)
                if price and price > 0:
                    return price
            except Exception as e:
                logger.warning(f"Hyperliquid spot price failed for {ticker}, falling back to yfinance: {e}")
        
        try:
            import yfinance as yf
            data = yf.download(ticker, period="1d", progress=False)
            if data.empty:
                raise ValueError("No data returned")
            close = data["Close"].iloc[-1]
            return float(close.iloc[0]) if hasattr(close, 'iloc') else float(close)
        except Exception as e:
            logger.warning(f"Could not get current price for {ticker}: {e}")
            return 100.0


def should_use_ensemble(config: Dict[str, Any], provider: str) -> bool:
    """Check if ensemble mode should be used for this provider.
    
    Args:
        config: Application configuration
        provider: LLM provider name
        
    Returns:
        True if ensemble should be used
    """
    # User's explicit toggle takes priority
    if not config.get("enable_ensemble", False):
        return False
    
    # Explicitly disabled providers
    disabled = config.get("ensemble_disabled_providers", ["deepseek"])
    if provider.lower() in disabled:
        return False
    
    # Default to global setting
    return True
