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


def _is_fatal_provider_error(msg: Optional[str]) -> bool:
    """Auth / configuration failures that should abort immediately (not rate limits)."""
    if not msg:
        return False
    m = str(msg).lower()
    needles = (
        "401",
        "403",
        "invalid api key",
        "incorrect api key",
        "unauthorized",
        "authentication",
        "access denied",
        "forbidden",
    )
    return any(n in m for n in needles)


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
        self.timeout_per_run_sequential = config.get("ensemble_timeout_per_run_sequential", 300)
        self.sequential_fallback_runs = max(
            1, int(config.get("ensemble_sequential_fallback_runs", 1))
        )
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
        entry_price = self._get_current_price(ticker)
        ensemble_start_time = time.time()

        force_sequential = False

        for attempt in range(self.max_retries + 1):
            logger.info(f"Ensemble attempt {attempt + 1}/{self.max_retries + 1} for {ticker}")

            current_model = self.model
            if attempt > 0:
                fallback = self.config.get("openrouter_fallback_model")
                if fallback:
                    current_model = fallback
                    logger.info(f"Retry {attempt}: Switching to fallback model {current_model}")

            is_free_model = ":free" in self.model.lower()
            run_sequential = is_free_model or force_sequential

            if run_sequential:
                if force_sequential:
                    logger.info("Running members sequentially (rate-limit fallback)")
                runs_this_attempt = (
                    min(self.parallel_runs, self.sequential_fallback_runs)
                    if force_sequential
                    else self.parallel_runs
                )
                timeout_this_attempt = (
                    min(self.timeout_per_run, self.timeout_per_run_sequential)
                    if force_sequential
                    else self.timeout_per_run
                )
                results = []
                for i in range(runs_this_attempt):
                    result = await self._run_with_timeout(
                        member_id=i,
                        ticker=ticker,
                        trade_date=trade_date,
                        entry_price=entry_price,
                        temperature=self.temperatures[i % len(self.temperatures)],
                        selected_analysts=selected_analysts,
                        model_override=current_model if attempt > 0 else None,
                        timeout_seconds=timeout_this_attempt,
                    )
                    results.append(result)
                    if result.error:
                        logger.info(
                            "Ensemble member %s finished with error: %s",
                            i + 1,
                            result.error,
                        )
                    else:
                        logger.info(
                            "Ensemble member %s finished: %s",
                            i + 1,
                            result.signal,
                        )
                    if result.error and "timeout" in str(result.error):
                        logger.warning(f"Member {i} timed out sequentially; skipping remaining")
                        break
            else:
                tasks = [
                    self._run_with_timeout(
                        member_id=i,
                        ticker=ticker,
                        trade_date=trade_date,
                        entry_price=entry_price,
                        temperature=self.temperatures[i % len(self.temperatures)],
                        selected_analysts=selected_analysts,
                        model_override=current_model if attempt > 0 else None,
                        timeout_seconds=self.timeout_per_run,
                    )
                    for i in range(self.parallel_runs)
                ]
                results = await asyncio.gather(*tasks)
                for i, r in enumerate(results):
                    if r.error:
                        logger.info("Ensemble member %s failed: %s", i + 1, r.error)
                    else:
                        logger.info("Ensemble member %s: %s", i + 1, r.signal)

            valid_results = [r for r in results if r.error is None]
            error_results = [r for r in results if r.error is not None]

            if error_results:
                logger.warning(f"{len(error_results)} ensemble members failed: "
                             f"{[r.error for r in error_results]}")

            if len(valid_results) == 0 and error_results:
                if all(_is_fatal_provider_error(r.error) for r in error_results):
                    error_samples = list({r.error for r in error_results})[:3]
                    raise RuntimeError(
                        "Analysis aborted: API credentials or access denied. "
                        f"Details: {'; '.join(str(e) for e in error_samples)}"
                    )

            timeout_count = sum(1 for r in error_results if "timeout" in str(r.error))

            if timeout_count >= 2 and not run_sequential:
                force_sequential = True
                logger.info("Multiple timeouts detected; switching to sequential for next attempt")

            min_required = (
                1 if run_sequential and force_sequential else min(2, self.parallel_runs)
            )
            if len(valid_results) < min_required:
                logger.warning(f"Only {len(valid_results)} valid results, need at least {min_required} for consensus")
                if attempt < self.max_retries:
                    continue
                logger.warning(
                    "Returning empty consensus after retries (rate limits / timeouts / partial failures)"
                )
                # Plan Part 4.1: pass through per-member errors so UI can
                # show the concrete reason instead of a generic message.
                member_errs = [str(r.error) for r in error_results if r.error]
                return self.consensus_engine.compute_consensus(
                    [],
                    entry_price=entry_price,
                    ticker=ticker,
                    member_errors=member_errs,
                )
            
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
        timeout_seconds: Optional[int] = None,
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
            effective_timeout = timeout_seconds or self.timeout_per_run
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
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            effective_timeout = timeout_seconds or self.timeout_per_run
            logger.error(f"Ensemble member {member_id} timed out after {effective_timeout}s")
            return EnsembleMemberResult(
                member_id=member_id,
                signal="HOLD",
                confidence=0.5,
                stop_loss_price=None,
                take_profit_price=None,
                max_hold_days=3,
                reasoning="",
                error=f"timeout_after_{effective_timeout}s",
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
        
        # Create graph (stream progress on SSE for single run or primary member only)
        analysts = selected_analysts or ["market", "social", "news", "fundamentals"]
        eq = getattr(self, "_event_queue", None)
        use_progress = eq is not None and (
            self.parallel_runs == 1 or member_id == 0
        )
        suffix = ""
        if use_progress and self.parallel_runs > 1:
            suffix = f" (member {member_id + 1})"

        def _make_graph():
            return TradingAgentsGraph(
                selected_analysts=analysts,
                config=member_config,
                debug=False,
                progress_event_queue=eq if use_progress else None,
                progress_label_suffix=suffix,
            )

        graph = _make_graph()

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
                    graph = _make_graph()
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
