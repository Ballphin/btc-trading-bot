"""Decision-replay backtesting engine.

Runs the full TradingAgents pipeline across a date range of historical data,
logs each day's decision, and computes performance metrics.
"""

import logging
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import yfinance as yf

from tradingagents.backtesting.context import BACKTEST_MODE

from tradingagents.backtesting.portfolio import Portfolio
from tradingagents.backtesting.metrics import compute_metrics
from tradingagents.backtesting.report import generate_report
from tradingagents.backtesting.indicators import calculate_atr, calculate_volatility
from tradingagents.dataflows.asset_detection import is_crypto

logger = logging.getLogger(__name__)


_PRICE_CACHE: Dict[str, Dict] = {}
_FUNDING_CACHE: Dict[str, Dict] = {}
_INTRADAY_CACHE: Dict[str, Dict] = {}  # ticker -> {date_str -> [{t, o, h, l, c, v}, ...]}
_CACHE_LOCK = threading.RLock()
_CACHE_TTL: Dict[str, datetime] = {}
_CACHE_MAX_AGE_HOURS = 24

def _preload_chunk_if_needed(ticker: str, date_str: str):
    global _PRICE_CACHE, _FUNDING_CACHE, _INTRADAY_CACHE, _CACHE_TTL
    from tradingagents.dataflows.asset_detection import is_crypto
    from tradingagents.dataflows.hyperliquid_client import HyperliquidClient

    with _CACHE_LOCK:
        # Evict stale cache entries older than 24 hours
        if ticker in _CACHE_TTL:
            age = (datetime.now() - _CACHE_TTL[ticker]).total_seconds() / 3600
            if age > _CACHE_MAX_AGE_HOURS:
                _PRICE_CACHE.pop(ticker, None)
                _FUNDING_CACHE.pop(ticker, None)
                _INTRADAY_CACHE.pop(ticker, None)
                del _CACHE_TTL[ticker]
                logger.debug(f"Evicted stale price cache for {ticker} ({age:.1f}h old)")

        if ticker not in _PRICE_CACHE:
            _PRICE_CACHE[ticker] = {}
        
    date_fmt = "%Y-%m-%d %H:%M:%S" if " " in date_str else "%Y-%m-%d"
    dt = datetime.strptime(date_str, date_fmt)

    with _CACHE_LOCK:
        _CACHE_TTL.setdefault(ticker, datetime.now())

    # Check if we have data within 3 days of target
    closest_distance = float('inf')
    for d_str in _PRICE_CACHE.get(ticker, {}).keys():
        d_fmt = "%Y-%m-%d %H:%M:%S" if " " in d_str else "%Y-%m-%d"
        d_dt = datetime.strptime(d_str, d_fmt)
        diff = abs((d_dt - dt).days)
        if diff < closest_distance:
            closest_distance = diff
            
    if closest_distance <= 3:
        return  # Cache hit within acceptable range (no need to fetch chunk)
        
    # Cache miss or entirely out of bounds -> fetch next chunk
    try:
        is_c = is_crypto(ticker)
        if is_c:
            base_asset = ticker.replace("-USD", "").upper()
            interval = "4h" if " " in date_str else "1d"
            fetch_start = (dt - timedelta(days=5)).strftime("%Y-%m-%d")
            fetch_end = (dt + timedelta(days=120)).strftime("%Y-%m-%d")

            hl = HyperliquidClient()
            df = hl.get_ohlcv(base_asset, interval, fetch_start, fetch_end)
            if not df.empty:
                with _CACHE_LOCK:
                    for _, row in df.iterrows():
                        c_dt = row["timestamp"].to_pydatetime()
                        c_str = c_dt.strftime("%Y-%m-%d %H:%M:%S") if " " in date_str else c_dt.strftime("%Y-%m-%d")
                        _PRICE_CACHE[ticker][c_str] = float(row["close"])

            # Also fetch 1H candles for intraday context (MEDIUM 9)
            try:
                intraday_df = hl.get_ohlcv(base_asset, "1h", fetch_start, fetch_end)
                if not intraday_df.empty:
                    with _CACHE_LOCK:
                        if ticker not in _INTRADAY_CACHE:
                            _INTRADAY_CACHE[ticker] = {}
                        for _, row in intraday_df.iterrows():
                            day_key = row["timestamp"].strftime("%Y-%m-%d")
                            if day_key not in _INTRADAY_CACHE[ticker]:
                                _INTRADAY_CACHE[ticker][day_key] = []
                            _INTRADAY_CACHE[ticker][day_key].append({
                                "t": row["timestamp"].isoformat(),
                                "o": float(row["open"]),
                                "h": float(row["high"]),
                                "l": float(row["low"]),
                                "c": float(row["close"]),
                                "v": float(row["volume"]),
                            })
            except Exception as e:
                logger.debug(f"Intraday cache preload failed for {ticker}: {e}")

            # Also fetch funding history chunk
            fund_df = hl.get_funding_history(base_asset, fetch_start, fetch_end)
            if ticker not in _FUNDING_CACHE:
                _FUNDING_CACHE[ticker] = {}
            if not fund_df.empty:
                with _CACHE_LOCK:
                    for _, row in fund_df.iterrows():
                        f_dt = row["timestamp"].to_pydatetime()
                        f_str = f_dt.strftime("%Y-%m-%d %H:%M:%S") if " " in date_str else f_dt.strftime("%Y-%m-%d")
                        _FUNDING_CACHE[ticker][f_str] = float(row["funding_rate"])

        else:
            fetch_start = dt - timedelta(days=5)
            fetch_end = dt + timedelta(days=365)
            try:
                data = yf.download(
                    ticker,
                    start=fetch_start.strftime("%Y-%m-%d"),
                    end=fetch_end.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                )
                if not data.empty:
                    data.index = data.index.tz_localize(None) if data.index.tz else data.index
                    with _CACHE_LOCK:
                        for index, row in data.iterrows():
                            d_str = index.strftime("%Y-%m-%d")
                            close = row["Close"]
                            price = float(close.iloc[0]) if hasattr(close, 'iloc') else float(close)
                            _PRICE_CACHE[ticker][d_str] = price
            except Exception as e:
                logger.warning(f"yf.download failed: {e}")
                
    except Exception as e:
        logger.warning(f"chunk fetch failed for {ticker}: {e}")

def _get_price_on_date(ticker: str, date_str: str) -> Optional[float]:
    """Fetch closing price for a ticker on a specific date (powered by robust Hyperliquid/YF Cache)."""
    _preload_chunk_if_needed(ticker, date_str)
    
    if date_str in _PRICE_CACHE.get(ticker, {}):
        return _PRICE_CACHE[ticker][date_str]
        
    # Resolve closest prior date
    date_fmt = "%Y-%m-%d %H:%M:%S" if " " in date_str else "%Y-%m-%d"
    dt = datetime.strptime(date_str, date_fmt)
    closest_price = None
    closest_dt = None
    
    for d_str, price in _PRICE_CACHE.get(ticker, {}).items():
        d_fmt = "%Y-%m-%d %H:%M:%S" if " " in d_str else "%Y-%m-%d"
        d_dt = datetime.strptime(d_str, d_fmt)
        if d_dt <= dt:
            if closest_dt is None or d_dt > closest_dt:
                closest_dt = d_dt
                closest_price = price
                
    if closest_price is not None and (dt - closest_dt).days <= 5:
        return closest_price
    return None

def get_intraday_candles(ticker: str, date_str: str) -> List[dict]:
    """Return cached 1H candles for a given date. Used by simulation context."""
    _preload_chunk_if_needed(ticker, date_str)
    base_date = date_str.split(" ")[0]
    with _CACHE_LOCK:
        return _INTRADAY_CACHE.get(ticker, {}).get(base_date, [])


def _get_funding_on_date(ticker: str, date_str: str) -> Optional[float]:
    """Fetch exact historical funding rate from Hyperliquid Cache if available."""
    if ticker in _FUNDING_CACHE:
        # Resolve closest prior date within 12 hours
        date_fmt = "%Y-%m-%d %H:%M:%S" if " " in date_str else "%Y-%m-%d"
        dt = datetime.strptime(date_str, date_fmt)
        closest_rate = None
        closest_dt = None
        
        for d_str, rate in _FUNDING_CACHE[ticker].items():
            d_fmt = "%Y-%m-%d %H:%M:%S" if " " in d_str else "%Y-%m-%d"
            d_dt = datetime.strptime(d_str, d_fmt)
            if d_dt <= dt:
                if closest_dt is None or d_dt > closest_dt:
                    closest_dt = d_dt
                    closest_rate = rate
                    
        if closest_rate is not None and (dt - closest_dt).total_seconds() <= 12 * 3600:
            return closest_rate
    return None


def _generate_trade_dates(
    start_date: str,
    end_date: str,
    frequency: str = "weekly",
    ticker: str = "",
) -> List[str]:
    """Generate a list of trading dates between start and end."""
    start = datetime.strptime(start_date.split(" ")[0], "%Y-%m-%d")
    end = datetime.strptime(end_date.split(" ")[0], "%Y-%m-%d")

    dates = []
    current = start

    if frequency == "daily":
        step = timedelta(days=1)
    elif frequency == "4h":
        step = timedelta(hours=4)  # Only valid in replay mode; capped to 90 dates by caller
    elif frequency == "weekly":
        step = timedelta(weeks=1)
    elif frequency == "biweekly":
        step = timedelta(weeks=2)
    elif frequency == "monthly":
        step = timedelta(days=30)
    else:
        step = timedelta(weeks=1)

    while current <= end:
        # Skip weekends for equity (crypto trades 24/7)
        date_str = current.strftime("%Y-%m-%d %H:%M:%S") if frequency == "4h" else current.strftime("%Y-%m-%d")
        
        if is_crypto(ticker):
            dates.append(date_str)
        elif current.weekday() < 5:
            dates.append(date_str)
        current += step

    return dates


class BacktestEngine:
    """
    Decision-replay backtesting engine.

    Runs the full TradingAgents agent pipeline on each date in the range,
    captures the signal, processes it through the portfolio, and computes metrics.
    """

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        config: Dict[str, Any] = None,
        initial_capital: float = 100_000.0,
        position_size_pct: float = 0.25,
        trading_frequency: str = "weekly",
        selected_analysts: List[str] = None,
        debug: bool = False,
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.config = config
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.trading_frequency = trading_frequency
        self.selected_analysts = selected_analysts or [
            "market", "social", "news", "fundamentals"
        ]
        self.debug = debug

        # Extract portfolio config with defaults
        leverage = config.get("leverage", 1.0) if config else 1.0
        maker_fee = config.get("maker_fee", 0.0002) if config else 0.0002
        taker_fee = config.get("taker_fee", 0.0005) if config else 0.0005
        use_funding = config.get("use_funding", True) if config else True
        position_sizing = config.get("position_sizing", "fixed") if config else "fixed"
        risk_per_trade = config.get("risk_per_trade", 0.01) if config else 0.01
        
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            leverage=leverage,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            use_funding=use_funding,
            position_sizing=position_sizing,
            risk_per_trade=risk_per_trade,
        )
        self.decisions: List[Dict] = []
        self.errors: List[Dict] = []

    def run(self) -> Dict[str, Any]:
        """
        Execute the full backtest.

        Returns:
            Dict with keys: metrics, decisions, equity_curve, report_path
        """
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        logger.info(
            f"Starting backtest: {self.ticker} from {self.start_date} to {self.end_date} "
            f"({self.trading_frequency})"
        )

        # Generate trade dates
        trade_dates = _generate_trade_dates(
            self.start_date, self.end_date, self.trading_frequency, self.ticker
        )

        # HIGH 10: 4h in full-simulation = too many LLM calls; enforce hard cap of 90 dates
        if self.trading_frequency == "4h" and len(trade_dates) > 90:
            logger.warning(
                f"4h frequency generated {len(trade_dates)} dates — capping at 90 to prevent excessive LLM calls"
            )
            trade_dates = trade_dates[:90]

        logger.info(f"Generated {len(trade_dates)} trading dates")

        if not trade_dates:
            logger.error("No trading dates generated")
            return {"metrics": {}, "decisions": [], "equity_curve": []}

        # Initialize the agent graph with backtest_mode to disable realtime tools
        self.config["backtest_mode"] = True
        graph = TradingAgentsGraph(
            selected_analysts=self.selected_analysts,
            debug=self.debug,
            config=self.config,
        )

        # Load persisted memories — filter to only memories saved before start_date
        # to prevent look-ahead bias (BLOCKER 5)
        graph.load_memories_for_ticker(self.ticker, before_date=self.start_date)

        # Rebuild graph for asset type if needed
        if is_crypto(self.ticker):
            graph._rebuild_graph_for_asset(self.ticker, self.selected_analysts)

        # Run the pipeline on each date
        BACKTEST_MODE.set(True)
        for i, trade_date in enumerate(trade_dates):
            logger.info(f"[{i+1}/{len(trade_dates)}] Processing {trade_date}...")

            # Get price for this date
            price = _get_price_on_date(self.ticker, trade_date)
            if price is None:
                logger.warning(f"No price data for {trade_date}, skipping")
                continue

            try:
                # Run the full agent pipeline
                final_state, signal = graph.propagate(self.ticker, trade_date)

                # Extract structured signal fields if available
                stop_loss_price = final_state.get("stop_loss_price")
                take_profit_price = final_state.get("take_profit_price")
                max_hold_days = final_state.get("max_hold_days")
                confidence = final_state.get("confidence")

                # Kelly position sizing via ConfidenceScorer (BLOCKER 4)
                kelly_size = self.position_size_pct
                try:
                    from tradingagents.graph.confidence import ConfidenceScorer
                    from tradingagents.backtesting.regime import detect_regime_context
                    scorer = ConfidenceScorer()
                    regime_ctx = detect_regime_context(self.ticker, trade_date)
                    scored = scorer.score(
                        ticker=self.ticker,
                        signal=signal,
                        llm_confidence=confidence if confidence is not None else 0.50,
                        regime_ctx=regime_ctx,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                    )
                    if scored.get("gated"):
                        signal = "HOLD"
                        logger.debug(f"Signal gated by ConfidenceScorer on {trade_date}")
                    else:
                        kelly_size = scored.get("position_size_pct", self.position_size_pct)
                except Exception as e:
                    logger.warning(f"Kelly sizing skipped on {trade_date}: {e}")

                # Calculate dynamic ATR and volatility for advanced position sizing
                atr = None
                volatility = None
                if self.portfolio.position_sizing in ("atr_risk", "volatility"):
                    atr = calculate_atr(self.ticker, trade_date, period=14)
                    volatility = calculate_volatility(self.ticker, trade_date, period=20)
                    if atr:
                        logger.debug(f"ATR for {self.ticker} on {trade_date}: {atr:.2f}")
                    if volatility:
                        logger.debug(f"Volatility for {self.ticker} on {trade_date}: {volatility:.4f}")

                # Apply Kelly-derived position size for this trade only
                _orig_size = self.portfolio.position_size_pct
                self.portfolio.position_size_pct = kelly_size

                # Process signal through portfolio with risk parameters
                funding_rate = _get_funding_on_date(self.ticker, trade_date)
                action = self.portfolio.process_signal(
                    signal, 
                    price, 
                    trade_date, 
                    funding_rate=funding_rate,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    max_hold_days=max_hold_days,
                    atr=atr,
                )
                self.portfolio.position_size_pct = _orig_size  # restore after each trade

                # Record the decision with structured fields
                decision = {
                    "date": trade_date,
                    "price": price,
                    "signal": signal,
                    "action": action,
                    "portfolio_value": self.portfolio.portfolio_value(price),
                    "position": self.portfolio.position_side.value,
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price,
                    "confidence": confidence,
                    "max_hold_days": max_hold_days,
                    "kelly_size": kelly_size,
                }
                self.decisions.append(decision)

                # Compute returns for reflection
                if len(self.decisions) >= 2:
                    prev = self.decisions[-2]
                    returns = (price - prev["price"]) / prev["price"]
                    returns_str = f"{returns:+.2%} since last decision ({prev['date']})"
                    try:
                        graph.reflect_and_remember(returns_str)
                    except Exception as e:
                        logger.warning(f"Reflection failed on {trade_date}: {e}")

                logger.info(
                    f"  Signal={signal} | Action={action} | "
                    f"Portfolio=${self.portfolio.portfolio_value(price):,.2f}"
                )

            except Exception as e:
                logger.error(f"Error on {trade_date}: {e}")
                self.errors.append({"date": trade_date, "error": str(e)})
                continue

        # Reset backtest mode after the loop
        BACKTEST_MODE.set(False)

        # Force close any open position at end
        if trade_dates:
            final_price = _get_price_on_date(self.ticker, trade_dates[-1])
            if final_price:
                self.portfolio.force_close(final_price, trade_dates[-1])

        # Save agent memories to disk for cross-session learning
        try:
            results_dir = self.config.get("results_dir", "./eval_results") if self.config else "./eval_results"
            graph.bull_memory.save_to_disk(self.ticker, results_dir)
            graph.bear_memory.save_to_disk(self.ticker, results_dir)
            graph.trader_memory.save_to_disk(self.ticker, results_dir)
            graph.invest_judge_memory.save_to_disk(self.ticker, results_dir)
            graph.portfolio_manager_memory.save_to_disk(self.ticker, results_dir)
            logger.info(f"Agent memories saved to {results_dir}/{self.ticker}/agent_memory/")
        except Exception as e:
            logger.warning(f"Failed to save agent memories: {e}")

        # Refresh backtest knowledge store lessons with latest results
        try:
            graph.backtest_knowledge_store.refresh_lessons(self.ticker)
            logger.info(f"Backtest lessons refreshed for {self.ticker}")
        except Exception as e:
            logger.warning(f"Failed to refresh backtest lessons: {e}")

        # Compute benchmark return for alpha calculation (HIGH 7)
        benchmark_return_pct = None
        try:
            start_price = _get_price_on_date(self.ticker, trade_dates[0])
            end_price = _get_price_on_date(self.ticker, trade_dates[-1])
            if start_price and end_price and start_price > 0:
                benchmark_return_pct = (end_price - start_price) / start_price * 100
                logger.info(f"Benchmark (buy & hold) return: {benchmark_return_pct:+.2f}%")
        except Exception as e:
            logger.warning(f"Benchmark return calculation failed: {e}")

        # Compute metrics (BLOCKER 2: pass is_crypto for correct trading_days_per_year)
        metrics = compute_metrics(
            equity_curve=self.portfolio.equity_curve,
            closed_positions=self.portfolio.closed_positions,
            initial_capital=self.initial_capital,
            total_fees=self.portfolio.total_fees_paid,
            total_funding=self.portfolio.total_funding_paid,
            liquidations=self.portfolio.liquidations,
            leverage=self.portfolio.leverage,
            stops_hit=self.portfolio.stops_hit,
            takes_hit=self.portfolio.takes_hit,
            is_crypto=is_crypto(self.ticker),
            benchmark_return_pct=benchmark_return_pct,
        )

        # Generate report
        report_path = self._save_results(metrics)

        return {
            "metrics": metrics,
            "decisions": self.decisions,
            "equity_curve": self.portfolio.equity_curve,
            "errors": self.errors,
            "report_path": report_path,
        }

    def _save_results(self, metrics: Dict) -> str:
        """Save backtest results to disk."""
        results_dir = Path(
            self.config.get("results_dir", "./eval_results") if self.config else "./eval_results"
        )
        backtest_dir = results_dir / "backtests" / self.ticker
        backtest_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results JSON
        results_file = backtest_dir / f"backtest_{timestamp}.json"
        results_data = {
            "config": {
                "ticker": self.ticker,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "frequency": self.trading_frequency,
                "initial_capital": self.initial_capital,
                "position_size_pct": self.position_size_pct,
            },
            "metrics": metrics,
            "decisions": self.decisions,
            "equity_curve": self.portfolio.equity_curve,
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
                    "liquidation_price": t.liquidation_price,
                    "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit,
                    "hold_days": t.hold_days,
                    "exit_reason": t.exit_reason,
                    "atr_at_entry": t.atr_at_entry,
                }
                for t in self.portfolio.trade_history
            ],
            "errors": self.errors,
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        # Generate markdown report
        report_file = backtest_dir / f"report_{timestamp}.md"
        report = generate_report(
            ticker=self.ticker,
            metrics=metrics,
            decisions=self.decisions,
            equity_curve=self.portfolio.equity_curve,
            config={
                "start_date": self.start_date,
                "end_date": self.end_date,
                "frequency": self.trading_frequency,
                "initial_capital": self.initial_capital,
            },
        )
        report_file.write_text(report)

        logger.info(f"Results saved to {backtest_dir}")
        return str(report_file)
