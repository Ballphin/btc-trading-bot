"""Decision-replay backtesting engine.

Runs the full TradingAgents pipeline across a date range of historical data,
logs each day's decision, and computes performance metrics.
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import yfinance as yf

from tradingagents.backtesting.portfolio import Portfolio
from tradingagents.backtesting.metrics import compute_metrics
from tradingagents.backtesting.report import generate_report
from tradingagents.backtesting.indicators import calculate_atr, calculate_volatility
from tradingagents.dataflows.asset_detection import is_crypto

logger = logging.getLogger(__name__)


_PRICE_CACHE = {}
_FUNDING_CACHE = {}

def _preload_chunk_if_needed(ticker: str, date_str: str):
    global _PRICE_CACHE
    import requests
    from tradingagents.dataflows.asset_detection import is_crypto
    
    if ticker not in _PRICE_CACHE:
        _PRICE_CACHE[ticker] = {}
        
    date_fmt = "%Y-%m-%d %H:%M:%S" if " " in date_str else "%Y-%m-%d"
    dt = datetime.strptime(date_str, date_fmt)
    
    # Check if we have data within 3 days of target
    closest_distance = float('inf')
    for d_str in _PRICE_CACHE[ticker].keys():
        d_fmt = "%Y-%m-%d %H:%M:%S" if " " in d_str else "%Y-%m-%d"
        d_dt = datetime.strptime(d_str, d_fmt)
        diff = abs((d_dt - dt).days)
        if diff < closest_distance:
            closest_distance = diff
            
    if closest_distance <= 3:
        return # Cache hit within acceptable range (no need to fetch chunk)
        
    # Cache miss or entirely out of bounds -> fetch next chunk (~1 year API max)
    try:
        is_c = is_crypto(ticker)
        if is_c:
            base_asset = ticker.replace("-USD", "").upper()
            start_ts = int((dt - timedelta(days=5)).timestamp() * 1000)
            end_ts = int((dt + timedelta(days=120)).timestamp() * 1000) # 120 days to stay under 5k candle limit
            
            interval = "4h" if " " in date_str else "1d"
            url = "https://api.hyperliquid.xyz/info"
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": base_asset,
                    "interval": interval,
                    "startTime": start_ts,
                    "endTime": end_ts
                }
            }
            resp = requests.post(url, json=payload)
            candles = resp.json()
            if isinstance(candles, list):
                for c in candles:
                    c_dt = datetime.fromtimestamp(c["t"] / 1000.0)
                    c_str = c_dt.strftime("%Y-%m-%d %H:%M:%S") if " " in date_str else c_dt.strftime("%Y-%m-%d")
                    _PRICE_CACHE[ticker][c_str] = float(c["c"])
                    
            # Also fetch funding history chunk
            fund_payload = {
                "type": "fundingHistory",
                "coin": base_asset,
                "startTime": start_ts,
                "endTime": end_ts
            }
            f_resp = requests.post(url, json=fund_payload)
            funding_data = f_resp.json()
            if ticker not in _FUNDING_CACHE:
                _FUNDING_CACHE[ticker] = {}
            if isinstance(funding_data, list):
                for f in funding_data:
                    f_dt = datetime.fromtimestamp(f["time"] / 1000.0)
                    f_str = f_dt.strftime("%Y-%m-%d %H:%M:%S") if " " in date_str else f_dt.strftime("%Y-%m-%d")
                    _FUNDING_CACHE[ticker][f_str] = float(f["fundingRate"])
                
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
        step = timedelta(hours=4)
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
        logger.info(f"Generated {len(trade_dates)} trading dates")

        if not trade_dates:
            logger.error("No trading dates generated")
            return {"metrics": {}, "decisions": [], "equity_curve": []}

        # Initialize the agent graph
        graph = TradingAgentsGraph(
            selected_analysts=self.selected_analysts,
            debug=self.debug,
            config=self.config,
        )

        # Rebuild graph for asset type if needed
        if is_crypto(self.ticker):
            graph._rebuild_graph_for_asset(self.ticker, self.selected_analysts)

        # Run the pipeline on each date
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

        # Force close any open position at end
        if trade_dates:
            final_price = _get_price_on_date(self.ticker, trade_dates[-1])
            if final_price:
                self.portfolio.force_close(final_price, trade_dates[-1])

        # Compute metrics
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
            self.config.get("results_dir", "./results") if self.config else "./results"
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
