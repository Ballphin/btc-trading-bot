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
from tradingagents.dataflows.asset_detection import is_crypto

logger = logging.getLogger(__name__)


def _get_price_on_date(ticker: str, date_str: str) -> Optional[float]:
    """Fetch closing price for a ticker on a specific date."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        start = dt - timedelta(days=5)
        end = dt + timedelta(days=1)
        data = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            return None
        # Get the closest date <= target
        data.index = data.index.tz_localize(None) if data.index.tz else data.index
        mask = data.index <= dt
        if mask.any():
            close = data.loc[mask, "Close"].iloc[-1]
            return float(close.iloc[0]) if hasattr(close, 'iloc') else float(close)
        return float(data["Close"].iloc[-1].iloc[0]) if hasattr(data["Close"].iloc[-1], 'iloc') else float(data["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Could not fetch price for {ticker} on {date_str}: {e}")
        return None


def _generate_trade_dates(
    start_date: str,
    end_date: str,
    frequency: str = "weekly",
) -> List[str]:
    """Generate a list of trading dates between start and end."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start

    if frequency == "daily":
        step = timedelta(days=1)
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
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        elif is_crypto(start_date):
            # For crypto, this doesn't apply — but we need a ticker, not a date
            dates.append(current.strftime("%Y-%m-%d"))
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

        self.portfolio = Portfolio(initial_capital, position_size_pct)
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
            self.start_date, self.end_date, self.trading_frequency
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

                # Process signal through portfolio
                action = self.portfolio.process_signal(signal, price, trade_date)

                # Record the decision
                decision = {
                    "date": trade_date,
                    "price": price,
                    "signal": signal,
                    "action": action,
                    "portfolio_value": self.portfolio.portfolio_value(price),
                    "position": self.portfolio.position_side.value,
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
