#!/usr/bin/env python3
"""CLI entry point for the TradingAgents decision-replay backtesting engine.

Usage:
    python backtest.py --ticker BTC-USD --start 2024-01-01 --end 2024-06-30
    python backtest.py --ticker NVDA --start 2024-01-01 --end 2024-06-30 --frequency weekly
    python backtest.py --ticker BTC-USD --start 2024-01-01 --end 2024-03-31 --capital 50000 --position-size 0.3
"""

import argparse
import logging
import sys

from tradingagents.backtesting.engine import BacktestEngine
from tradingagents.default_config import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="TradingAgents Decision-Replay Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py --ticker BTC-USD --start 2024-01-01 --end 2024-06-30
  python backtest.py --ticker NVDA --start 2024-01-01 --end 2024-06-30 --frequency weekly
  python backtest.py --ticker BTC-USD --start 2024-03-01 --end 2024-03-31 --capital 50000
        """,
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. BTC-USD, NVDA)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--frequency",
        default="weekly",
        choices=["daily", "weekly", "biweekly", "monthly"],
        help="Trading frequency (default: weekly)",
    )
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital (default: 100000)")
    parser.add_argument("--position-size", type=float, default=0.25, help="Position size as fraction (default: 0.25)")
    parser.add_argument(
        "--analysts",
        nargs="+",
        default=["market", "social", "news", "fundamentals"],
        help="Analyst types to include (default: all four)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with tracing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Build config
    config = DEFAULT_CONFIG.copy()

    print(f"\n{'='*60}")
    print(f"TradingAgents Backtest")
    print(f"{'='*60}")
    print(f"  Ticker:         {args.ticker}")
    print(f"  Period:         {args.start} to {args.end}")
    print(f"  Frequency:      {args.frequency}")
    print(f"  Initial Capital: ${args.capital:,.2f}")
    print(f"  Position Size:  {args.position_size*100:.0f}%")
    print(f"  Analysts:       {', '.join(args.analysts)}")
    print(f"  Debug:          {args.debug}")
    print(f"{'='*60}\n")

    # Run backtest
    engine = BacktestEngine(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        config=config,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        trading_frequency=args.frequency,
        selected_analysts=args.analysts,
        debug=args.debug,
    )

    results = engine.run()

    # Print summary
    metrics = results.get("metrics", {})
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Final Value:      ${metrics.get('final_value', 0):,.2f}")
    print(f"  Total Return:     {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Sortino Ratio:    {metrics.get('sortino_ratio', 0):.3f}")
    print(f"  Max Drawdown:     {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Win Rate:         {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"  Total Trades:     {metrics.get('total_trades', 0)}")
    print(f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
    print(f"{'='*60}")

    if results.get("report_path"):
        print(f"\n  Full report: {results['report_path']}")

    if results.get("errors"):
        print(f"\n  Errors encountered: {len(results['errors'])}")
        for err in results["errors"][:5]:
            print(f"    - {err['date']}: {err['error']}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
