"""Markdown report generation for backtest results."""

from typing import Dict, List, Any


def generate_report(
    ticker: str,
    metrics: Dict[str, Any],
    decisions: List[Dict],
    equity_curve: List[Dict],
    config: Dict[str, Any],
) -> str:
    """
    Generate a formatted markdown backtest report.

    Args:
        ticker: The instrument ticker
        metrics: Computed performance metrics dict
        decisions: List of decision records
        equity_curve: List of equity curve data points
        config: Backtest configuration

    Returns:
        Markdown-formatted report string.
    """
    lines = []

    # Header
    lines.append(f"# Backtest Report: {ticker}")
    lines.append("")
    lines.append(f"**Period**: {config.get('start_date', 'N/A')} to {config.get('end_date', 'N/A')}")
    lines.append(f"**Frequency**: {config.get('frequency', 'weekly')}")
    lines.append(f"**Initial Capital**: ${config.get('initial_capital', 100000):,.2f}")
    lines.append("")

    # Performance Summary
    lines.append("---")
    lines.append("")
    lines.append("## Performance Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Final Portfolio Value** | ${metrics.get('final_value', 0):,.2f} |")
    lines.append(f"| **Total Return** | {metrics.get('total_return_pct', 0):+.2f}% |")
    lines.append(f"| **Total P&L** | ${metrics.get('total_pnl', 0):+,.2f} |")
    lines.append(f"| **Annualized Return** | {metrics.get('annualized_return_pct', 0):+.2f}% |")
    lines.append(f"| **Annualized Volatility** | {metrics.get('annualized_volatility_pct', 0):.2f}% |")
    lines.append(f"| **Sharpe Ratio** | {metrics.get('sharpe_ratio', 0):.3f} |")
    lines.append(f"| **Sortino Ratio** | {metrics.get('sortino_ratio', 0):.3f} |")
    lines.append(f"| **Max Drawdown** | {metrics.get('max_drawdown_pct', 0):.2f}% |")
    lines.append(f"| **Calmar Ratio** | {metrics.get('calmar_ratio', 0):.3f} |")
    lines.append("")

    # Trade Statistics
    lines.append("## Trade Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Total Trades** | {metrics.get('total_trades', 0)} |")
    lines.append(f"| **Winning Trades** | {metrics.get('winning_trades', 0)} |")
    lines.append(f"| **Losing Trades** | {metrics.get('losing_trades', 0)} |")
    lines.append(f"| **Win Rate** | {metrics.get('win_rate_pct', 0):.1f}% |")
    lines.append(f"| **Average Win** | ${metrics.get('avg_win', 0):+,.2f} |")
    lines.append(f"| **Average Loss** | ${metrics.get('avg_loss', 0):+,.2f} |")
    lines.append(f"| **Profit Factor** | {metrics.get('profit_factor', 0):.2f} |")
    lines.append(f"| **Expectancy** | ${metrics.get('expectancy', 0):+,.2f} |")
    lines.append("")

    # Drawdown Details
    if metrics.get("max_drawdown_pct", 0) > 0:
        lines.append("## Maximum Drawdown")
        lines.append("")
        lines.append(f"- **Peak-to-trough**: {metrics.get('max_drawdown_pct', 0):.2f}%")
        lines.append(f"- **Start**: {metrics.get('max_drawdown_start', 'N/A')}")
        lines.append(f"- **End**: {metrics.get('max_drawdown_end', 'N/A')}")
        lines.append("")

    # Decision Log
    lines.append("## Decision Log")
    lines.append("")
    if decisions:
        lines.append("| Date | Price | Signal | Action | Portfolio Value | Position |")
        lines.append("|------|-------|--------|--------|----------------|----------|")
        for d in decisions:
            price = d.get("price", 0)
            lines.append(
                f"| {d.get('date', '')} "
                f"| ${price:,.2f} "
                f"| {d.get('signal', '')} "
                f"| {d.get('action', '')[:50]} "
                f"| ${d.get('portfolio_value', 0):,.2f} "
                f"| {d.get('position', '')} |"
            )
    else:
        lines.append("No decisions recorded.")
    lines.append("")

    # Equity Curve (text-based sparkline)
    if equity_curve and len(equity_curve) >= 2:
        lines.append("## Equity Curve")
        lines.append("")
        values = [e["portfolio_value"] for e in equity_curve]
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val if max_val != min_val else 1

        # Simple text chart (20 chars wide)
        chart_width = 40
        for e in equity_curve:
            v = e["portfolio_value"]
            bar_len = int((v - min_val) / val_range * chart_width)
            bar = "█" * bar_len
            lines.append(f"  {e['date']} | {bar} ${v:,.0f}")
        lines.append("")

    # Signal Distribution
    if decisions:
        lines.append("## Signal Distribution")
        lines.append("")
        signal_counts = {}
        for d in decisions:
            sig = d.get("signal", "UNKNOWN")
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        lines.append("| Signal | Count | Percentage |")
        lines.append("|--------|-------|-----------|")
        total = len(decisions)
        for sig, count in sorted(signal_counts.items()):
            pct = count / total * 100
            lines.append(f"| {sig} | {count} | {pct:.1f}% |")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by TradingAgents Backtesting Engine*")

    return "\n".join(lines)
