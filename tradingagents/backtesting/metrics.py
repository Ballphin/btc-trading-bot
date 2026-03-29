"""Performance metrics for backtesting: Sharpe, Sortino, drawdown, win rate, etc."""

import math
from typing import List, Dict, Any


def compute_metrics(
    equity_curve: List[dict],
    closed_positions: list,
    initial_capital: float,
    risk_free_rate: float = 0.04,
    trading_days_per_year: int = 252,
    benchmark_returns: List[float] = None,
    total_fees: float = 0.0,
    total_funding: float = 0.0,
    liquidations: int = 0,
    leverage: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute comprehensive backtest performance metrics with crypto enhancements.

    Args:
        equity_curve: List of {date, portfolio_value, ...} dicts
        closed_positions: List of closed Position objects
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate (default 4%)
        trading_days_per_year: Annualization factor
        benchmark_returns: Optional list of benchmark returns for comparison
        total_fees: Total trading fees paid
        total_funding: Total funding costs (positive = paid, negative = received)
        liquidations: Number of liquidations
        leverage: Average leverage used

    Returns:
        Dict of computed metrics.
    """
    if len(equity_curve) < 2:
        return _empty_metrics(initial_capital)

    values = [e["portfolio_value"] for e in equity_curve]
    dates = [e["date"] for e in equity_curve]

    # Returns
    returns = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        if prev != 0:
            returns.append((values[i] - prev) / prev)
        else:
            returns.append(0.0)

    # Total return
    final_value = values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    total_pnl = final_value - initial_capital

    # Annualized return
    n_periods = len(equity_curve)
    years = n_periods / trading_days_per_year if trading_days_per_year > 0 else 1
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Volatility
    if len(returns) > 1:
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        daily_vol = math.sqrt(variance)
        annualized_vol = daily_vol * math.sqrt(trading_days_per_year)
    else:
        daily_vol = 0
        annualized_vol = 0
        mean_return = 0

    # Sharpe Ratio
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
    if daily_vol > 0:
        sharpe = (mean_return - daily_rf) / daily_vol * math.sqrt(trading_days_per_year)
    else:
        sharpe = 0.0

    # Sortino Ratio (uses downside deviation only)
    downside_returns = [r for r in returns if r < daily_rf]
    if len(downside_returns) > 1:
        downside_variance = sum((r - daily_rf) ** 2 for r in downside_returns) / len(downside_returns)
        downside_dev = math.sqrt(downside_variance)
        sortino = (mean_return - daily_rf) / downside_dev * math.sqrt(trading_days_per_year) if downside_dev > 0 else 0
    else:
        sortino = 0.0

    # Maximum Drawdown
    peak = values[0]
    max_dd = 0.0
    max_dd_start = dates[0]
    max_dd_end = dates[0]
    current_dd_start = dates[0]

    for i, val in enumerate(values):
        if val > peak:
            peak = val
            current_dd_start = dates[i]
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_start = current_dd_start
            max_dd_end = dates[i]

    # Win rate
    winning_trades = [p for p in closed_positions if p.pnl > 0]
    losing_trades = [p for p in closed_positions if p.pnl <= 0]
    total_trades = len(closed_positions)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

    # Average win / loss
    avg_win = (
        sum(p.pnl for p in winning_trades) / len(winning_trades)
        if winning_trades else 0
    )
    avg_loss = (
        sum(p.pnl for p in losing_trades) / len(losing_trades)
        if losing_trades else 0
    )

    # Profit factor
    gross_profit = sum(p.pnl for p in winning_trades)
    gross_loss = abs(sum(p.pnl for p in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Expectancy
    expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss) if total_trades > 0 else 0

    # Calmar ratio
    calmar = annualized_return / max_dd if max_dd > 0 else 0

    # Crypto-specific metrics
    # Omega ratio (probability-weighted gains/losses)
    if len(returns) > 0:
        positive_returns = [r for r in returns if r > 0]
        negative_returns = [r for r in returns if r < 0]
        
        if positive_returns and negative_returns:
            omega = sum(positive_returns) / abs(sum(negative_returns)) if sum(negative_returns) != 0 else 0.0
        else:
            omega = 0.0 if positive_returns else 0.0
    else:
        omega = 0.0

    # Tail ratio (95th percentile gain / 5th percentile loss)
    if len(returns) >= 10:
        sorted_returns = sorted(returns)
        tail_95_idx = int(len(sorted_returns) * 0.95)
        tail_5_idx = int(len(sorted_returns) * 0.05)
        tail_ratio = abs(sorted_returns[tail_95_idx] / sorted_returns[tail_5_idx]) if sorted_returns[tail_5_idx] != 0 else 0
    else:
        tail_ratio = 0.0

    # Benchmark comparison
    alpha = 0.0
    beta = 0.0
    information_ratio = 0.0
    up_capture = 0.0
    down_capture = 0.0
    
    if benchmark_returns and len(benchmark_returns) == len(returns):
        # Beta (market correlation)
        benchmark_mean = sum(benchmark_returns) / len(benchmark_returns)
        returns_mean = sum(returns) / len(returns)
        
        covar = sum((r - returns_mean) * (b - benchmark_mean) for r, b in zip(returns, benchmark_returns))
        bench_var = sum((b - benchmark_mean) ** 2 for b in benchmark_returns)
        
        beta = covar / bench_var if bench_var > 0 else 0
        
        # Alpha (excess return)
        alpha = (returns_mean - (daily_rf + beta * (benchmark_mean - daily_rf))) * trading_days_per_year
        
        # Information ratio
        tracking_error = math.sqrt(sum((r - b - (returns_mean - benchmark_mean)) ** 2 for r, b in zip(returns, benchmark_returns)) / len(returns))
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0
        
        # Up/Down capture
        up_returns = [(r, b) for r, b in zip(returns, benchmark_returns) if b > 0]
        down_returns = [(r, b) for r, b in zip(returns, benchmark_returns) if b < 0]
        
        if up_returns:
            up_capture = sum(r for r, _ in up_returns) / sum(b for _, b in up_returns) if sum(b for _, b in up_returns) > 0 else 0
        if down_returns:
            down_capture = sum(r for r, _ in down_returns) / sum(b for _, b in down_returns) if sum(b for _, b in down_returns) < 0 else 0

    # Fee and funding impact
    fee_impact_pct = (total_fees / initial_capital) * 100 if initial_capital > 0 else 0
    funding_impact_pct = (total_funding / initial_capital) * 100 if initial_capital > 0 else 0

    # Leverage-adjusted metrics
    leverage_adjusted_return = total_return * leverage if leverage > 0 else total_return

    return {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return_pct": total_return * 100,
        "total_pnl": total_pnl,
        "annualized_return_pct": annualized_return * 100,
        "annualized_volatility_pct": annualized_vol * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd * 100,
        "max_drawdown_start": max_dd_start,
        "max_drawdown_end": max_dd_end,
        "calmar_ratio": calmar,
        "omega_ratio": omega,
        "tail_ratio": tail_ratio,
        "total_trades": total_trades,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate_pct": win_rate * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        # Crypto-specific
        "total_fees": total_fees,
        "total_funding": total_funding,
        "fee_impact_pct": fee_impact_pct,
        "funding_impact_pct": funding_impact_pct,
        "liquidations": liquidations,
        "avg_leverage": leverage,
        "leverage_adjusted_return_pct": leverage_adjusted_return * 100,
        # Benchmark
        "alpha": alpha,
        "beta": beta,
        "information_ratio": information_ratio,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "start_date": dates[0],
        "end_date": dates[-1],
        "n_periods": n_periods,
    }


def _empty_metrics(initial_capital: float) -> Dict[str, Any]:
    """Return zeroed metrics when there's insufficient data."""
    return {
        "initial_capital": initial_capital,
        "final_value": initial_capital,
        "total_return_pct": 0.0,
        "total_pnl": 0.0,
        "annualized_return_pct": 0.0,
        "annualized_volatility_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_start": "",
        "max_drawdown_end": "",
        "calmar_ratio": 0.0,
        "omega_ratio": 0.0,
        "tail_ratio": 0.0,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate_pct": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        # Crypto-specific
        "total_fees": 0.0,
        "total_funding": 0.0,
        "fee_impact_pct": 0.0,
        "funding_impact_pct": 0.0,
        "liquidations": 0,
        "avg_leverage": 1.0,
        "leverage_adjusted_return_pct": 0.0,
        # Benchmark
        "alpha": 0.0,
        "beta": 0.0,
        "information_ratio": 0.0,
        "up_capture": 0.0,
        "down_capture": 0.0,
        "start_date": "",
        "end_date": "",
        "n_periods": 0,
    }
