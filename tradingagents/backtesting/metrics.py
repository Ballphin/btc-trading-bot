"""Performance metrics for backtesting: Sharpe, Sortino, drawdown, win rate, etc."""

import logging
import math
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def compute_metrics(
    equity_curve: List[dict],
    closed_positions: list,
    initial_capital: float,
    risk_free_rate: float = 0.04,
    trading_days_per_year: int = 252,
    benchmark_returns: List[float] = None,
    benchmark_return_pct: float = None,
    total_fees: float = 0.0,
    total_funding: float = 0.0,
    liquidations: int = 0,
    leverage: float = 1.0,
    stops_hit: int = 0,
    takes_hit: int = 0,
    is_crypto: bool = False,
    frequency: str = "daily",  # "daily", "4h", "1h"
) -> Dict[str, Any]:
    """
    Compute comprehensive backtest performance metrics with crypto enhancements.

    Args:
        equity_curve: List of {date, portfolio_value, ...} dicts
        closed_positions: List of closed Position objects
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate (default 4%)
        trading_days_per_year: Annualization factor (overridden to 365 when is_crypto=True)
        benchmark_returns: Optional list of daily benchmark returns for CAPM metrics
        benchmark_return_pct: Optional total buy-and-hold benchmark return % for alpha display
        total_fees: Total trading fees paid
        total_funding: Total funding costs (positive = paid, negative = received)
        liquidations: Number of liquidations
        leverage: Average leverage used
        is_crypto: If True, uses 365 trading days/year for annualisation

    Returns:
        Dict of computed metrics.
    """
    # Annualization periods: frequency overrides is_crypto
    if frequency == "4h":
        trading_days_per_year = 2190  # 365 days * 6 four-hour periods
    elif frequency == "1h":
        trading_days_per_year = 8760  # 365 * 24
    elif is_crypto:
        trading_days_per_year = 365
    if len(equity_curve) < 2:
        return _empty_metrics(initial_capital)

    values = [e["portfolio_value"] for e in equity_curve]
    dates = [e["date"] for e in equity_curve]
    n_periods = len(equity_curve)

    # Sample size tier for statistical reliability
    # SE(SR) ≈ √((1 + 0.5×SR²) / n); n<10 → SE>0.39 (unreliable), n<30 → SE<0.24 (decent)
    if n_periods < 10:
        sample_size_tier = "unreliable"
    elif n_periods < 30:
        sample_size_tier = "limited"
    else:
        sample_size_tier = "reliable"

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

    # Annualized return — use actual calendar span, not n_periods
    try:
        _fmt1 = "%Y-%m-%d %H:%M:%S" if " " in dates[0] else "%Y-%m-%d"
        _fmt2 = "%Y-%m-%d %H:%M:%S" if " " in dates[-1] else "%Y-%m-%d"
        _dt_first = datetime.strptime(dates[0], _fmt1)
        _dt_last = datetime.strptime(dates[-1], _fmt2)
        calendar_days = (_dt_last - _dt_first).days
        years = calendar_days / 365.25 if calendar_days > 0 else (1 / trading_days_per_year)
    except (ValueError, TypeError):
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

    # Profit factor — None means ∞ (no losing trades); serialised as null in JSON
    gross_profit = sum(p.pnl for p in winning_trades)
    gross_loss = abs(sum(p.pnl for p in losing_trades))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-9 else (None if gross_profit > 0 else 0.0)

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
    alpha_pct = (total_return * 100 - benchmark_return_pct) if benchmark_return_pct is not None else None
    
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

    # Risk management metrics
    positions_with_hold_days = [p for p in closed_positions if hasattr(p, 'entry_date') and hasattr(p, 'exit_date') and p.exit_date]
    if positions_with_hold_days:
        total_hold_days = 0
        for p in positions_with_hold_days:
            try:
                entry_fmt = "%Y-%m-%d %H:%M:%S" if " " in p.entry_date else "%Y-%m-%d"
                exit_fmt = "%Y-%m-%d %H:%M:%S" if " " in p.exit_date else "%Y-%m-%d"
                entry_dt = datetime.strptime(p.entry_date, entry_fmt)
                exit_dt = datetime.strptime(p.exit_date, exit_fmt)
                hold_days = (exit_dt - entry_dt).days
                total_hold_days += hold_days
            except (ValueError, AttributeError) as e:
                logger.debug(f"Malformed position dates: {e}")
        avg_hold_days = total_hold_days / len(positions_with_hold_days) if positions_with_hold_days else 0
    else:
        avg_hold_days = 0

    # Average Risk:Reward ratio (for positions with stop/take defined)
    positions_with_rr = [p for p in closed_positions if hasattr(p, 'stop_loss_price') and hasattr(p, 'take_profit_price') and p.stop_loss_price and p.take_profit_price]
    if positions_with_rr:
        rr_ratios = []
        for p in positions_with_rr:
            if hasattr(p, 'side'):
                if p.side.value == 'LONG':
                    risk = abs(p.entry_price - p.stop_loss_price)
                    reward = abs(p.take_profit_price - p.entry_price)
                elif p.side.value == 'SHORT':
                    risk = abs(p.stop_loss_price - p.entry_price)
                    reward = abs(p.entry_price - p.take_profit_price)
                else:
                    continue
                if risk > 0:
                    rr_ratios.append(reward / risk)
        avg_rr_ratio = sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0
    else:
        avg_rr_ratio = 0

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
        "profit_factor": profit_factor,  # None = ∞ (no losing trades)
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
        "benchmark_return_pct": benchmark_return_pct,
        "alpha_pct": alpha_pct,
        "alpha": alpha,
        "beta": beta,
        "information_ratio": information_ratio,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "start_date": dates[0],
        "end_date": dates[-1],
        "n_periods": n_periods,
        "sample_size_tier": sample_size_tier,
        "sharpe_se": math.sqrt((1 + 0.5 * sharpe ** 2) / max(n_periods, 1)),
        # Risk management
        "stops_hit": stops_hit,
        "takes_hit": takes_hit,
        "avg_hold_days": avg_hold_days,
        "avg_rr_ratio": avg_rr_ratio,
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
        "profit_factor": None,
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
        "benchmark_return_pct": None,
        "alpha_pct": None,
        "alpha": 0.0,
        "beta": 0.0,
        "information_ratio": 0.0,
        "up_capture": 0.0,
        "down_capture": 0.0,
        "start_date": "",
        "end_date": "",
        "n_periods": 0,
        "sample_size_tier": "unreliable",
        "sharpe_se": 0.0,
        # Risk management
        "stops_hit": 0,
        "takes_hit": 0,
        "avg_hold_days": 0,
        "avg_rr_ratio": 0,
    }
