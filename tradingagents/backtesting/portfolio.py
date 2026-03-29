"""Enhanced Portfolio with leverage, fees, and funding for professional crypto backtesting."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    """Tracks a single position entry with enhanced crypto features."""
    side: PositionSide
    entry_price: float
    entry_date: str
    size: float  # number of units
    leverage: float = 1.0
    exit_price: Optional[float] = None
    exit_date: Optional[str] = None
    pnl: float = 0.0
    entry_fees: float = 0.0
    exit_fees: float = 0.0
    funding_costs: float = 0.0
    liquidation_price: Optional[float] = None

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def notional_value(self) -> float:
        """Notional position value (size * price)."""
        price = self.exit_price if self.exit_price else self.entry_price
        return self.size * price

    def close(self, exit_price: float, exit_date: str, exit_fees: float = 0.0):
        """Close the position and compute P&L with fees."""
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.exit_fees = exit_fees
        
        gross_pnl = 0.0
        if self.side == PositionSide.LONG:
            gross_pnl = (exit_price - self.entry_price) * self.size
        elif self.side == PositionSide.SHORT:
            gross_pnl = (self.entry_price - exit_price) * self.size
        
        # Net P&L after fees and funding
        self.pnl = gross_pnl - self.entry_fees - exit_fees - self.funding_costs

    def add_funding(self, funding_cost: float):
        """Add funding cost (positive = paying, negative = receiving)."""
        self.funding_costs += funding_cost

    def calculate_liquidation_price(self, maintenance_margin: float = 0.005) -> Optional[float]:
        """Calculate liquidation price based on leverage and maintenance margin."""
        if self.leverage <= 1:
            return None
        
        # For longs: price drops; For shorts: price rises
        if self.side == PositionSide.LONG:
            liq_price = self.entry_price * (1 - (1 / self.leverage) + maintenance_margin)
            return max(0, liq_price)
        elif self.side == PositionSide.SHORT:
            liq_price = self.entry_price * (1 + (1 / self.leverage) - maintenance_margin)
            return liq_price
        return None


@dataclass
class TradeRecord:
    """A single backtest decision record with crypto details."""
    date: str
    signal: str
    price: float
    action_taken: str
    position_side: str
    portfolio_value: float
    cash: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0
    leverage: float = 1.0
    liquidation_price: Optional[float] = None


class Portfolio:
    """Enhanced portfolio with leverage, fees, and funding for crypto backtesting."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        position_size_pct: float = 0.25,
        leverage: float = 1.0,
        maker_fee: float = 0.0002,  # 0.02%
        taker_fee: float = 0.0005,   # 0.05%
        funding_interval_hours: float = 8.0,
        use_funding: bool = True,
        position_sizing: str = "fixed",  # fixed, kelly, volatility
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_size_pct = position_size_pct
        self.leverage = leverage
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.funding_interval_hours = funding_interval_hours
        self.use_funding = use_funding
        self.position_sizing = position_sizing
        
        self.current_position: Optional[Position] = None
        self.closed_positions: List[Position] = []
        self.trade_history: List[TradeRecord] = []
        self.equity_curve: List[dict] = []
        
        # Tracking
        self.total_fees_paid: float = 0.0
        self.total_funding_paid: float = 0.0
        self.liquidations: int = 0
        self.last_funding_date: Optional[str] = None

    @property
    def position_side(self) -> PositionSide:
        if self.current_position and self.current_position.is_open:
            return self.current_position.side
        return PositionSide.FLAT

    def _position_size_units(self, price: float, volatility: Optional[float] = None) -> float:
        """Calculate position size with crypto enhancements."""
        portfolio_val = self.portfolio_value(price)
        
        if self.position_sizing == "fixed":
            allocation = portfolio_val * self.position_size_pct
        elif self.position_sizing == "kelly":
            win_rate = self._estimate_win_rate()
            avg_win = self._estimate_avg_win()
            avg_loss = self._estimate_avg_loss()
            if avg_loss > 0:
                b = avg_win / avg_loss
                kelly = (win_rate * b - (1 - win_rate)) / b
                kelly = max(0.01, min(kelly * 0.5, 0.25))
                allocation = portfolio_val * kelly
            else:
                allocation = portfolio_val * self.position_size_pct
        elif self.position_sizing == "volatility":
            if volatility and volatility > 0:
                target_vol = 0.02
                vol_adj = target_vol / volatility
                allocation = portfolio_val * self.position_size_pct * min(vol_adj, 2.0)
            else:
                allocation = portfolio_val * self.position_size_pct
        else:
            allocation = portfolio_val * self.position_size_pct
        
        # Apply leverage
        effective_allocation = allocation * self.leverage
        return effective_allocation / price if price > 0 else 0

    def _estimate_win_rate(self) -> float:
        if not self.closed_positions:
            return 0.5
        winners = sum(1 for p in self.closed_positions if p.pnl > 0)
        return winners / len(self.closed_positions)

    def _estimate_avg_win(self) -> float:
        wins = [p.pnl for p in self.closed_positions if p.pnl > 0]
        return sum(wins) / len(wins) if wins else 1000.0

    def _estimate_avg_loss(self) -> float:
        losses = [abs(p.pnl) for p in self.closed_positions if p.pnl < 0]
        return sum(losses) / len(losses) if losses else 500.0

    def portfolio_value(self, current_price: float) -> float:
        """Total portfolio value = cash + locked margin + unrealized P&L."""
        locked_margin = 0.0
        if self.current_position and self.current_position.is_open:
            pos = self.current_position
            locked_margin = (pos.size * pos.entry_price) / pos.leverage if pos.leverage > 0 else pos.size * pos.entry_price
            
        return self.cash + locked_margin + self.unrealized_pnl(current_price)

    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized P&L of the current open position."""
        if not self.current_position or not self.current_position.is_open:
            return 0.0
        pos = self.current_position
        if pos.side == PositionSide.LONG:
            return (current_price - pos.entry_price) * pos.size
        elif pos.side == PositionSide.SHORT:
            return (pos.entry_price - current_price) * pos.size
        return 0.0

    def total_realized_pnl(self) -> float:
        """Sum of all closed position P&Ls."""
        return sum(p.pnl for p in self.closed_positions)

    def _calculate_funding(self, date: str, price: float, actual_rate: Optional[float] = None) -> float:
        """Calculate funding cost for shorts (paid) or longs (received)."""
        if not self.use_funding or not self.current_position:
            return 0.0
        
        if self.last_funding_date is None:
            self.last_funding_date = date
            return 0.0
        
        last_fmt = "%Y-%m-%d %H:%M:%S" if " " in self.last_funding_date else "%Y-%m-%d"
        curr_fmt = "%Y-%m-%d %H:%M:%S" if " " in date else "%Y-%m-%d"
        last_dt = datetime.strptime(self.last_funding_date, last_fmt)
        current_dt = datetime.strptime(date, curr_fmt)
        hours_passed = (current_dt - last_dt).total_seconds() / 3600
        
        if hours_passed >= self.funding_interval_hours:
            self.last_funding_date = date
            # If API found dynamic rate, use it, else default fallback
            funding_rate = actual_rate if actual_rate is not None else 0.0001
            notional = self.current_position.size * price
            funding_cost = notional * funding_rate
            
            if self.current_position.side == PositionSide.SHORT:
                return funding_cost
            elif self.current_position.side == PositionSide.LONG:
                # Longs historically receive funding if funding is positive
                # Hyperliquid pays out the actual fundingRate
                return -funding_cost * 0.5
        
        return 0.0

    def _check_liquidation(self, price: float) -> bool:
        """Check if position should be liquidated."""
        if not self.current_position or self.leverage <= 1:
            return False
        
        pos = self.current_position
        if pos.liquidation_price is None:
            return False
        
        if pos.side == PositionSide.LONG and price <= pos.liquidation_price:
            return True
        elif pos.side == PositionSide.SHORT and price >= pos.liquidation_price:
            return True
        return False

    def _liquidate_position(self, price: float, date: str) -> str:
        """Liquidate position - total loss of margin."""
        if not self.current_position:
            return "No position to liquidate"
        
        pos = self.current_position
        margin = (pos.size * pos.entry_price) / self.leverage
        
        pos.pnl = -margin - pos.entry_fees - pos.funding_costs
        pos.exit_price = price
        pos.exit_date = date
        pos.exit_fees = 0
        
        self.cash -= margin
        self.closed_positions.append(pos)
        self.current_position = None
        self.liquidations += 1
        
        logger.error(f"[{date}] LIQUIDATION at ${price:,.2f}! Margin ${margin:,.2f} lost.")
        return f"LIQUIDATED: ${margin:,.2f} margin lost"

    def process_signal(self, signal: str, price: float, date: str, use_limit_order: bool = False, funding_rate: Optional[float] = None) -> str:
        """
        Process a trading signal with crypto enhancements.

        Args:
            signal: One of BUY, SELL, HOLD, SHORT, COVER, OVERWEIGHT, UNDERWEIGHT
            price: Current market price
            date: Current date string
            use_limit_order: If True, use maker fees (lower)
            funding_rate: Fetched actual funding rate (historically dynamic)

        Returns:
            Description of the action taken.
        """
        signal = signal.upper().strip()
        action = "HOLD — no action"
        current_side = self.position_side
        
        # Calculate funding costs first
        funding_cost = self._calculate_funding(date, price, funding_rate)
        if self.current_position and funding_cost != 0:
            self.current_position.add_funding(funding_cost)
            self.total_funding_paid += funding_cost
        
        # Check for liquidation
        if self._check_liquidation(price):
            action = self._liquidate_position(price, date)
            
            record = TradeRecord(
                date=date,
                signal=signal,
                price=price,
                action_taken=action,
                position_side=self.position_side.value,
                portfolio_value=self.portfolio_value(price),
                cash=self.cash,
                realized_pnl=self.total_realized_pnl(),
                funding_paid=funding_cost,
                leverage=self.leverage,
            )
            self.trade_history.append(record)
            self.equity_curve.append({
                "date": date,
                "portfolio_value": self.portfolio_value(price),
                "cash": self.cash,
                "position_side": self.position_side.value,
                "fees": self.total_fees_paid,
                "funding": self.total_funding_paid,
            })
            return action

        # Determine fee rate
        fee_rate = self.maker_fee if use_limit_order else self.taker_fee

        if signal in ("BUY", "OVERWEIGHT"):
            if current_side == PositionSide.FLAT:
                # Enter long with leverage and fees
                size = self._position_size_units(price)
                entry_fees = size * price * fee_rate
                
                self.current_position = Position(
                    side=PositionSide.LONG,
                    entry_price=price,
                    entry_date=date,
                    size=size,
                    leverage=self.leverage,
                    entry_fees=entry_fees,
                )
                self.current_position.liquidation_price = self.current_position.calculate_liquidation_price()
                
                margin = (size * price) / self.leverage
                self.cash -= margin + entry_fees
                self.total_fees_paid += entry_fees
                
                action = f"ENTERED LONG: {size:.6f} units @ ${price:,.2f} (leverage {self.leverage}x, fees ${entry_fees:,.2f})"
                
            elif current_side == PositionSide.SHORT:
                # Cover short first
                cover_fees = self._close_position(price, date, fee_rate)
                
                # Then go long
                size = self._position_size_units(price)
                entry_fees = size * price * fee_rate
                
                self.current_position = Position(
                    side=PositionSide.LONG,
                    entry_price=price,
                    entry_date=date,
                    size=size,
                    leverage=self.leverage,
                    entry_fees=entry_fees,
                )
                self.current_position.liquidation_price = self.current_position.calculate_liquidation_price()
                
                margin = (size * price) / self.leverage
                self.cash -= margin + entry_fees
                self.total_fees_paid += entry_fees
                
                action = f"COVERED SHORT + ENTERED LONG: {size:.6f} units @ ${price:,.2f} (fees ${entry_fees + cover_fees:,.2f})"
            else:
                action = "HOLD — already long"

        elif signal == "SELL":
            if current_side == PositionSide.LONG:
                exit_fees = self._close_position(price, date, fee_rate)
                action = f"CLOSED LONG @ ${price:,.2f} (fees ${exit_fees:,.2f})"
            elif current_side == PositionSide.FLAT:
                action = "HOLD — no position to sell"
            elif current_side == PositionSide.SHORT:
                # SELL on a short = COVER (LLM uses SELL to mean exit position)
                exit_fees = self._close_position(price, date, fee_rate)
                action = f"COVERED SHORT (via SELL) @ ${price:,.2f} (fees ${exit_fees:,.2f})"

        elif signal == "SHORT":
            if current_side == PositionSide.FLAT:
                size = self._position_size_units(price)
                entry_fees = size * price * fee_rate
                
                self.current_position = Position(
                    side=PositionSide.SHORT,
                    entry_price=price,
                    entry_date=date,
                    size=size,
                    leverage=self.leverage,
                    entry_fees=entry_fees,
                )
                self.current_position.liquidation_price = self.current_position.calculate_liquidation_price()
                
                # Short: margin is locked
                margin = (size * price) / self.leverage
                self.cash -= margin + entry_fees
                self.total_fees_paid += entry_fees
                
                liq_str = f"${self.current_position.liquidation_price:,.2f}" if self.current_position.liquidation_price is not None else "N/A"
                action = f"ENTERED SHORT: {size:.6f} units @ ${price:,.2f} (leverage {self.leverage}x, liq @ {liq_str})"
                
            elif current_side == PositionSide.LONG:
                # Close long first
                close_fees = self._close_position(price, date, fee_rate)
                
                # Then short
                size = self._position_size_units(price)
                entry_fees = size * price * fee_rate
                
                self.current_position = Position(
                    side=PositionSide.SHORT,
                    entry_price=price,
                    entry_date=date,
                    size=size,
                    leverage=self.leverage,
                    entry_fees=entry_fees,
                )
                self.current_position.liquidation_price = self.current_position.calculate_liquidation_price()
                
                margin = (size * price) / self.leverage
                self.cash -= margin + entry_fees
                self.total_fees_paid += entry_fees
                
                liq_str = f"${self.current_position.liquidation_price:,.2f}" if self.current_position.liquidation_price is not None else "N/A"
                action = f"CLOSED LONG + ENTERED SHORT: {size:.6f} units @ ${price:,.2f} (liq @ {liq_str}, fees ${entry_fees + close_fees:,.2f})"
            else:
                action = "HOLD — already short"

        elif signal == "COVER":
            if current_side == PositionSide.SHORT:
                exit_fees = self._close_position(price, date, fee_rate)
                action = f"COVERED SHORT @ ${price:,.2f} (fees ${exit_fees:,.2f})"
            else:
                action = "HOLD — no short position to cover"

        elif signal in ("UNDERWEIGHT",):
            if current_side == PositionSide.LONG:
                # Partial close (50% of position)
                pos = self.current_position
                close_size = pos.size * 0.5
                partial_pnl = (price - pos.entry_price) * close_size
                self.cash += close_size * price
                pos.size -= close_size
                action = f"REDUCED LONG by 50% @ ${price:,.2f} (partial P&L: ${partial_pnl:,.2f})"
                if pos.size <= 0:
                    self._close_position(price, date)
            else:
                action = "HOLD — no long position to reduce"

        # Record the trade
        record = TradeRecord(
            date=date,
            signal=signal,
            price=price,
            action_taken=action,
            position_side=self.position_side.value,
            portfolio_value=self.portfolio_value(price),
            cash=self.cash,
            unrealized_pnl=self.unrealized_pnl(price),
            realized_pnl=self.total_realized_pnl(),
            fees_paid=self.total_fees_paid,
            funding_paid=self.total_funding_paid,
            leverage=self.leverage,
            liquidation_price=self.current_position.liquidation_price if self.current_position else None,
        )
        self.trade_history.append(record)

        # Update equity curve
        self.equity_curve.append({
            "date": date,
            "portfolio_value": self.portfolio_value(price),
            "cash": self.cash,
            "position_side": self.position_side.value,
            "fees": self.total_fees_paid,
            "funding": self.total_funding_paid,
            "leverage": self.leverage,
        })

        logger.info(f"[{date}] Signal={signal} | {action} | Portfolio=${self.portfolio_value(price):,.2f}")
        return action

    def _close_position(self, price: float, date: str, fee_rate: float = 0.0005) -> float:
        """Close the current open position and return exit fees paid."""
        if not self.current_position or not self.current_position.is_open:
            return 0.0

        pos = self.current_position
        exit_fees = pos.size * price * fee_rate
        self.total_fees_paid += exit_fees
        
        # Calculate margin to return
        margin = (pos.size * pos.entry_price) / pos.leverage if pos.leverage > 0 else pos.size * pos.entry_price
        
        pos.close(price, date, exit_fees)
        
        # Update cash: return margin + P&L
        if pos.side == PositionSide.LONG:
            self.cash += margin + (pos.exit_price - pos.entry_price) * pos.size - exit_fees
        elif pos.side == PositionSide.SHORT:
            self.cash += margin + (pos.entry_price - pos.exit_price) * pos.size - exit_fees

        self.closed_positions.append(pos)
        self.current_position = None
        
        return exit_fees

    def force_close(self, price: float, date: str, fee_rate: float = 0.0005):
        """Force-close any open position at end of backtest."""
        if self.current_position and self.current_position.is_open:
            exit_fees = self._close_position(price, date, fee_rate)
            logger.info(f"[{date}] FORCE CLOSED at end of backtest @ ${price:,.2f} (fees ${exit_fees:,.2f})")

    def get_stats(self) -> Dict[str, Any]:
        """Get portfolio statistics."""
        return {
            "total_fees_paid": self.total_fees_paid,
            "total_funding_paid": self.total_funding_paid,
            "liquidations": self.liquidations,
            "total_trades": len(self.closed_positions),
            "winning_trades": sum(1 for p in self.closed_positions if p.pnl > 0),
            "losing_trades": sum(1 for p in self.closed_positions if p.pnl < 0),
            "avg_leverage": self.leverage,
            "position_sizing": self.position_sizing,
        }
