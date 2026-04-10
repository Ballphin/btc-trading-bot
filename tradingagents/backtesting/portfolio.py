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
    # Risk management fields
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_hold_days: Optional[int] = None
    exit_reason: str = ""  # stopped_out, take_profit, time_exit, signal, liquidation

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
    # Risk management fields
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    hold_days: Optional[int] = None
    exit_reason: str = ""
    atr_at_entry: Optional[float] = None


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
        position_sizing: str = "fixed",  # fixed, kelly, volatility, atr_risk
        risk_per_trade: float = 0.01,  # 1% risk per trade for ATR sizing
        slippage_bps: float = 5.0,  # Slippage in basis points (5 for BTC, 15 for alts)
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
        self.risk_per_trade = risk_per_trade
        self.slippage_bps = slippage_bps
        
        self.current_position: Optional[Position] = None
        self.closed_positions: List[Position] = []
        self.trade_history: List[TradeRecord] = []
        self.equity_curve: List[dict] = []
        
        # Tracking
        self.total_fees_paid: float = 0.0
        self.total_funding_paid: float = 0.0
        self.liquidations: int = 0
        self.last_funding_date: Optional[str] = None
        self.stops_hit: int = 0
        self.takes_hit: int = 0

    @property
    def position_side(self) -> PositionSide:
        if self.current_position and self.current_position.is_open:
            return self.current_position.side
        return PositionSide.FLAT

    def _position_size_units(self, price: float, volatility: Optional[float] = None, stop_loss_price: Optional[float] = None) -> float:
        """Calculate position size with crypto enhancements."""
        portfolio_val = self.portfolio_value(price)
        
        if self.position_sizing == "atr_risk" and stop_loss_price is not None:
            # ATR-based risk sizing: risk exactly risk_per_trade % of portfolio
            risk_amount = portfolio_val * self.risk_per_trade
            stop_distance = abs(price - stop_loss_price)
            if stop_distance > 0:
                size = risk_amount / stop_distance
                # Apply leverage to size
                return size * self.leverage
            else:
                # Fallback to fixed if no valid stop
                allocation = portfolio_val * self.position_size_pct
        elif self.position_sizing == "fixed":
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
        """Calculate funding cost for shorts (paid) or longs (received). Now charges EVERY period."""
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
        
        # Charge funding for every elapsed interval (e.g., 3 intervals for 24h at 8h cadence)
        if hours_passed >= self.funding_interval_hours:
            n_intervals = max(1, int(hours_passed // self.funding_interval_hours))
            self.last_funding_date = date
            # If API found dynamic rate, use it, else zero (don't fabricate costs)
            funding_rate = actual_rate if actual_rate is not None else 0.0
            if actual_rate is None:
                logger.debug(f"[{date}] No funding rate data — defaulting to 0.0")
            notional = self.current_position.size * price
            funding_cost = notional * funding_rate * n_intervals
            
            if self.current_position.side == PositionSide.SHORT:
                return funding_cost
            elif self.current_position.side == PositionSide.LONG:
                # Longs receive funding if funding is positive
                # Hyperliquid pays 100% of the fundingRate to longs
                return -funding_cost * 1.0
        
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

    def process_signal(
        self,
        signal: str,
        price: float,
        date: str,
        use_limit_order: bool = False,
        funding_rate: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        max_hold_days: Optional[int] = None,
        atr: Optional[float] = None,
    ) -> str:
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

        # Apply slippage to execution price (adverse fill simulation)
        slippage_mult = self.slippage_bps / 10_000
        if signal in ("BUY", "OVERWEIGHT", "COVER"):
            price = price * (1 + slippage_mult)  # buy higher (adverse for buys and buy-to-close)
        elif signal in ("SELL", "UNDERWEIGHT"):
            price = price * (1 - slippage_mult)  # sell lower (adverse for sells)
        elif signal == "SHORT":
            price = price * (1 - slippage_mult)  # short entry lower (adverse for short opens)
        
        # ATR-adaptive stop validation — aggressive settings per user preference
        # Floor: 1.0x ATR minimum (anything tighter is inside normal noise)
        # Override to: 1.5x ATR (backtest) or 2.0x ATR (live thin-book hours)
        if atr and atr > 0 and stop_loss_price and stop_loss_price > 0:
            stop_distance = abs(price - stop_loss_price)
            atr_multiple = stop_distance / atr

            # Determine floor based on mode (backtest vs live) and liquidity
            from tradingagents.backtesting.context import BACKTEST_MODE
            if BACKTEST_MODE.get():
                atr_floor = 1.5  # backtest: static aggressive
            else:
                _now_utc = datetime.utcnow()
                _is_thin_book = _now_utc.hour in range(0, 8) or _now_utc.weekday() >= 5
                atr_floor = 2.0 if _is_thin_book else 1.5

            if atr_multiple < 1.0:
                logger.warning(
                    f"Stop {atr_multiple:.1f}x ATR inside noise, widening to {atr_floor}x ATR"
                )
                if signal in ("BUY", "OVERWEIGHT"):
                    stop_loss_price = price - atr_floor * atr
                else:
                    stop_loss_price = price + atr_floor * atr

        # Calculate funding costs first (continuous funding)
        funding_cost = self._calculate_funding(date, price, funding_rate)
        if self.current_position and funding_cost != 0:
            self.current_position.add_funding(funding_cost)
            self.total_funding_paid += funding_cost
        
        # Check for stop loss hit BEFORE processing new signal
        if self.current_position and self.current_position.stop_loss_price is not None:
            pos = self.current_position
            if (pos.side == PositionSide.LONG and price <= pos.stop_loss_price) or \
               (pos.side == PositionSide.SHORT and price >= pos.stop_loss_price):
                exit_fees = self._close_position(price, date, self.taker_fee)
                pos.exit_reason = "stopped_out"
                self.stops_hit += 1
                action = f"STOPPED OUT @ ${price:,.2f} (stop was ${pos.stop_loss_price:,.2f}, fees ${exit_fees:,.2f})"
                
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
                    exit_reason="stopped_out",
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
        
        # Check for take profit hit BEFORE processing new signal
        if self.current_position and self.current_position.take_profit_price is not None:
            pos = self.current_position
            if (pos.side == PositionSide.LONG and price >= pos.take_profit_price) or \
               (pos.side == PositionSide.SHORT and price <= pos.take_profit_price):
                exit_fees = self._close_position(price, date, self.maker_fee)
                pos.exit_reason = "take_profit"
                self.takes_hit += 1
                action = f"TAKE PROFIT @ ${price:,.2f} (target was ${pos.take_profit_price:,.2f}, fees ${exit_fees:,.2f})"
                
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
                    exit_reason="take_profit",
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
        
        # Check for time-based exit BEFORE processing new signal
        if self.current_position and self.current_position.max_hold_days is not None:
            pos = self.current_position
            entry_fmt = "%Y-%m-%d %H:%M:%S" if " " in pos.entry_date else "%Y-%m-%d"
            curr_fmt = "%Y-%m-%d %H:%M:%S" if " " in date else "%Y-%m-%d"
            entry_dt = datetime.strptime(pos.entry_date, entry_fmt)
            current_dt = datetime.strptime(date, curr_fmt)
            days_held = (current_dt - entry_dt).days
            
            if days_held >= pos.max_hold_days:
                exit_fees = self._close_position(price, date, self.taker_fee)
                pos.exit_reason = "time_exit"
                action = f"TIME EXIT @ ${price:,.2f} (held {days_held} days, max was {pos.max_hold_days}, fees ${exit_fees:,.2f})"
                
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
                    exit_reason="time_exit",
                    hold_days=days_held,
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
        
        # Check for liquidation
        if self._check_liquidation(price):
            action = self._liquidate_position(price, date)
            if self.current_position:
                self.current_position.exit_reason = "liquidation"
            
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
                size = self._position_size_units(price, stop_loss_price=stop_loss_price)
                entry_fees = size * price * fee_rate
                
                self.current_position = Position(
                    side=PositionSide.LONG,
                    entry_price=price,
                    entry_date=date,
                    size=size,
                    leverage=self.leverage,
                    entry_fees=entry_fees,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    max_hold_days=max_hold_days,
                )
                self.current_position.liquidation_price = self.current_position.calculate_liquidation_price()
                
                margin = (size * price) / self.leverage
                self.cash -= margin + entry_fees
                self.total_fees_paid += entry_fees
                
                action = f"ENTERED LONG: {size:.6f} units @ ${price:,.2f} (leverage {self.leverage}x, fees ${entry_fees:,.2f})"
                if stop_loss_price:
                    action += f" [Stop: ${stop_loss_price:,.2f}]"
                if take_profit_price:
                    action += f" [Target: ${take_profit_price:,.2f}]"
                
            elif current_side == PositionSide.SHORT:
                # Cover short first
                cover_fees = self._close_position(price, date, fee_rate)
                
                # Then go long
                size = self._position_size_units(price, stop_loss_price=stop_loss_price)
                entry_fees = size * price * fee_rate
                
                self.current_position = Position(
                    side=PositionSide.LONG,
                    entry_price=price,
                    entry_date=date,
                    size=size,
                    leverage=self.leverage,
                    entry_fees=entry_fees,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    max_hold_days=max_hold_days,
                )
                self.current_position.liquidation_price = self.current_position.calculate_liquidation_price()
                
                margin = (size * price) / self.leverage
                self.cash -= margin + entry_fees
                self.total_fees_paid += entry_fees
                
                action = f"COVERED SHORT + ENTERED LONG: {size:.6f} units @ ${price:,.2f} (fees ${entry_fees + cover_fees:,.2f})"
                if stop_loss_price:
                    action += f" [Stop: ${stop_loss_price:,.2f}]"
                if take_profit_price:
                    action += f" [Target: ${take_profit_price:,.2f}]"
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
                size = self._position_size_units(price, stop_loss_price=stop_loss_price)
                entry_fees = size * price * fee_rate
                
                self.current_position = Position(
                    side=PositionSide.SHORT,
                    entry_price=price,
                    entry_date=date,
                    size=size,
                    leverage=self.leverage,
                    entry_fees=entry_fees,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    max_hold_days=max_hold_days,
                )
                self.current_position.liquidation_price = self.current_position.calculate_liquidation_price()
                
                # Short: margin is locked
                margin = (size * price) / self.leverage
                self.cash -= margin + entry_fees
                self.total_fees_paid += entry_fees
                
                liq_str = f"${self.current_position.liquidation_price:,.2f}" if self.current_position.liquidation_price is not None else "N/A"
                action = f"ENTERED SHORT: {size:.6f} units @ ${price:,.2f} (leverage {self.leverage}x, liq @ {liq_str})"
                if stop_loss_price:
                    action += f" [Stop: ${stop_loss_price:,.2f}]"
                if take_profit_price:
                    action += f" [Target: ${take_profit_price:,.2f}]"
                
            elif current_side == PositionSide.LONG:
                # Close long first
                close_fees = self._close_position(price, date, fee_rate)
                
                # Then short
                size = self._position_size_units(price, stop_loss_price=stop_loss_price)
                entry_fees = size * price * fee_rate
                
                self.current_position = Position(
                    side=PositionSide.SHORT,
                    entry_price=price,
                    entry_date=date,
                    size=size,
                    leverage=self.leverage,
                    entry_fees=entry_fees,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    max_hold_days=max_hold_days,
                )
                self.current_position.liquidation_price = self.current_position.calculate_liquidation_price()
                
                margin = (size * price) / self.leverage
                self.cash -= margin + entry_fees
                self.total_fees_paid += entry_fees
                
                liq_str = f"${self.current_position.liquidation_price:,.2f}" if self.current_position.liquidation_price is not None else "N/A"
                action = f"CLOSED LONG + ENTERED SHORT: {size:.6f} units @ ${price:,.2f} (liq @ {liq_str}, fees ${entry_fees + close_fees:,.2f})"
                if stop_loss_price:
                    action += f" [Stop: ${stop_loss_price:,.2f}]"
                if take_profit_price:
                    action += f" [Target: ${take_profit_price:,.2f}]"
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

        # Record the trade with risk management fields
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
            stop_loss=self.current_position.stop_loss_price if self.current_position else None,
            take_profit=self.current_position.take_profit_price if self.current_position else None,
            hold_days=None,  # Will be calculated in metrics
            exit_reason="",  # Set on exit
            atr_at_entry=atr,
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

    def scale_position(self, price: float, date: str, scale_factor: float = 0.5, fee_rate: float = 0.0005) -> str:
        """Scale into existing position (pyramiding).
        
        Args:
            price: Current price
            date: Current date
            scale_factor: Fraction of original position to add (default 0.5 = 50%)
            fee_rate: Fee rate for the additional entry
            
        Returns:
            Action description
        """
        if not self.current_position or not self.current_position.is_open:
            return "No position to scale"
        
        pos = self.current_position
        additional_size = pos.size * scale_factor
        entry_fees = additional_size * price * fee_rate
        
        # Add to position
        margin = (additional_size * price) / self.leverage
        self.cash -= margin + entry_fees
        self.total_fees_paid += entry_fees
        
        # Update position (weighted average entry price)
        total_size = pos.size + additional_size
        pos.entry_price = (pos.entry_price * pos.size + price * additional_size) / total_size
        pos.size = total_size
        pos.entry_fees += entry_fees
        
        logger.info(f"[{date}] SCALED {pos.side.value} by {scale_factor*100:.0f}% @ ${price:,.2f}")
        return f"SCALED {pos.side.value}: added {additional_size:.6f} units @ ${price:,.2f} (fees ${entry_fees:,.2f})"
    
    def rebalance_position(self, price: float, date: str, target_pct: float = 0.5, fee_rate: float = 0.0005) -> str:
        """Rebalance position to target percentage of portfolio.
        
        Args:
            price: Current price
            date: Current date
            target_pct: Target position size as % of portfolio (default 0.5 = 50%)
            fee_rate: Fee rate for rebalancing
            
        Returns:
            Action description
        """
        if not self.current_position or not self.current_position.is_open:
            return "No position to rebalance"
        
        pos = self.current_position
        portfolio_val = self.portfolio_value(price)
        target_value = portfolio_val * target_pct
        current_value = pos.size * price
        
        if abs(current_value - target_value) / portfolio_val < 0.05:  # Within 5% tolerance
            return "Position already balanced"
        
        if current_value > target_value:
            # Reduce position
            reduce_size = (current_value - target_value) / price
            exit_fees = reduce_size * price * fee_rate
            
            # Partial close
            margin_return = (reduce_size * pos.entry_price) / pos.leverage
            pnl = (price - pos.entry_price) * reduce_size if pos.side == PositionSide.LONG else (pos.entry_price - price) * reduce_size
            
            self.cash += margin_return + pnl - exit_fees
            self.total_fees_paid += exit_fees
            pos.size -= reduce_size
            
            logger.info(f"[{date}] REBALANCED (reduced) {pos.side.value} by {reduce_size:.6f} units")
            return f"REBALANCED: reduced {pos.side.value} by {reduce_size:.6f} units @ ${price:,.2f}"
        else:
            # Increase position
            add_size = (target_value - current_value) / price
            entry_fees = add_size * price * fee_rate
            
            margin = (add_size * price) / self.leverage
            self.cash -= margin + entry_fees
            self.total_fees_paid += entry_fees
            
            # Update weighted average entry
            total_size = pos.size + add_size
            pos.entry_price = (pos.entry_price * pos.size + price * add_size) / total_size
            pos.size = total_size
            pos.entry_fees += entry_fees
            
            logger.info(f"[{date}] REBALANCED (increased) {pos.side.value} by {add_size:.6f} units")
            return f"REBALANCED: increased {pos.side.value} by {add_size:.6f} units @ ${price:,.2f}"

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
            "stops_hit": self.stops_hit,
            "takes_hit": self.takes_hit,
        }
