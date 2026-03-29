"""Technical indicators for backtesting."""

import logging
from typing import List, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def calculate_atr(ticker: str, date: str, period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range (ATR) for a given ticker and date.
    
    Args:
        ticker: Asset ticker symbol
        date: Date string in YYYY-MM-DD format
        period: ATR period (default 14)
    
    Returns:
        ATR value or None if calculation fails
    """
    try:
        # Parse date
        date_fmt = "%Y-%m-%d %H:%M:%S" if " " in date else "%Y-%m-%d"
        target_date = datetime.strptime(date, date_fmt)
        
        # Fetch historical data (need period + 1 days for ATR calculation)
        start_date = target_date - timedelta(days=period * 3)  # Buffer for weekends/holidays
        end_date = target_date + timedelta(days=1)
        
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        
        if data.empty or len(data) < period:
            logger.warning(f"Insufficient data for ATR calculation: {ticker} on {date}")
            return None
        
        # Calculate True Range
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        # Use pandas maximum function for element-wise max
        import pandas as pd
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR = moving average of TR
        atr = true_range.rolling(window=period).mean()
        
        # Get ATR value closest to target date
        atr_value = atr.iloc[-1] if not atr.empty else None
        
        if atr_value is not None and atr_value > 0:
            logger.debug(f"ATR for {ticker} on {date}: {atr_value:.2f}")
            return float(atr_value)
        
        return None
        
    except Exception as e:
        logger.error(f"Error calculating ATR for {ticker} on {date}: {e}")
        return None


def calculate_volatility(ticker: str, date: str, period: int = 20) -> Optional[float]:
    """
    Calculate historical volatility (standard deviation of returns).
    
    Args:
        ticker: Asset ticker symbol
        date: Date string in YYYY-MM-DD format
        period: Lookback period (default 20)
    
    Returns:
        Volatility value or None if calculation fails
    """
    try:
        date_fmt = "%Y-%m-%d %H:%M:%S" if " " in date else "%Y-%m-%d"
        target_date = datetime.strptime(date, date_fmt)
        
        start_date = target_date - timedelta(days=period * 3)
        end_date = target_date + timedelta(days=1)
        
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        
        if data.empty or len(data) < period:
            return None
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < period:
            return None
        
        # Calculate volatility (standard deviation)
        vol_series = returns.rolling(window=period).std()
        
        if len(vol_series) > 0:
            vol_value = vol_series.iloc[-1]
            # Handle both scalar and Series return types
            if isinstance(vol_value, pd.Series):
                vol_value = vol_value.iloc[0] if len(vol_value) > 0 else None
            if vol_value is not None and not pd.isna(vol_value) and vol_value > 0:
                return float(vol_value)
        
        return None
        
    except Exception as e:
        logger.error(f"Error calculating volatility for {ticker} on {date}: {e}")
        return None
