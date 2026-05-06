import datetime
import logging
import os
import pandas as pd
import yfinance as yf
import time

logger = logging.getLogger(__name__)

from tradingagents.hedgefund.data.cache import get_cache
from tradingagents.hedgefund.data.models import (
    CompanyNews,
    FinancialMetrics,
    Price,
    LineItem,
    InsiderTrade,
)

# Global cache instance
_cache = get_cache()

def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from cache or yfinance."""
    cache_key = f"{ticker}_{start_date}_{end_date}"
    
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if df.empty:
            return []
            
        # yfinance returns MultiIndex columns if multiple tickers, but we pass one ticker
        # Flatten columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna(subset=['Open', 'Close', 'Volume'])
        df = df.reset_index()
        prices = []
        for _, row in df.iterrows():
            prices.append(Price(
                open=float(row['Open']),
                close=float(row['Close']),
                high=float(row['High']),
                low=float(row['Low']),
                volume=int(row['Volume']),
                time=row['Date'].strftime('%Y-%m-%d')
            ))
            
        _cache.set_prices(cache_key, [p.model_dump() for p in prices])
        return prices
    except Exception as e:
        logger.warning(f"Failed to fetch prices for {ticker}: {e}")
        return []

def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or yfinance."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    try:
        info = yf.Ticker(ticker).info
        
        # Prevent lookahead bias for historical requests
        today = datetime.datetime.now()
        req_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        is_historical = (today - req_date).days > 7
        
        metric = FinancialMetrics(
            ticker=ticker,
            report_period=end_date,
            period=period,
            currency=info.get("currency", "USD"),
            market_cap=info.get("marketCap") if not is_historical else None,
            enterprise_value=info.get("enterpriseValue") if not is_historical else None,
            price_to_earnings_ratio=info.get("trailingPE") if not is_historical else None,
            price_to_book_ratio=info.get("priceToBook") if not is_historical else None,
            price_to_sales_ratio=info.get("priceToSalesTrailing12Months") if not is_historical else None,
            enterprise_value_to_ebitda_ratio=None, 
            enterprise_value_to_revenue_ratio=info.get("enterpriseToRevenue") if not is_historical else None,
            free_cash_flow_yield=None,
            peg_ratio=info.get("pegRatio") if not is_historical else None,
            gross_margin=info.get("grossMargins"),
            operating_margin=info.get("operatingMargins"),
            net_margin=info.get("profitMargins"),
            return_on_equity=info.get("returnOnEquity"),
            return_on_assets=info.get("returnOnAssets"),
            return_on_invested_capital=None,
            asset_turnover=None,
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=info.get("currentRatio"),
            quick_ratio=info.get("quickRatio"),
            cash_ratio=None,
            operating_cash_flow_ratio=None,
            debt_to_equity=info.get("debtToEquity"),
            debt_to_assets=None,
            interest_coverage=None,
            revenue_growth=info.get("revenueGrowth"),
            earnings_growth=info.get("earningsGrowth"),
            book_value_growth=None,
            earnings_per_share_growth=None,
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=info.get("payoutRatio"),
            earnings_per_share=info.get("trailingEps"),
            book_value_per_share=info.get("bookValue"),
            free_cash_flow_per_share=None,
        )
        
        financial_metrics = [metric]
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
        return financial_metrics
    except Exception as e:
        logger.warning(f"Failed to fetch metrics for {ticker}: {e}")
        return []

def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items from yfinance."""
    try:
        t = yf.Ticker(ticker)
        
        if period == "ttm":
            income = t.quarterly_income_stmt
            balance = t.quarterly_balance_sheet
            cashflow = t.quarterly_cashflow
            
            # Sum the first 4 columns to get TTM for flow metrics, keep most recent for balance sheet
            if not income.empty and len(income.columns) >= 4:
                income = income.iloc[:, :4].sum(axis=1).to_frame(name=income.columns[0])
            if not cashflow.empty and len(cashflow.columns) >= 4:
                cashflow = cashflow.iloc[:, :4].sum(axis=1).to_frame(name=cashflow.columns[0])
            if not balance.empty and len(balance.columns) >= 1:
                balance = balance.iloc[:, :1] # Balance sheet is point-in-time, don't sum
        else:
            income = t.income_stmt
            balance = t.balance_sheet
            cashflow = t.cashflow
        
        if income.empty and balance.empty and cashflow.empty:
            return []
            
        # Combine all available statements
        combined = pd.concat([income, balance, cashflow])
        # Drop duplicates if any overlap
        combined = combined[~combined.index.duplicated(keep='first')]
        
        # Hardcoded mapping dict for exact matches
        mapping_dict = {
            "net_income": "Net Income",
            "revenue": "Total Revenue",
            "total_revenue": "Total Revenue",
            "operating_income": "Operating Income",
            "free_cash_flow": "Free Cash Flow",
            "total_assets": "Total Assets",
            "total_liabilities": "Total Liabilities Net Minority Interest",
            "total_equity": "Stockholders Equity",
            "cash_and_equivalents": "Cash And Cash Equivalents",
            "total_debt": "Total Debt",
            "ebitda": "Normalized EBITDA",
            "ebit": "EBIT",
            "gross_profit": "Gross Profit",
            "cost_of_revenue": "Cost Of Revenue",
            "operating_expense": "Operating Expense",
            "net_debt": "Net Debt"
        }
        
        results = []
        # Iterate over the columns (which are dates)
        for date_col in combined.columns[:limit]:
            item_data = {
                "ticker": ticker,
                "report_period": date_col.strftime('%Y-%m-%d'),
                "period": period,
                "currency": "USD"
            }
            
            for req_item in line_items:
                match_val = None
                
                # Check exact dict mapping first
                if req_item in mapping_dict:
                    yf_name = mapping_dict[req_item]
                    if yf_name in combined.index:
                        match_val = combined.loc[yf_name, date_col]
                
                # Fallback to exact lowercase matching
                if match_val is None or pd.isna(match_val):
                    clean_req = req_item.replace('_', ' ').lower()
                    for idx in combined.index:
                        if clean_req == str(idx).lower():
                            match_val = combined.loc[idx, date_col]
                            break
                            
                if match_val is not None and not pd.isna(match_val):
                    item_data[req_item] = float(match_val)
                    
            results.append(LineItem(**item_data))
            
        return results
    except Exception as e:
        logger.warning(f"Failed to fetch line items for {ticker}: {e}")
        return []

def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or yfinance."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    try:
        t = yf.Ticker(ticker)
        trades_df = t.insider_transactions
        if trades_df is None or trades_df.empty:
            return []
            
        trades = []
        for _, row in trades_df.iterrows():
            # Yfinance returns columns like 'Insider', 'Position', 'URL', 'Transaction', 'Text', 'Value'
            # The format is highly variable, doing best-effort mapping
            try:
                date_val = str(row.get('Start Date', row.name))
                if isinstance(date_val, str) and ' ' in date_val:
                    date_val = date_val.split(' ')[0]
                    
                shares = float(row.get('Shares', 0))
                value = float(row.get('Value', 0))
                price = value / shares if shares > 0 else 0
                
                trades.append(InsiderTrade(
                    ticker=ticker,
                    issuer=ticker,
                    name=str(row.get('Insider', 'Unknown')),
                    title=str(row.get('Position', 'Unknown')),
                    is_board_director=False,
                    transaction_date=date_val,
                    transaction_shares=shares,
                    transaction_price_per_share=price,
                    transaction_value=value,
                    shares_owned_before_transaction=None,
                    shares_owned_after_transaction=None,
                    security_title="Common Stock",
                    filing_date=date_val
                ))
            except Exception:
                continue
                
        # Limit and filter by date if needed
        trades = trades[:limit]
        _cache.set_insider_trades(cache_key, [t.model_dump() for t in trades])
        return trades
    except Exception as e:
        logger.warning(f"Failed to fetch insider trades for {ticker}: {e}")
        return []

def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from cache or yfinance."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    try:
        news_items = yf.Ticker(ticker).news
        if not news_items:
            return []
            
        all_news = []
        for item in news_items:
            # yfinance returns timestamp in seconds
            pub_date = datetime.datetime.fromtimestamp(item.get('providerPublishTime', 0))
            all_news.append(CompanyNews(
                ticker=ticker,
                title=item.get('title', ''),
                author=None,
                source=item.get('publisher', ''),
                date=pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                url=item.get('link', ''),
                sentiment=None
            ))
            
        all_news = all_news[:limit]
        _cache.set_company_news(cache_key, [n.model_dump() for n in all_news])
        return all_news
    except Exception as e:
        logger.warning(f"Failed to fetch news for {ticker}: {e}")
        return []

def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("marketCap")
    except Exception:
        return None

def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    if not prices:
        return pd.DataFrame()
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df

def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
