"""RSS-based crypto news scraper for Cointelegraph, CoinDesk, and BeInCrypto.

Fetches articles from public RSS feeds, filters by date and ticker keyword,
deduplicates, and returns formatted markdown suitable for agent consumption.
Results are disk-cached with a configurable TTL.
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import feedparser

logger = logging.getLogger(__name__)

# ── Feed URLs ────────────────────────────────────────────────────────
CRYPTO_RSS_FEEDS: Dict[str, str] = {
    "Cointelegraph": "https://cointelegraph.com/rss",
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "BeInCrypto": "https://beincrypto.com/feed/",
}

# ── Ticker keyword mapping ───────────────────────────────────────────
TICKER_KEYWORDS: Dict[str, List[str]] = {
    "BTC": ["bitcoin", "btc", "₿"],
    "ETH": ["ethereum", "eth", "ether"],
    "SOL": ["solana", "sol"],
    "XRP": ["ripple", "xrp"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "AVAX": ["avalanche", "avax"],
    "DOT": ["polkadot", "dot"],
    "MATIC": ["polygon", "matic"],
    "LINK": ["chainlink", "link"],
    "UNI": ["uniswap", "uni"],
    "ATOM": ["cosmos", "atom"],
    "LTC": ["litecoin", "ltc"],
    "BNB": ["binance coin", "bnb"],
    "ARB": ["arbitrum", "arb"],
    "OP": ["optimism"],
}

CACHE_DIR = Path(__file__).resolve().parent / "data_cache" / "crypto_news"
CACHE_TTL_SECONDS = 900  # 15 minutes


def _cache_key(prefix: str, *parts: str) -> str:
    raw = f"{prefix}:{'|'.join(parts)}"
    return hashlib.md5(raw.encode()).hexdigest()


def _read_cache(key: str) -> Optional[str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if time.time() - data.get("ts", 0) > CACHE_TTL_SECONDS:
            return None
        return data.get("content")
    except Exception:
        return None


def _write_cache(key: str, content: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({"ts": time.time(), "content": content}))


def _parse_pub_date(entry: dict) -> Optional[datetime]:
    """Extract publication date from a feed entry."""
    for field in ("published_parsed", "updated_parsed"):
        parsed = entry.get(field)
        if parsed:
            try:
                return datetime(*parsed[:6])
            except Exception:
                continue
    for field in ("published", "updated"):
        raw = entry.get(field, "")
        if raw:
            try:
                return datetime.strptime(raw[:25].strip(), "%a, %d %b %Y %H:%M:%S")
            except Exception:
                pass
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                pass
    return None


def _matches_ticker(text: str, ticker: str) -> bool:
    """Check if text mentions the given ticker."""
    base = ticker.split("-")[0].upper()
    keywords = TICKER_KEYWORDS.get(base, [base.lower()])
    text_lower = text.lower()
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            return True
    return False


def _fetch_feeds(
    start_dt: datetime,
    end_dt: datetime,
    ticker_filter: Optional[str] = None,
    limit: int = 30,
) -> List[dict]:
    """Fetch and merge articles from all RSS feeds."""
    articles: List[dict] = []
    seen_titles: set = set()
    
    logger.info(f"Fetching RSS feeds for ticker={ticker_filter}, date range={start_dt.date()} to {end_dt.date()}")

    for source, url in CRYPTO_RSS_FEEDS.items():
        try:
            logger.debug(f"Fetching feed from {source}: {url}")
            feed = feedparser.parse(url)
            logger.info(f"{source}: fetched {len(feed.entries)} entries")
            
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                if not title or title in seen_titles:
                    continue

                pub_date = _parse_pub_date(entry)
                if pub_date and not (start_dt <= pub_date <= end_dt + timedelta(days=1)):
                    continue

                summary = entry.get("summary", entry.get("description", ""))
                # Strip HTML tags from summary
                summary = re.sub(r"<[^>]+>", "", summary).strip()
                if len(summary) > 300:
                    summary = summary[:297] + "..."

                link = entry.get("link", "")

                # Ticker filtering
                if ticker_filter:
                    searchable = f"{title} {summary}"
                    if not _matches_ticker(searchable, ticker_filter):
                        continue

                seen_titles.add(title)
                articles.append({
                    "title": title,
                    "summary": summary,
                    "source": source,
                    "link": link,
                    "pub_date": pub_date,
                })
        except Exception as e:
            logger.warning(f"Failed to fetch {source} RSS feed: {e}")
            continue
    
    logger.info(f"Total articles found: {len(articles)}")

    # Sort by date descending
    articles.sort(key=lambda a: a.get("pub_date") or datetime.min, reverse=True)
    return articles[:limit]


def _format_articles(articles: List[dict], header: str) -> str:
    if not articles:
        return f"{header}\n\nNo articles found.\n"
    lines = [header, ""]
    for a in articles:
        date_str = a["pub_date"].strftime("%Y-%m-%d %H:%M") if a.get("pub_date") else "Unknown date"
        lines.append(f"### {a['title']} (source: {a['source']}, {date_str})")
        if a["summary"]:
            lines.append(a["summary"])
        if a["link"]:
            lines.append(f"Link: {a['link']}")
        lines.append("")
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────

def get_crypto_news(ticker: str, start_date: str, end_date: str) -> str:
    """Fetch crypto news for a specific ticker from Cointelegraph, CoinDesk, BeInCrypto.
    Falls back to yfinance news if RSS feeds return no results.

    Args:
        ticker: e.g. "BTC-USD", "ETH-USD", "SOL"
        start_date: yyyy-mm-dd
        end_date: yyyy-mm-dd

    Returns:
        Formatted markdown string of relevant articles.
    """
    key = _cache_key("ticker_news", ticker, start_date, end_date)
    cached = _read_cache(key)
    if cached:
        return cached

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    articles = _fetch_feeds(start_dt, end_dt, ticker_filter=ticker, limit=20)
    
    # If no articles found, try yfinance as fallback
    if not articles:
        logger.info(f"No RSS articles found for {ticker}, trying yfinance fallback")
        try:
            from tradingagents.dataflows.yfinance_news import get_news_yfinance
            yf_result = get_news_yfinance(ticker, start_date, end_date)
            if "No news found" not in yf_result:
                logger.info(f"Found articles via yfinance for {ticker}")
                return yf_result
        except Exception as e:
            logger.warning(f"yfinance news fallback failed: {e}")
    
    header = f"## Crypto News for {ticker}, from {start_date} to {end_date}:"
    result = _format_articles(articles, header)

    _write_cache(key, result)
    return result


def get_crypto_global_news(curr_date: str, look_back_days: int = 7, limit: int = 15) -> str:
    """Fetch broad crypto market news from Cointelegraph, CoinDesk, BeInCrypto.

    Args:
        curr_date: yyyy-mm-dd
        look_back_days: number of days to look back
        limit: max articles

    Returns:
        Formatted markdown string of global crypto news.
    """
    start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    key = _cache_key("global_news", start_date, curr_date, str(limit))
    cached = _read_cache(key)
    if cached:
        return cached

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")

    articles = _fetch_feeds(start_dt, end_dt, ticker_filter=None, limit=limit)
    header = f"## Global Crypto News, from {start_date} to {curr_date}:"
    result = _format_articles(articles, header)

    _write_cache(key, result)
    return result
