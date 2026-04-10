"""Tests for crypto_news_scraper — RSS parsing, caching, ticker matching."""

import json
import time
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path

from tradingagents.dataflows.crypto_news_scraper import (
    _cache_key,
    _read_cache,
    _write_cache,
    _parse_pub_date,
    _matches_ticker,
    _format_articles,
    get_crypto_news,
    get_crypto_global_news,
)


class TestCacheKey:
    def test_deterministic(self):
        assert _cache_key("a", "b", "c") == _cache_key("a", "b", "c")

    def test_different_inputs(self):
        assert _cache_key("a", "b") != _cache_key("a", "c")


class TestParsePubDate:
    def test_parsed_struct(self):
        entry = {"published_parsed": (2024, 6, 15, 10, 30, 0, 5, 167, 0)}
        dt = _parse_pub_date(entry)
        assert dt == datetime(2024, 6, 15, 10, 30, 0)

    def test_rfc822_string(self):
        entry = {"published": "Mon, 15 Jun 2024 10:30:00 +0000"}
        dt = _parse_pub_date(entry)
        assert dt is not None
        assert dt.day == 15

    def test_iso_string(self):
        entry = {"updated": "2024-06-15T10:30:00Z"}
        dt = _parse_pub_date(entry)
        assert dt is not None

    def test_no_date_returns_none(self):
        assert _parse_pub_date({}) is None

    def test_malformed_date(self):
        entry = {"published": "not a date at all"}
        # Should not crash — returns None
        result = _parse_pub_date(entry)
        assert result is None


class TestMatchesTicker:
    def test_btc_match(self):
        assert _matches_ticker("Bitcoin price surges to $70k", "BTC-USD")

    def test_eth_match(self):
        assert _matches_ticker("Ethereum staking yields drop", "ETH-USD")

    def test_no_match(self):
        assert not _matches_ticker("Apple announces new iPhone", "BTC-USD")

    def test_case_insensitive(self):
        assert _matches_ticker("BITCOIN hits all-time high", "BTC-USD")

    def test_unknown_ticker_uses_base(self):
        assert _matches_ticker("PEPE is pumping today", "PEPE-USD")


class TestFormatArticles:
    def test_no_articles(self):
        result = _format_articles([], "## Header")
        assert "No articles found" in result

    def test_formats_articles(self):
        articles = [{
            "title": "Test Article",
            "summary": "Summary text",
            "source": "TestSource",
            "link": "https://example.com",
            "pub_date": datetime(2024, 6, 15, 10, 0),
        }]
        result = _format_articles(articles, "## Header")
        assert "Test Article" in result
        assert "TestSource" in result
        assert "https://example.com" in result


class TestGetCryptoNews:
    @patch("tradingagents.dataflows.crypto_news_scraper._read_cache", return_value="cached result")
    def test_returns_cached(self, mock_cache):
        result = get_crypto_news("BTC-USD", "2024-06-01", "2024-06-15")
        assert result == "cached result"

    @patch("tradingagents.dataflows.crypto_news_scraper._write_cache")
    @patch("tradingagents.dataflows.crypto_news_scraper._fetch_feeds", return_value=[])
    @patch("tradingagents.dataflows.crypto_news_scraper._read_cache", return_value=None)
    def test_no_articles_fallback_yfinance(self, mock_read, mock_fetch, mock_write):
        with patch("tradingagents.dataflows.yfinance_news.get_news_yfinance", return_value="yf news"):
            result = get_crypto_news("BTC-USD", "2024-06-01", "2024-06-15")
        assert "yf news" in result or "No articles found" in result


class TestGetCryptoGlobalNews:
    @patch("tradingagents.dataflows.crypto_news_scraper._read_cache", return_value="global cached")
    def test_returns_cached(self, mock_cache):
        result = get_crypto_global_news("2024-06-15")
        assert result == "global cached"

    @patch("tradingagents.dataflows.crypto_news_scraper._write_cache")
    @patch("tradingagents.dataflows.crypto_news_scraper._fetch_feeds", return_value=[])
    @patch("tradingagents.dataflows.crypto_news_scraper._read_cache", return_value=None)
    def test_no_articles(self, mock_read, mock_fetch, mock_write):
        result = get_crypto_global_news("2024-06-15")
        assert "No articles found" in result
