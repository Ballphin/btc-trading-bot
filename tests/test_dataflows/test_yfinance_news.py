"""Tests for yfinance_news — article extraction, backtest label, date filtering."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from tradingagents.dataflows.yfinance_news import (
    _extract_article_data,
    get_news_yfinance,
    get_global_news_yfinance,
)


class TestExtractArticleData:
    def test_nested_content(self):
        article = {
            "content": {
                "title": "BTC surges",
                "summary": "Bitcoin hit $70k",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://reuters.com/btc"},
                "pubDate": "2024-06-15T10:00:00Z",
            }
        }
        data = _extract_article_data(article)
        assert data["title"] == "BTC surges"
        assert data["publisher"] == "Reuters"
        assert data["link"] == "https://reuters.com/btc"
        assert data["pub_date"] is not None

    def test_flat_structure(self):
        article = {"title": "ETH news", "publisher": "CoinDesk"}
        data = _extract_article_data(article)
        assert data["title"] == "ETH news"
        assert data["publisher"] == "CoinDesk"

    def test_missing_fields(self):
        data = _extract_article_data({"content": {}})
        assert data["title"] == "No title"
        assert data["publisher"] == "Unknown"


class TestGetNewsYfinance:
    @patch("tradingagents.dataflows.yfinance_news.yf.Ticker")
    def test_no_news_returns_message(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.get_news.return_value = []
        mock_ticker_cls.return_value = mock_ticker
        result = get_news_yfinance("AAPL", "2024-06-01", "2024-06-15")
        assert "No news found" in result

    @patch("tradingagents.dataflows.yfinance_news.yf.Ticker")
    def test_backtest_label_when_no_articles_match(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.get_news.return_value = [
            {
                "content": {
                    "title": "Future article",
                    "summary": "...",
                    "provider": {"displayName": "Test"},
                    "pubDate": "2030-01-01T00:00:00Z",
                }
            }
        ]
        mock_ticker_cls.return_value = mock_ticker
        result = get_news_yfinance("AAPL", "2024-06-01", "2024-06-15")
        assert "[BACKTEST]" in result

    @patch("tradingagents.dataflows.yfinance_news.yf.Ticker")
    def test_returns_formatted_articles(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.get_news.return_value = [
            {
                "content": {
                    "title": "Apple earnings beat",
                    "summary": "Great quarter",
                    "provider": {"displayName": "Bloomberg"},
                    "pubDate": "2024-06-10T12:00:00Z",
                }
            }
        ]
        mock_ticker_cls.return_value = mock_ticker
        result = get_news_yfinance("AAPL", "2024-06-01", "2024-06-15")
        assert "Apple earnings beat" in result
        assert "Bloomberg" in result

    @patch("tradingagents.dataflows.yfinance_news.yf.Ticker")
    def test_exception_returns_error_message(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = Exception("API down")
        result = get_news_yfinance("AAPL", "2024-06-01", "2024-06-15")
        assert "Error fetching news" in result
