"""Tests for propagation.py — state initialization and graph args."""

import pytest
from tradingagents.graph.propagation import Propagator


@pytest.fixture
def propagator():
    return Propagator(max_recur_limit=50)


class TestCreateInitialState:
    def test_has_required_keys(self, propagator):
        state = propagator.create_initial_state("BTC-USD", "2024-01-15")
        required = [
            "messages", "company_of_interest", "trade_date",
            "investment_debate_state", "risk_debate_state",
            "market_report", "fundamentals_report",
            "sentiment_report", "news_report",
        ]
        for key in required:
            assert key in state, f"Missing key: {key}"

    def test_ticker_stored(self, propagator):
        state = propagator.create_initial_state("BTC-USD", "2024-01-15")
        assert state["company_of_interest"] == "BTC-USD"

    def test_date_stringified(self, propagator):
        state = propagator.create_initial_state("AAPL", "2024-01-15")
        assert state["trade_date"] == "2024-01-15"
        assert isinstance(state["trade_date"], str)

    def test_debate_state_initialized_zero(self, propagator):
        state = propagator.create_initial_state("AAPL", "2024-01-15")
        assert state["investment_debate_state"]["count"] == 0
        assert state["risk_debate_state"]["count"] == 0

    def test_reports_empty(self, propagator):
        state = propagator.create_initial_state("AAPL", "2024-01-15")
        assert state["market_report"] == ""
        assert state["sentiment_report"] == ""


class TestGetGraphArgs:
    def test_default_args(self, propagator):
        args = propagator.get_graph_args()
        assert args["stream_mode"] == "values"
        assert args["config"]["recursion_limit"] == 50

    def test_callbacks_passed(self, propagator):
        cb = [lambda x: x]
        args = propagator.get_graph_args(callbacks=cb)
        assert args["config"]["callbacks"] == cb

    def test_no_callbacks(self, propagator):
        args = propagator.get_graph_args()
        assert "callbacks" not in args["config"]
