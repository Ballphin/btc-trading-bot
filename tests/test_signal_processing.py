"""Tests for signal processing with expanded action space."""

import pytest
from unittest.mock import MagicMock


class TestSignalProcessorActions:
    """Verify SHORT/COVER are accepted by SignalProcessor."""

    def _make_processor(self, return_value: str):
        from tradingagents.graph.signal_processing import SignalProcessor
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=return_value)
        return SignalProcessor(mock_llm)

    def test_extracts_buy(self):
        sp = self._make_processor("BUY")
        assert sp.process_signal("I recommend buying") == "BUY"

    def test_extracts_sell(self):
        sp = self._make_processor("SELL")
        assert sp.process_signal("I recommend selling") == "SELL"

    def test_extracts_hold(self):
        sp = self._make_processor("HOLD")
        assert sp.process_signal("Hold position") == "HOLD"

    def test_extracts_short(self):
        sp = self._make_processor("SHORT")
        assert sp.process_signal("Enter short position") == "SHORT"

    def test_extracts_cover(self):
        sp = self._make_processor("COVER")
        assert sp.process_signal("Cover the short") == "COVER"

    def test_extracts_overweight(self):
        sp = self._make_processor("OVERWEIGHT")
        assert sp.process_signal("Increase position") == "OVERWEIGHT"

    def test_extracts_underweight(self):
        sp = self._make_processor("UNDERWEIGHT")
        assert sp.process_signal("Reduce exposure") == "UNDERWEIGHT"

    def test_prompt_includes_short_cover(self):
        """Verify the system prompt sent to the LLM includes SHORT and COVER."""
        from tradingagents.graph.signal_processing import SignalProcessor
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="BUY")
        sp = SignalProcessor(mock_llm)
        sp.process_signal("test")

        # Check the system message content
        call_args = mock_llm.invoke.call_args[0][0]
        system_msg = call_args[0][1]  # (role, content) tuple
        assert "SHORT" in system_msg
        assert "COVER" in system_msg
