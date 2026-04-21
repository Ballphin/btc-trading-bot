"""Final-proposal-line extraction tests for SignalProcessor.

Regression coverage for the Apr-2026 "HOLD vs SHORT" mismatch where the
UI badge disagreed with the trailing ``FINAL TRANSACTION PROPOSAL``
line because the LLM fallback extractor got distracted by hedging prose
earlier in the judge's output.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from tradingagents.graph.signal_processing import (
    SignalProcessor,
    _extract_from_proposal_line,
)


# The exact screenshot prose that triggered the bug (paraphrased from
# the user's report — HOLD badge, SHORT proposal).
SCREENSHOT_PROSE = """\
Based on the provided investment plan and my analysis of the technical and
fundamental confluence, the current price action near major resistance
($486-$520) presents a high-risk, low-reward setup for initiating a long
position. The plan correctly identifies that the bull case, while
fundamentally strong, requires a confirmed breakout above $525 to become
actionable. Conversely, the bear case highlights significant vulnerability
at this technical ceiling, exacerbated by stretched valuations and high
leverage.

The strategic recommendation within the plan is to adopt a short bias,
waiting for a confirmed rejection at resistance to act. Given that the
stock is currently stalling in this critical zone after a substantial
bounce, the prudent tactical decision is to position for a downside move.
The optimal action is to initiate a short position, capitalizing on the
anticipated failure at resistance, with strict risk management above the
200-day SMA.

FINAL TRANSACTION PROPOSAL: SHORT
"""


# ── Pure regex helper ────────────────────────────────────────────────

class TestProposalRegex:
    def test_basic_match(self):
        assert _extract_from_proposal_line("FINAL TRANSACTION PROPOSAL: SHORT") == "SHORT"

    def test_case_insensitive(self):
        assert _extract_from_proposal_line(
            "final transaction proposal: hold"
        ) == "HOLD"

    def test_markdown_bold_wrapper(self):
        assert _extract_from_proposal_line(
            "FINAL TRANSACTION PROPOSAL: **SHORT**"
        ) == "SHORT"

    def test_no_colon_form(self):
        assert _extract_from_proposal_line(
            "FINAL TRANSACTION PROPOSAL BUY"
        ) == "BUY"

    def test_last_proposal_wins_over_earlier(self):
        """Debate transcripts commonly contain intermediate proposals
        from bulls/bears; the judge's final override comes last."""
        txt = (
            "Bull argues: FINAL TRANSACTION PROPOSAL: BUY. "
            "Bear rebuts: FINAL TRANSACTION PROPOSAL: SHORT. "
            "Judge concludes: FINAL TRANSACTION PROPOSAL: HOLD."
        )
        assert _extract_from_proposal_line(txt) == "HOLD"

    def test_no_marker_returns_none(self):
        assert _extract_from_proposal_line("no marker here, just prose") is None

    def test_empty_input(self):
        assert _extract_from_proposal_line("") is None
        assert _extract_from_proposal_line(None) is None

    @pytest.mark.parametrize("rating",
        ["BUY", "SELL", "SHORT", "COVER", "HOLD", "OVERWEIGHT", "UNDERWEIGHT"])
    def test_all_seven_ratings_extractable(self, rating):
        assert _extract_from_proposal_line(
            f"FINAL TRANSACTION PROPOSAL: {rating}"
        ) == rating


# ── Integration tests against SignalProcessor ───────────────────────

def _mk_processor(mock_extract: str = "HOLD") -> SignalProcessor:
    """Build a SignalProcessor whose LLM is a MagicMock — isolates
    tests from real LLM calls. ``mock_extract`` is what the LLM would
    return if it were called."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=mock_extract)
    return SignalProcessor(llm)


def test_final_proposal_line_wins_over_hedging_prose():
    """The Apr-2026 regression: the screenshot prose must extract as
    SHORT even though a naive LLM pass would return HOLD."""
    processor = _mk_processor(mock_extract="HOLD")  # the bug: LLM returns HOLD
    result = processor.process_signal(SCREENSHOT_PROSE)
    assert result == "SHORT", (
        "regex fix must preempt the LLM fallback — FINAL TRANSACTION "
        "PROPOSAL: SHORT is the authoritative signal"
    )
    # Critical: the LLM should NEVER have been called.
    processor.quick_thinking_llm.invoke.assert_not_called()


def test_last_proposal_wins_when_multiple():
    processor = _mk_processor()
    txt = (
        "Bull proposes: FINAL TRANSACTION PROPOSAL: BUY.\n"
        "Bear counters: FINAL TRANSACTION PROPOSAL: SHORT.\n"
        "Judge concludes the short thesis is stronger. "
        "FINAL TRANSACTION PROPOSAL: SHORT"
    )
    assert processor.process_signal(txt) == "SHORT"


def test_legacy_path_without_marker_falls_through_to_llm():
    """Preserve backward compatibility: judge prose without the marker
    still triggers the LLM extractor."""
    processor = _mk_processor(mock_extract="BUY")
    result = processor.process_signal(
        "The thesis is bullish given strong momentum and clean tape."
    )
    assert result == "BUY"
    processor.quick_thinking_llm.invoke.assert_called_once()


def test_junk_llm_output_collapses_to_hold():
    """Defensive: if the LLM returns an unknown string, we don't
    propagate it — default to HOLD."""
    processor = _mk_processor(mock_extract="MAYBE?")
    result = processor.process_signal("no marker, ambiguous prose")
    assert result == "HOLD"


def test_json_signal_disagreeing_with_proposal_prefers_proposal(caplog):
    """When the LLM emits BOTH a JSON block with one signal AND a
    trailing FINAL TRANSACTION PROPOSAL with a different one, the
    proposal line wins + we log a warning."""
    processor = _mk_processor()
    signal = """Here is my analysis:
```json
{
  "signal": "HOLD",
  "stop_loss_price": 0,
  "take_profit_price": 0,
  "confidence": 0.55,
  "max_hold_days": 7,
  "reasoning": "Too much uncertainty near resistance."
}
```

On reflection, the bear case is decisive.

FINAL TRANSACTION PROPOSAL: SHORT
"""
    with caplog.at_level(logging.WARNING, logger="tradingagents.graph.signal_processing"):
        result = processor.process_signal(signal)
    assert isinstance(result, dict)
    assert result["signal"] == "SHORT", "proposal line must override JSON disagreement"
    # SL/TP from the HOLD JSON must be invalidated — they were computed
    # for a HOLD (zeros here), but more importantly a long-stop on a
    # short is worse than no stop.
    assert result["stop_loss_price"] == 0
    assert result["take_profit_price"] == 0
    # Confirm the disagreement was logged (easier to spot prompt regressions).
    assert any("JSON/proposal disagreement" in rec.message for rec in caplog.records)


def test_json_agreeing_with_proposal_is_silent(caplog):
    """No warning when JSON and proposal agree — the common case must
    stay quiet so warnings have signal."""
    processor = _mk_processor()
    signal = """```json
{
  "signal": "SHORT",
  "stop_loss_price": 525,
  "take_profit_price": 440,
  "confidence": 0.72,
  "max_hold_days": 5,
  "reasoning": "Rejection at resistance."
}
```
FINAL TRANSACTION PROPOSAL: SHORT
"""
    with caplog.at_level(logging.WARNING, logger="tradingagents.graph.signal_processing"):
        result = processor.process_signal(signal)
    assert result["signal"] == "SHORT"
    assert not any("disagreement" in rec.message for rec in caplog.records)


def test_stops_flipped_against_direction_get_nulled_on_override():
    """When the override flips LONG→SHORT, a long-oriented SL (below
    entry) + TP (above entry) is actively dangerous — null them so the
    downstream sizing path computes defaults from the regime."""
    processor = _mk_processor()
    signal = """```json
{
  "signal": "BUY",
  "stop_loss_price": 100,
  "take_profit_price": 120,
  "confidence": 0.70,
  "max_hold_days": 5,
  "reasoning": "Long thesis."
}
```
FINAL TRANSACTION PROPOSAL: SHORT
"""
    result = processor.process_signal(signal)
    assert result["signal"] == "SHORT"
    assert result["stop_loss_price"] == 0
    assert result["take_profit_price"] == 0
