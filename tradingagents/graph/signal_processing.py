# TradingAgents/graph/signal_processing.py

import json
import logging
import re
from typing import Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Deterministic extractor for the team's stop-signal convention
# (see the four analyst system prompts under tradingagents/agents/analysts/*).
# Every agent is instructed to prefix/suffix its final answer with
# ``FINAL TRANSACTION PROPOSAL: **<RATING>**`` — this IS the authoritative
# finale. Matching it with a regex is ~zero cost and eliminates the
# failure mode where the fallback LLM latches onto earlier hedging prose
# (the "HOLD vs SHORT" bug in the Apr-2026 NVDA screenshot).
_RATINGS = ("BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL", "SHORT", "COVER")
_PROPOSAL_RE = re.compile(
    r"FINAL\s+TRANSACTION\s+PROPOSAL\s*:?[\s*]*"
    r"(BUY|SELL|SHORT|COVER|HOLD|OVERWEIGHT|UNDERWEIGHT)\b",
    re.IGNORECASE,
)


def _extract_from_proposal_line(text: str) -> Optional[str]:
    """Return the LAST ``FINAL TRANSACTION PROPOSAL: X`` rating in ``text``,
    or None if no such line exists.

    "Last" matters: debate-style transcripts often contain intermediate
    proposals (bull rebuttal → "PROPOSAL: BUY", bear rebuttal → "PROPOSAL:
    SHORT"), and the judge's override appears last by construction.
    """
    if not text:
        return None
    matches = _PROPOSAL_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper()


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> Union[str, Dict[str, Any]]:
        """
        Process a full trading signal to extract the core decision.
        
        Attempts to parse structured JSON first. If that fails, falls back to
        extracting just the signal word for backward compatibility.

        Args:
            full_signal: Complete trading signal text (JSON or freestyle)

        Returns:
            Dict with structured fields if JSON parsing succeeds, otherwise just the signal string
        """
        # Try to parse as JSON first
        structured_signal = self._try_parse_json(full_signal)
        if structured_signal:
            # Cross-check: when the portfolio_manager LLM emits BOTH a
            # JSON block AND a trailing FINAL TRANSACTION PROPOSAL line
            # that disagrees, prefer the proposal line — that line is
            # the team's contractually-authoritative stop signal and the
            # JSON is often the hedged/templated prelude. Log the
            # disagreement loudly so we can spot prompt regressions.
            proposal = _extract_from_proposal_line(full_signal)
            if proposal and proposal != structured_signal.get("signal"):
                logger.warning(
                    "[SignalProcessor] JSON/proposal disagreement: "
                    "json=%s proposal=%s — preferring proposal line",
                    structured_signal.get("signal"), proposal,
                )
                structured_signal["signal"] = proposal
                # Stops from the JSON block are likely computed for the
                # JSON's signal direction, not the override. Null them
                # so downstream sizing falls back to regime defaults
                # rather than using a long stop on a short trade.
                if proposal in ("SHORT", "SELL", "COVER"):
                    if (structured_signal.get("stop_loss_price") or 0) < (
                        structured_signal.get("take_profit_price") or 0
                    ):
                        structured_signal["stop_loss_price"] = 0
                        structured_signal["take_profit_price"] = 0
                elif proposal in ("BUY", "OVERWEIGHT"):
                    if (structured_signal.get("stop_loss_price") or 0) > (
                        structured_signal.get("take_profit_price") or 0
                    ):
                        structured_signal["stop_loss_price"] = 0
                        structured_signal["take_profit_price"] = 0
            return structured_signal
        
        # Fallback: extract signal word using LLM
        return self._extract_signal_word(full_signal)
    
    def _try_parse_json(self, text: str) -> Dict[str, Any]:
        """
        Attempt to extract and parse JSON from the text.
        
        Returns:
            Parsed dict if successful, None otherwise
        """
        # Try direct JSON parse first
        try:
            data = json.loads(text.strip())
            if self._validate_structured_signal(data):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try to extract JSON from markdown code blocks
        json_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                if self._validate_structured_signal(data):
                    return data
            except (json.JSONDecodeError, ValueError):
                continue
        
        # Try to find any JSON object in the text
        json_obj_pattern = r'\{[^{}]*"signal"[^{}]*\}'
        matches = re.findall(json_obj_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                if self._validate_structured_signal(data):
                    return data
            except (json.JSONDecodeError, ValueError):
                continue
        
        return None
    
    def _validate_structured_signal(self, data: Dict[str, Any]) -> bool:
        """
        Validate that the parsed JSON has required fields.
        
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['signal', 'stop_loss_price', 'take_profit_price', 'confidence', 'max_hold_days']
        
        if not isinstance(data, dict):
            return False
        
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate signal is one of the allowed values
        valid_signals = ['BUY', 'SELL', 'SHORT', 'COVER', 'HOLD', 'OVERWEIGHT', 'UNDERWEIGHT']
        if data['signal'].upper() not in valid_signals:
            return False
        
        # Normalize signal to uppercase
        data['signal'] = data['signal'].upper()

        # Validate confidence is a float in [0.0, 1.0]
        conf = data.get('confidence')
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            return False
        data['confidence'] = float(conf)

        # Validate max_hold_days is a positive integer; clamp to [1, 90]
        hold = data.get('max_hold_days')
        if not isinstance(hold, (int, float)) or int(hold) < 1:
            return False
        data['max_hold_days'] = max(1, min(90, int(hold)))

        # Validate reasoning is a non-trivial string
        if len(str(data.get('reasoning', '')).strip()) < 10:
            return False

        # Directional check: stop/take relationship must match signal direction
        sig = data['signal']
        sl = data.get('stop_loss_price', 0) or 0
        tp = data.get('take_profit_price', 0) or 0

        # Consistency check: a directional signal with both stops zeroed is
        # self-contradictory — only HOLD legitimately has sl=0/tp=0 per the
        # portfolio_manager prompt. When the LLM emits e.g. signal=SHORT
        # with stops=0 (see NVDA 2026-04-19 case where prose said HOLD but
        # JSON said SHORT), reject so process_signal falls through to the
        # LLM-based signal extractor which reads the "FINAL TRANSACTION
        # PROPOSAL" line and returns the correct tag.
        directional = {'BUY', 'SELL', 'SHORT', 'COVER', 'OVERWEIGHT', 'UNDERWEIGHT'}
        if sig in directional and sl == 0 and tp == 0:
            return False

        if sig in ('BUY', 'OVERWEIGHT') and sl > 0 and tp > 0:
            if not (tp > sl):
                return False
        if sig in ('SHORT', 'SELL') and sl > 0 and tp > 0:
            if not (sl > tp):
                return False

        return True
    
    def _extract_signal_word(self, full_signal: str) -> str:
        """
        Fallback method: extract just the signal word.

        Resolution order (cheapest + most deterministic first):
          1. Regex on the team's ``FINAL TRANSACTION PROPOSAL: X`` stop-signal
             convention — authoritative per the analyst system prompts.
          2. LLM extraction — only when the marker is absent (legacy logs,
             degraded judge output).

        The two-tier ordering closes the bug where an LLM extractor, faced
        with judge prose that opens with hedging language ("high-risk,
        low-reward") before the decisive turn, would latch onto the early
        tone and return HOLD despite a trailing ``FINAL TRANSACTION
        PROPOSAL: SHORT`` line.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted rating (BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL, SHORT, or COVER)
        """
        proposal = _extract_from_proposal_line(full_signal)
        if proposal:
            return proposal

        messages = [
            (
                "system",
                "You are an efficient assistant that extracts the trading decision from analyst reports. "
                "Extract the rating as exactly one of: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL, SHORT, COVER. "
                "Output only the single rating word, nothing else.",
            ),
            ("human", full_signal),
        ]

        extracted = self.quick_thinking_llm.invoke(messages).content.strip().upper()
        # Defensive: if the LLM returns junk, collapse to HOLD rather than
        # propagate an unknown rating through the pipeline.
        for r in _RATINGS:
            if r in extracted:
                return r
        return "HOLD"
