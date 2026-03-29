# TradingAgents/graph/signal_processing.py

import json
import re
from typing import Dict, Any, Union
from langchain_openai import ChatOpenAI


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
        
        return True
    
    def _extract_signal_word(self, full_signal: str) -> str:
        """
        Fallback method: extract just the signal word using LLM.
        
        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted rating (BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL, SHORT, or COVER)
        """
        messages = [
            (
                "system",
                "You are an efficient assistant that extracts the trading decision from analyst reports. "
                "Extract the rating as exactly one of: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL, SHORT, COVER. "
                "Output only the single rating word, nothing else.",
            ),
            ("human", full_signal),
        ]

        return self.quick_thinking_llm.invoke(messages).content.strip().upper()
