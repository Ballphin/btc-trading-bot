from unittest.mock import Mock, patch

from pydantic import BaseModel

from tradingagents.hedgefund.utils.llm import call_llm


class DummySignal(BaseModel):
    signal: str
    confidence: float
    reasoning: str


def _default_signal() -> DummySignal:
    return DummySignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")


class _NoJsonModelInfo:
    def has_json_mode(self) -> bool:
        return False


def test_call_llm_appends_model_init_error_to_reasoning():
    with patch("tradingagents.hedgefund.utils.llm.get_model", side_effect=RuntimeError("429 Too Many Requests")):
        out = call_llm(
            prompt="test",
            pydantic_model=DummySignal,
            max_retries=1,
            default_factory=_default_signal,
        )

    assert out.signal == "neutral"
    assert "error:" in out.reasoning
    assert "RuntimeError" in out.reasoning
    assert "429 Too Many Requests" in out.reasoning


def test_call_llm_appends_parse_error_for_non_json_models():
    llm = Mock()
    llm.invoke.return_value = Mock(content="this is not json")

    with patch("tradingagents.hedgefund.utils.llm.get_model_info", return_value=_NoJsonModelInfo()):
        with patch("tradingagents.hedgefund.utils.llm.get_model", return_value=llm):
            out = call_llm(
                prompt="test",
                pydantic_model=DummySignal,
                max_retries=1,
                default_factory=_default_signal,
            )

    assert out.signal == "neutral"
    assert "error:" in out.reasoning
    assert "Model response did not contain parseable JSON" in out.reasoning
