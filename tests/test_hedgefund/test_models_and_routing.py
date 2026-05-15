from unittest.mock import patch

from tradingagents.hedgefund.data.models import LineItem
from tradingagents.hedgefund.llm.models import ModelProvider, get_model


class TestLineItemDynamicFields:
    def test_missing_dynamic_field_returns_none(self):
        item = LineItem(
            ticker="AAPL",
            report_period="2024-12-31",
            period="annual",
            currency="USD",
        )
        assert item.revenue is None
        assert item.some_unknown_field is None

    def test_extra_dynamic_field_is_accessible(self):
        item = LineItem(
            ticker="AAPL",
            report_period="2024-12-31",
            period="annual",
            currency="USD",
            revenue=123.45,
        )
        assert item.revenue == 123.45


class TestDeepSeekRouting:
    def test_deepseek_forced_nvidia_route_normalizes_model_and_base_url(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_USE_NVIDIA", "1")
        monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
        monkeypatch.setenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com")
        monkeypatch.setenv("NVIDIA_DEEPSEEK_MODEL", "deepseek-v4-pro")
        monkeypatch.setenv("HEDGEFUND_LLM_TIMEOUT_S", "45")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        with patch("tradingagents.hedgefund.llm.models.ChatOpenAI") as mock_openai:
            sentinel = object()
            mock_openai.return_value = sentinel

            out = get_model("deepseek-v4-pro", ModelProvider.DEEPSEEK)

            assert out is sentinel
            mock_openai.assert_called_once_with(
                model="deepseek-ai/deepseek-v4-pro",
                api_key="nv-key",
                base_url="https://integrate.api.nvidia.com/v1",
                timeout=45.0,
            )

    def test_deepseek_uses_native_provider_when_deepseek_key_present(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_USE_NVIDIA", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        monkeypatch.setenv("HEDGEFUND_LLM_TIMEOUT_S", "45")
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

        with patch("tradingagents.hedgefund.llm.models.ChatDeepSeek") as mock_deepseek:
            sentinel = object()
            mock_deepseek.return_value = sentinel

            out = get_model("deepseek-v4-pro", ModelProvider.DEEPSEEK)

            assert out is sentinel
            mock_deepseek.assert_called_once_with(
                model="deepseek-v4-pro",
                api_key="ds-key",
                timeout=45.0,
            )

    def test_deepseek_uses_nvidia_only_when_use_nvidia_flag_set(self, monkeypatch):
        # New contract: NVIDIA route is opt-in, not auto-selected from key presence.
        monkeypatch.delenv("DEEPSEEK_USE_NVIDIA", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
        monkeypatch.setenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
        monkeypatch.setenv("HEDGEFUND_LLM_TIMEOUT_S", "45")

        with patch("tradingagents.hedgefund.llm.models.ChatOpenAI") as mock_openai:
            sentinel = object()
            mock_openai.return_value = sentinel

            out = get_model(
                "deepseek-v4-pro",
                ModelProvider.DEEPSEEK,
                use_nvidia=True,
            )

            assert out is sentinel
            mock_openai.assert_called_once_with(
                model="deepseek-ai/deepseek-v4-pro",
                api_key="nv-key",
                base_url="https://integrate.api.nvidia.com/v1",
                timeout=45.0,
            )

    def test_deepseek_without_flag_requires_deepseek_key(self, monkeypatch):
        # Without flag and without DEEPSEEK_API_KEY, must raise — never silently route to NVIDIA.
        monkeypatch.delenv("DEEPSEEK_USE_NVIDIA", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")

        import pytest
        with pytest.raises(ValueError, match="DeepSeek API key not found"):
            get_model("deepseek-v4-pro", ModelProvider.DEEPSEEK)
