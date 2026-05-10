from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import server as server_module
from tradingagents.hedgefund.utils.analysts import ANALYST_CONFIG


client = TestClient(server_module.app)


def _valid_analyst() -> str:
    return next(iter(ANALYST_CONFIG.keys()))


class TestHedgeFundAnalyzeValidation:
    def test_rejects_unknown_analyst(self):
        resp = client.post(
            "/api/hedgefund/analyze",
            json={
                "tickers": ["AAPL"],
                "selected_analysts": ["__not_real__"],
                "model_provider": "DeepSeek",
                "model_name": "deepseek-v4-pro",
            },
        )
        assert resp.status_code == 400
        assert "Unknown analysts" in resp.json()["detail"]

    def test_rejects_deepseek_without_deepseek_or_nvidia_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

        resp = client.post(
            "/api/hedgefund/analyze",
            json={
                "tickers": ["AAPL"],
                "selected_analysts": [_valid_analyst()],
                "model_provider": "DeepSeek",
                "model_name": "deepseek-v4-pro",
            },
        )
        assert resp.status_code == 400
        assert "DeepSeek API key missing" in resp.json()["detail"]

    def test_accepts_deepseek_with_nvidia_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("NVIDIA_API_KEY", "nv-test-key")

        fake_thread = Mock()
        fake_thread.start = Mock()

        with patch.object(server_module.threading, "Thread", return_value=fake_thread):
            resp = client.post(
                "/api/hedgefund/analyze",
                json={
                    "tickers": ["AAPL"],
                    "selected_analysts": [_valid_analyst()],
                    "model_provider": "DeepSeek",
                    "model_name": "deepseek-v4-pro",
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        fake_thread.start.assert_called_once()

    def test_rejects_openai_without_openai_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        resp = client.post(
            "/api/hedgefund/analyze",
            json={
                "tickers": ["AAPL"],
                "selected_analysts": [_valid_analyst()],
                "model_provider": "OpenAI",
                "model_name": "gpt-4o",
            },
        )
        assert resp.status_code == 400
        assert "OPENAI_API_KEY" in resp.json()["detail"]
