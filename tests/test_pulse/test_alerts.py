"""Tests for tradingagents/pulse/alerts.py."""

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.pulse.alerts import (
    AlertMessage,
    _reset_cooldowns_for_tests,
    dispatch_alert_if_eligible,
)


@pytest.fixture(autouse=True)
def _clear_cooldowns():
    _reset_cooldowns_for_tests()
    yield
    _reset_cooldowns_for_tests()


@pytest.fixture
def _cfg_with_alerts(monkeypatch):
    """Force a minimal YAML config with all backends disabled by default."""
    from tradingagents.pulse import config as cfg_mod

    class FakeCfg:
        engine_version = 3
        def __init__(self, d):
            self._d = d
        def get(self, *keys, default=None):
            node = self._d
            for k in keys:
                if not isinstance(node, dict) or k not in node:
                    return default
                node = node[k]
            return node
        def hash_short(self):
            return "abcd1234"

    def _get(alerts_cfg):
        return FakeCfg({"alerts": alerts_cfg})

    return _get


# ── AlertMessage ────────────────────────────────────────────────────

class TestAlertMessage:
    def test_from_pulse_populates_fields(self):
        pulse = {
            "signal": "BUY", "confidence": 0.72, "price": 75000,
            "normalized_score": 0.6, "tsmom_direction": 1,
            "regime_mode": "trend", "override_reason": None,
            "timeframe_bias": "1h", "stop_loss": 74000, "take_profit": 77000,
            "hold_minutes": 60, "engine_version": 3, "config_hash": "abc",
            "reasoning": "Strong confluence",
        }
        m = AlertMessage.from_pulse(pulse, "BTC-USD")
        assert m.ticker == "BTC-USD"
        assert m.signal == "BUY"
        assert m.confidence == 0.72

    def test_discord_embed_structure(self):
        pulse = {"signal": "SHORT", "confidence": 0.8, "price": 3000,
                 "normalized_score": -0.5, "tsmom_direction": -1,
                 "regime_mode": "trend", "reasoning": "x", "engine_version": 3}
        m = AlertMessage.from_pulse(pulse, "ETH-USD")
        embed = m.discord_embed()
        assert "title" in embed and "ETH-USD" in embed["title"]
        assert "fields" in embed
        assert embed["color"] == 0xEF4444   # red for SHORT

    def test_plain_text_formatting(self):
        pulse = {"signal": "BUY", "confidence": 0.55, "price": 100,
                 "normalized_score": 0.3, "tsmom_direction": 1,
                 "regime_mode": "trend", "timeframe_bias": "15m",
                 "reasoning": "x"}
        m = AlertMessage.from_pulse(pulse, "SOL-USD")
        text = m.plain_text()
        assert "SOL-USD" in text and "BUY" in text


# ── dispatch_alert_if_eligible ─────────────────────────────────────

class TestDispatch:
    def test_neutral_never_alerts(self, _cfg_with_alerts):
        cfg = _cfg_with_alerts({
            "min_confidence": 0.0,
            "cooldown_minutes_floor": 0,
            "backends": {"discord": {"enabled": True, "webhook_env": "TEST_URL"}},
        })
        pulse = {"signal": "NEUTRAL", "confidence": 1.0}
        with patch("tradingagents.pulse.alerts.get_config", return_value=cfg):
            result = dispatch_alert_if_eligible(pulse, "BTC-USD")
        assert result == {}

    def test_below_min_confidence_skipped(self, _cfg_with_alerts):
        cfg = _cfg_with_alerts({
            "min_confidence": 0.8,
            "cooldown_minutes_floor": 0,
            "backends": {"discord": {"enabled": True, "webhook_env": "TEST_URL"}},
        })
        pulse = {"signal": "BUY", "confidence": 0.5}
        with patch("tradingagents.pulse.alerts.get_config", return_value=cfg):
            result = dispatch_alert_if_eligible(pulse, "BTC-USD")
        assert result == {}

    def test_discord_disabled_no_dispatch(self, _cfg_with_alerts):
        cfg = _cfg_with_alerts({
            "min_confidence": 0.0, "cooldown_minutes_floor": 0,
            "backends": {"discord": {"enabled": False}},
        })
        pulse = {"signal": "BUY", "confidence": 0.9, "reasoning": "x"}
        with patch("tradingagents.pulse.alerts.get_config", return_value=cfg):
            result = dispatch_alert_if_eligible(pulse, "BTC-USD")
        assert result == {}

    def test_discord_enabled_but_env_missing(self, _cfg_with_alerts, monkeypatch):
        cfg = _cfg_with_alerts({
            "min_confidence": 0.0, "cooldown_minutes_floor": 0,
            "backends": {"discord": {"enabled": True, "webhook_env": "NOT_SET_VAR"}},
        })
        monkeypatch.delenv("NOT_SET_VAR", raising=False)
        pulse = {"signal": "BUY", "confidence": 0.9}
        with patch("tradingagents.pulse.alerts.get_config", return_value=cfg):
            result = dispatch_alert_if_eligible(pulse, "BTC-USD")
        assert result == {}

    def test_discord_success_calls_requests(self, _cfg_with_alerts, monkeypatch):
        cfg = _cfg_with_alerts({
            "min_confidence": 0.0, "cooldown_minutes_floor": 0,
            "backends": {"discord": {"enabled": True, "webhook_env": "TEST_URL"}},
        })
        monkeypatch.setenv("TEST_URL", "https://discord.test/webhook/abc")
        pulse = {"signal": "BUY", "confidence": 0.9, "reasoning": "strong confluence"}

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_requests = MagicMock()
        mock_requests.post.return_value = mock_resp
        with patch("tradingagents.pulse.alerts.get_config", return_value=cfg), \
             patch("tradingagents.pulse.alerts.requests", mock_requests):
            result = dispatch_alert_if_eligible(pulse, "BTC-USD")

        assert result == {"discord": True}
        mock_requests.post.assert_called_once()

    def test_cooldown_prevents_duplicate(self, _cfg_with_alerts, monkeypatch):
        cfg = _cfg_with_alerts({
            "min_confidence": 0.0, "cooldown_minutes_floor": 10,
            "backends": {"discord": {"enabled": True, "webhook_env": "TEST_URL"}},
        })
        monkeypatch.setenv("TEST_URL", "https://discord.test/webhook/abc")
        pulse = {"signal": "BUY", "confidence": 0.9, "reasoning": "x"}

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_requests = MagicMock()
        mock_requests.post.return_value = mock_resp
        with patch("tradingagents.pulse.alerts.get_config", return_value=cfg), \
             patch("tradingagents.pulse.alerts.requests", mock_requests):
            r1 = dispatch_alert_if_eligible(pulse, "BTC-USD")
            r2 = dispatch_alert_if_eligible(pulse, "BTC-USD")   # within cooldown

        assert r1 == {"discord": True}
        assert r2 == {}   # cooldown blocked

    def test_webhook_backend(self, _cfg_with_alerts, monkeypatch):
        cfg = _cfg_with_alerts({
            "min_confidence": 0.0, "cooldown_minutes_floor": 0,
            "backends": {"webhook": {"enabled": True, "url_env": "WH_URL"}},
        })
        monkeypatch.setenv("WH_URL", "https://my.webhook/")
        pulse = {"signal": "BUY", "confidence": 0.9}

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_requests = MagicMock()
        mock_requests.post.return_value = mock_resp
        with patch("tradingagents.pulse.alerts.get_config", return_value=cfg), \
             patch("tradingagents.pulse.alerts.requests", mock_requests):
            result = dispatch_alert_if_eligible(pulse, "BTC-USD")

        assert result == {"webhook": True}
        call_args = mock_requests.post.call_args
        assert call_args[0][0] == "https://my.webhook/"
        payload = call_args[1]["json"]
        assert payload["ticker"] == "BTC-USD"
        assert payload["signal"] == "BUY"

    def test_backend_failure_does_not_raise(self, _cfg_with_alerts, monkeypatch):
        cfg = _cfg_with_alerts({
            "min_confidence": 0.0, "cooldown_minutes_floor": 0,
            "backends": {"discord": {"enabled": True, "webhook_env": "TEST_URL"}},
        })
        monkeypatch.setenv("TEST_URL", "https://discord.test/webhook/abc")
        pulse = {"signal": "BUY", "confidence": 0.9}

        mock_requests = MagicMock()
        mock_requests.post.side_effect = Exception("network down")
        with patch("tradingagents.pulse.alerts.get_config", return_value=cfg), \
             patch("tradingagents.pulse.alerts.requests", mock_requests):
            # Must NOT raise
            result = dispatch_alert_if_eligible(pulse, "BTC-USD")

        assert result == {"discord": False}
