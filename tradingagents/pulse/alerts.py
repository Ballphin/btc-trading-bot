"""Multi-backend alerting for high-conviction pulse signals.

Supports:
    - Discord (webhook URL via env)
    - Generic webhook (JSON POST, any URL via env)
    - Email (SMTP via env; lazy import)

Usage:
    from tradingagents.pulse.alerts import dispatch_alert_if_eligible
    await dispatch_alert_if_eligible(pulse_entry, ticker="BTC-USD")

All backends are opt-in via ``alerts.backends.{backend}.enabled`` in the YAML.
Cooldowns prevent spam: one alert per (ticker, signal) per
``alerts.cooldown_minutes_floor`` minutes.

Failures are logged but never raised — alerting is best-effort.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None

from tradingagents.pulse.config import get_config

logger = logging.getLogger(__name__)


# ── Cooldown state (in-memory) ────────────────────────────────────────

_cooldown_lock = Lock()
_last_alert_ts: Dict[str, float] = {}   # key = f"{ticker}:{signal}"


def _cooldown_ok(ticker: str, signal: str, floor_min: int) -> bool:
    key = f"{ticker}:{signal}"
    now = time.time()
    with _cooldown_lock:
        last = _last_alert_ts.get(key, 0)
        if now - last < floor_min * 60:
            return False
        _last_alert_ts[key] = now
        return True


# ── Message formatting ─────────────────────────────────────────────────

@dataclass
class AlertMessage:
    ticker: str
    signal: str
    confidence: float
    price: Optional[float]
    normalized_score: Optional[float]
    tsmom_direction: Optional[int]
    regime_mode: Optional[str]
    override_reason: Optional[str]
    timeframe_bias: Optional[str]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    hold_minutes: Optional[int]
    engine_version: Optional[int]
    config_hash: Optional[str]
    reasoning: str

    @classmethod
    def from_pulse(cls, pulse: dict, ticker: str) -> "AlertMessage":
        return cls(
            ticker=ticker,
            signal=str(pulse.get("signal", "?")),
            confidence=float(pulse.get("confidence", 0.0)),
            price=pulse.get("price"),
            normalized_score=pulse.get("normalized_score"),
            tsmom_direction=pulse.get("tsmom_direction"),
            regime_mode=pulse.get("regime_mode"),
            override_reason=pulse.get("override_reason"),
            timeframe_bias=pulse.get("timeframe_bias"),
            stop_loss=pulse.get("stop_loss"),
            take_profit=pulse.get("take_profit"),
            hold_minutes=pulse.get("hold_minutes"),
            engine_version=pulse.get("engine_version"),
            config_hash=pulse.get("config_hash"),
            reasoning=str(pulse.get("reasoning", "")),
        )

    def _tsmom_label(self) -> str:
        d = self.tsmom_direction
        if d is None:
            return "n/a"
        return "↑" if d > 0 else "↓" if d < 0 else "flat"

    def _signal_emoji(self) -> str:
        return {"BUY": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}.get(self.signal, "❔")

    def discord_embed(self) -> dict:
        color = {"BUY": 0x22C55E, "SHORT": 0xEF4444, "NEUTRAL": 0x9CA3AF}.get(self.signal, 0x3B82F6)
        fields = [
            {"name": "Confidence", "value": f"{self.confidence:.2f}", "inline": True},
            {"name": "Score", "value": f"{self.normalized_score:+.3f}" if self.normalized_score is not None else "n/a", "inline": True},
            {"name": "Price", "value": f"${self.price:,.2f}" if self.price else "n/a", "inline": True},
            {"name": "TSMOM", "value": self._tsmom_label(), "inline": True},
            {"name": "Regime", "value": self.regime_mode or "n/a", "inline": True},
            {"name": "Bias", "value": self.timeframe_bias or "n/a", "inline": True},
        ]
        if self.stop_loss is not None and self.take_profit is not None:
            fields.append({"name": "SL / TP",
                           "value": f"${self.stop_loss:,.2f} / ${self.take_profit:,.2f}",
                           "inline": False})
        if self.override_reason:
            fields.append({"name": "Override", "value": str(self.override_reason), "inline": False})
        if self.hold_minutes is not None:
            fields.append({"name": "Hold", "value": f"{self.hold_minutes} min", "inline": True})
        if self.engine_version is not None:
            fields.append({
                "name": "Engine",
                "value": f"v{self.engine_version} · {self.config_hash or ''}",
                "inline": True,
            })

        return {
            "title": f"{self._signal_emoji()} {self.ticker} — {self.signal}",
            "description": self.reasoning[:1024] if self.reasoning else "",
            "color": color,
            "fields": fields,
        }

    def plain_text(self) -> str:
        lines = [
            f"{self._signal_emoji()} {self.ticker} — {self.signal} (conf={self.confidence:.2f})",
            f"  price=${self.price:,.2f}" if self.price else "",
            f"  score={self.normalized_score:+.3f}  tsmom={self._tsmom_label()}  regime={self.regime_mode}",
            f"  bias={self.timeframe_bias}  hold={self.hold_minutes}m" if self.timeframe_bias else "",
        ]
        if self.stop_loss is not None and self.take_profit is not None:
            lines.append(f"  SL=${self.stop_loss:,.2f}  TP=${self.take_profit:,.2f}")
        if self.override_reason:
            lines.append(f"  override={self.override_reason}")
        return "\n".join(l for l in lines if l)


# ── Backends ───────────────────────────────────────────────────────────

def _send_discord(message: AlertMessage, webhook_url: str, timeout: float = 5.0) -> bool:
    if requests is None:
        logger.warning("[alerts] requests not installed; skipping discord")
        return False
    try:
        payload = {"embeds": [message.discord_embed()]}
        resp = requests.post(webhook_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"[alerts.discord] failed: {e}")
        return False


def _send_webhook(message: AlertMessage, url: str, timeout: float = 5.0) -> bool:
    if requests is None:
        return False
    try:
        payload = {
            "ticker": message.ticker,
            "signal": message.signal,
            "confidence": message.confidence,
            "price": message.price,
            "normalized_score": message.normalized_score,
            "tsmom_direction": message.tsmom_direction,
            "regime_mode": message.regime_mode,
            "override_reason": message.override_reason,
            "stop_loss": message.stop_loss,
            "take_profit": message.take_profit,
            "hold_minutes": message.hold_minutes,
            "engine_version": message.engine_version,
            "config_hash": message.config_hash,
            "reasoning": message.reasoning,
        }
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"[alerts.webhook] failed: {e}")
        return False


def _send_email(message: AlertMessage, prefix: str, to_addr: str) -> bool:
    import smtplib
    from email.mime.text import MIMEText

    host = os.getenv(f"{prefix}_HOST")
    port = int(os.getenv(f"{prefix}_PORT", "587"))
    user = os.getenv(f"{prefix}_USER")
    pw = os.getenv(f"{prefix}_PASS")
    from_addr = os.getenv(f"{prefix}_FROM", user or "pulse@localhost")
    if not all([host, user, pw, to_addr]):
        logger.warning("[alerts.email] missing env vars; skipping")
        return False
    try:
        msg = MIMEText(message.plain_text())
        msg["Subject"] = f"[Pulse] {message.ticker} {message.signal} conf={message.confidence:.2f}"
        msg["From"] = from_addr
        msg["To"] = to_addr
        with smtplib.SMTP(host, port, timeout=10) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
        return True
    except Exception as e:
        logger.warning(f"[alerts.email] failed: {e}")
        return False


# ── Top-level dispatcher ──────────────────────────────────────────────

def dispatch_alert_if_eligible(pulse: dict, ticker: str) -> Dict[str, bool]:
    """Send alerts across all enabled backends if the pulse is eligible.

    Eligibility:
        - pulse["signal"] ∈ {BUY, SHORT}  (NEUTRAL never alerts)
        - pulse["confidence"] ≥ alerts.min_confidence
        - cooldown since last (ticker, signal) alert ≥ floor

    Returns:
        Dict mapping backend name → success bool. Empty dict if ineligible.
    """
    cfg = get_config()
    alerts_cfg = cfg.get("alerts", default={}) or {}

    signal = pulse.get("signal")
    if signal not in ("BUY", "SHORT"):
        return {}

    min_conf = float(alerts_cfg.get("min_confidence", 0.5))
    if float(pulse.get("confidence", 0)) < min_conf:
        return {}

    floor_min = int(alerts_cfg.get("cooldown_minutes_floor", 10))
    if not _cooldown_ok(ticker, signal, floor_min):
        logger.info(f"[alerts] cooldown active for {ticker}:{signal}, skipping")
        return {}

    message = AlertMessage.from_pulse(pulse, ticker)
    backends_cfg = alerts_cfg.get("backends", {}) or {}
    results: Dict[str, bool] = {}

    # Discord
    disc = backends_cfg.get("discord", {}) or {}
    if disc.get("enabled"):
        url = os.getenv(disc.get("webhook_env", "PULSE_DISCORD_WEBHOOK_URL"))
        if url:
            results["discord"] = _send_discord(message, url)
        else:
            logger.info("[alerts.discord] enabled but webhook env unset")

    # Generic webhook
    wh = backends_cfg.get("webhook", {}) or {}
    if wh.get("enabled"):
        url = os.getenv(wh.get("url_env", "PULSE_WEBHOOK_URL"))
        if url:
            results["webhook"] = _send_webhook(message, url)
        else:
            logger.info("[alerts.webhook] enabled but url env unset")

    # Email
    em = backends_cfg.get("email", {}) or {}
    if em.get("enabled"):
        prefix = em.get("smtp_env_prefix", "PULSE_SMTP")
        to_env = em.get("to_env", "PULSE_EMAIL_TO")
        to_addr = os.getenv(to_env)
        if to_addr:
            results["email"] = _send_email(message, prefix, to_addr)
        else:
            logger.info("[alerts.email] enabled but recipient env unset")

    if results:
        logger.info(
            f"[alerts] {ticker} {signal} conf={message.confidence:.2f} "
            f"dispatched: {results}"
        )
    return results


# ── Testing / reset helpers ───────────────────────────────────────────

def _reset_cooldowns_for_tests() -> None:
    """Testing helper — resets the in-memory cooldown table."""
    with _cooldown_lock:
        _last_alert_ts.clear()
