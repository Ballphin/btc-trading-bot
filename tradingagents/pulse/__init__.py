"""Quant Pulse v3 — TSMOM-primary, technical-only advisory signal engine.

Sub-modules:
    stats        — Sharpe, N_eff, deflated Sharpe, PBO helpers.
    config       — YAML loader + hot-reload + hash.
    tsmom        — Time-series momentum (primary alpha layer).
    regime       — GARCH-residual-style regime detector.
    fills        — 4 fill models + square-root market impact.
    alerts       — Discord / SMTP / webhook backends with dedup.
    liquidations — Hyperliquid liquidation cluster detector.
"""
