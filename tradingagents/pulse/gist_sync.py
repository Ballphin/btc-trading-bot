"""Free-tier persistence for pulse.jsonl via GitHub Gist.

Render's free tier wipes the filesystem on every restart. This module
solves it with zero infra: one GitHub Gist per ticker, updated after every
pulse write and pulled on startup.

Design:
    * Two env vars: GITHUB_TOKEN (a PAT with `gist` scope) and
      PULSE_GIST_ID (the ID of a single gist you created manually).
    * The gist contains one file per ticker: `pulse_BTC-USD.jsonl`, etc.
    * On startup: download each file and seed the local `eval_results/pulse/<ticker>/pulse.jsonl`.
    * After every pulse append: upload the local file back to the same gist
      filename. Debounced to one PATCH per ticker per call.

Rate limits: GitHub gives 5000 authenticated REST calls/hour. At 5-min
pulse cadence across 3 tickers that's 3 × 12 × 24 = 864 calls/day. Well
under the limit.

If either env var is missing, this module no-ops — everything keeps working
on a paid plan with a persistent disk.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_GIST_API = "https://api.github.com/gists"


def _enabled() -> bool:
    return bool(os.environ.get("GITHUB_TOKEN") and os.environ.get("PULSE_GIST_ID"))


def _headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN", "")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _filename_for(ticker: str) -> str:
    # Gist filenames can't contain '/'. Use a safe convention.
    return f"pulse_{ticker.replace('/', '_')}.jsonl"


def pull_all(pulse_dir: Path) -> dict:
    """Download every file from the configured gist into pulse_dir/<ticker>/pulse.jsonl.

    Returns {"pulled": [ticker, ...], "skipped": bool} — skipped=True if
    persistence is disabled (missing env vars).
    """
    if not _enabled():
        return {"pulled": [], "skipped": True}

    try:
        import requests
    except ImportError:
        logger.warning("[GistSync] `requests` not installed; skipping pull")
        return {"pulled": [], "skipped": True}

    gist_id = os.environ["PULSE_GIST_ID"]
    try:
        r = requests.get(f"{_GIST_API}/{gist_id}", headers=_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"[GistSync] Pull failed: {e}")
        return {"pulled": [], "skipped": False, "error": str(e)}

    pulled = []
    for filename, file_meta in (data.get("files") or {}).items():
        if not filename.startswith("pulse_") or not filename.endswith(".jsonl"):
            continue
        ticker = filename[len("pulse_"):-len(".jsonl")]
        content = file_meta.get("content")
        # Truncated files have .truncated=True; fetch via raw_url
        if file_meta.get("truncated") and file_meta.get("raw_url"):
            try:
                rr = requests.get(file_meta["raw_url"], headers=_headers(), timeout=30)
                rr.raise_for_status()
                content = rr.text
            except Exception as e:
                logger.warning(f"[GistSync] Raw fetch failed for {filename}: {e}")
                continue
        if content is None:
            continue

        dest = pulse_dir / ticker / "pulse.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Only seed if local file is missing or smaller (don't overwrite fresher local data)
        if dest.exists() and dest.stat().st_size >= len(content.encode("utf-8")):
            continue
        dest.write_text(content)
        pulled.append(ticker)

    logger.info(f"[GistSync] Pulled {len(pulled)} ticker(s) from gist: {pulled}")
    return {"pulled": pulled, "skipped": False}


def push_ticker(pulse_dir: Path, ticker: str) -> bool:
    """Upload the full pulse.jsonl for a ticker to the gist.

    Returns True on success, False on failure (including env not configured).
    Safe to call on every write; latency 100-300ms typical.
    """
    if not _enabled():
        return False

    try:
        import requests
    except ImportError:
        return False

    path = pulse_dir / ticker / "pulse.jsonl"
    if not path.exists():
        return False

    # Cap uploaded size: if the file exceeds ~900KB we rotate (keep last 5000 lines)
    try:
        text = path.read_text()
        max_bytes = 900_000
        if len(text.encode("utf-8")) > max_bytes:
            lines = text.splitlines()[-5000:]
            text = "\n".join(lines) + "\n"
    except Exception as e:
        logger.warning(f"[GistSync] Read {path} failed: {e}")
        return False

    gist_id = os.environ["PULSE_GIST_ID"]
    filename = _filename_for(ticker)
    payload = {"files": {filename: {"content": text}}}
    try:
        r = requests.patch(
            f"{_GIST_API}/{gist_id}",
            headers=_headers(),
            json=payload,
            timeout=15,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"[GistSync] Push {filename} failed: {e}")
        return False


def is_enabled() -> bool:
    """Public helper for startup logs."""
    return _enabled()


# ── Shadow decisions sync (analysis-based backtests) ──────────────────
# Mirrors the pulse pull/push pair but uses a `shadow_<TICKER>.jsonl`
# filename prefix inside the same gist so pulse and shadow don't collide.

def _shadow_filename_for(ticker: str) -> str:
    return f"shadow_{ticker.replace('/', '_')}.jsonl"


def pull_shadow_all(shadow_root: Path) -> dict:
    """Download every `shadow_*.jsonl` from the gist into
    ``shadow_root/<ticker>/decisions.jsonl``.
    """
    if not _enabled():
        return {"pulled": [], "skipped": True}
    try:
        import requests
    except ImportError:
        return {"pulled": [], "skipped": True}

    gist_id = os.environ["PULSE_GIST_ID"]
    try:
        r = requests.get(f"{_GIST_API}/{gist_id}", headers=_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"[GistSync] Shadow pull failed: {e}")
        return {"pulled": [], "skipped": False, "error": str(e)}

    pulled = []
    for filename, file_meta in (data.get("files") or {}).items():
        if not filename.startswith("shadow_") or not filename.endswith(".jsonl"):
            continue
        ticker = filename[len("shadow_"):-len(".jsonl")]
        content = file_meta.get("content")
        if file_meta.get("truncated") and file_meta.get("raw_url"):
            try:
                rr = requests.get(file_meta["raw_url"], headers=_headers(), timeout=30)
                rr.raise_for_status()
                content = rr.text
            except Exception as e:
                logger.warning(f"[GistSync] Shadow raw fetch failed for {filename}: {e}")
                continue
        if content is None:
            continue

        dest = shadow_root / ticker / "decisions.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size >= len(content.encode("utf-8")):
            continue
        dest.write_text(content)
        pulled.append(ticker)

    logger.info(f"[GistSync] Shadow pulled {len(pulled)} ticker(s): {pulled}")
    return {"pulled": pulled, "skipped": False}


def push_shadow(shadow_root: Path, ticker: str) -> bool:
    """Upload ``shadow_root/<ticker>/decisions.jsonl`` to the gist."""
    if not _enabled():
        return False
    try:
        import requests
    except ImportError:
        return False

    path = shadow_root / ticker / "decisions.jsonl"
    if not path.exists():
        return False

    try:
        text = path.read_text()
        max_bytes = 900_000
        if len(text.encode("utf-8")) > max_bytes:
            lines = text.splitlines()[-5000:]
            text = "\n".join(lines) + "\n"
    except Exception as e:
        logger.warning(f"[GistSync] Shadow read {path} failed: {e}")
        return False

    gist_id = os.environ["PULSE_GIST_ID"]
    filename = _shadow_filename_for(ticker)
    payload = {"files": {filename: {"content": text}}}
    try:
        r = requests.patch(
            f"{_GIST_API}/{gist_id}",
            headers=_headers(),
            json=payload,
            timeout=15,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"[GistSync] Shadow push {filename} failed: {e}")
        return False
