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


def _history_enabled() -> bool:
    """History syncs to a SEPARATE gist to isolate large analysis JSONL files
    from the hot-path pulse gist."""
    return bool(os.environ.get("GITHUB_TOKEN") and os.environ.get("HISTORY_GIST_ID"))


def _count_lines(text: str) -> int:
    """Count non-empty JSONL lines. Used to compare local vs remote freshness
    without being fooled by trailing whitespace differences."""
    return sum(1 for ln in text.splitlines() if ln.strip())


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
        # Only seed if local file is missing or has fewer lines than remote
        # (line-count beats byte-count: immune to trailing-whitespace round-trip
        # differences; see plan action #8).
        if dest.exists():
            try:
                local_lines = _count_lines(dest.read_text())
                if local_lines >= _count_lines(content):
                    continue
            except Exception:
                pass
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
        if dest.exists():
            try:
                local_lines = _count_lines(dest.read_text())
                if local_lines >= _count_lines(content):
                    continue
            except Exception:
                pass
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


# ── Analysis history sync (full_states_log JSON via secondary gist) ────
# Uses HISTORY_GIST_ID (separate from PULSE_GIST_ID) so large history
# payloads never contend with hot-path pulse writes. Each ticker gets a
# JSONL file where each line is {"filename": ..., "content": {...}}.
# Rolling cap = 100 entries per ticker (~9 MB at 90KB each, under gist's
# 10MB/file limit). Math: 100 entries × 4h cadence = 16.67 days of history
# per ticker. Calibration minimum is 30 decisions; at 6/day that's 5 days,
# so 100 is safely >6x the minimum. See plan action #9.
_HISTORY_ROLLING_CAP = 100


def _history_filename_for(ticker: str) -> str:
    return f"history_{ticker.replace('/', '_')}.jsonl"


def pull_history_all(logs_root: Path) -> dict:
    """Download every `history_*.jsonl` from the HISTORY gist and reconstruct
    individual `full_states_log_*.json` files under
    ``logs_root/<ticker>/TradingAgentsStrategy_logs/``.

    Idempotent: only writes files that don't already exist locally.
    """
    if not _history_enabled():
        return {"pulled": [], "skipped": True}
    try:
        import requests
    except ImportError:
        return {"pulled": [], "skipped": True}

    gist_id = os.environ["HISTORY_GIST_ID"]
    try:
        r = requests.get(f"{_GIST_API}/{gist_id}", headers=_headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"[GistSync] History pull failed: {e}")
        return {"pulled": [], "skipped": False, "error": str(e)}

    total_files = 0
    tickers_pulled = []
    for filename, file_meta in (data.get("files") or {}).items():
        if not filename.startswith("history_") or not filename.endswith(".jsonl"):
            continue
        ticker = filename[len("history_"):-len(".jsonl")]
        content = file_meta.get("content")
        if file_meta.get("truncated") and file_meta.get("raw_url"):
            try:
                rr = requests.get(file_meta["raw_url"], headers=_headers(), timeout=60)
                rr.raise_for_status()
                content = rr.text
            except Exception as e:
                logger.warning(f"[GistSync] History raw fetch failed for {filename}: {e}")
                continue
        if not content:
            continue

        logs_dir = logs_root / ticker / "TradingAgentsStrategy_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ticker_pulled = 0
        import json as _json
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
                fn = rec.get("filename")
                body = rec.get("content")
                if not fn or body is None:
                    continue
                dest = logs_dir / fn
                if dest.exists():
                    continue
                dest.write_text(_json.dumps(body, indent=2, default=str))
                ticker_pulled += 1
                total_files += 1
            except Exception as e:
                logger.warning(f"[GistSync] History line parse failed ({filename}): {e}")
                continue
        if ticker_pulled > 0:
            tickers_pulled.append({"ticker": ticker, "restored": ticker_pulled})

    logger.info(f"[GistSync] History pulled {total_files} file(s) across {len(tickers_pulled)} ticker(s)")
    return {"pulled": tickers_pulled, "total_files": total_files, "skipped": False}


def push_history(logs_root: Path, ticker: str) -> bool:
    """Collect all `full_states_log_*.json` files for a ticker, pack them as a
    JSONL blob (capped at last _HISTORY_ROLLING_CAP by mtime), and PATCH into
    the HISTORY gist.
    """
    if not _history_enabled():
        return False
    try:
        import requests
    except ImportError:
        return False

    logs_dir = logs_root / ticker / "TradingAgentsStrategy_logs"
    if not logs_dir.exists():
        return False

    # Collect all state logs, sort newest-first by mtime, cap at rolling limit
    json_files = sorted(
        logs_dir.glob("full_states_log_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:_HISTORY_ROLLING_CAP]
    if not json_files:
        return False

    import json as _json
    lines = []
    for path in reversed(json_files):  # chronological order in the file
        try:
            body = _json.loads(path.read_text())
            rec = {"filename": path.name, "content": body}
            lines.append(_json.dumps(rec, default=str))
        except Exception as e:
            logger.warning(f"[GistSync] History read {path.name} failed: {e}")
            continue

    if not lines:
        return False

    text = "\n".join(lines) + "\n"

    # Enforce 10MB gist-file limit defensively: drop oldest entries if we blew past
    max_bytes = 9_500_000
    while len(text.encode("utf-8")) > max_bytes and len(lines) > 1:
        lines.pop(0)  # drop oldest
        text = "\n".join(lines) + "\n"
        logger.warning(f"[GistSync] history_{ticker}.jsonl trimmed to {len(lines)} entries to fit gist size cap")

    gist_id = os.environ["HISTORY_GIST_ID"]
    filename = _history_filename_for(ticker)
    payload = {"files": {filename: {"content": text}}}
    try:
        r = requests.patch(
            f"{_GIST_API}/{gist_id}",
            headers=_headers(),
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"[GistSync] History push {filename} failed: {e}")
        return False


def is_history_enabled() -> bool:
    return _history_enabled()


# ── Generic state-dict sync (scheduler_state, model_config) ───────────
# Tiny JSON files kept in the PRIMARY gist so scheduler "enabled" flag and
# model choice survive Render free-tier restarts. Low-frequency writes
# (only on toggle / model change) = no rate-limit risk.

def push_state(filename: str, content: dict) -> bool:
    """PATCH a small JSON blob to the primary gist under ``filename``."""
    if not _enabled():
        return False
    try:
        import requests
    except ImportError:
        return False

    import json as _json
    try:
        text = _json.dumps(content, indent=2, default=str)
    except Exception as e:
        logger.warning(f"[GistSync] State serialize {filename} failed: {e}")
        return False

    gist_id = os.environ["PULSE_GIST_ID"]
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
        logger.warning(f"[GistSync] State push {filename} failed: {e}")
        return False


def pull_state(filename: str):
    """Fetch a single state file from the primary gist. Returns parsed dict
    or None if missing / disabled / error."""
    if not _enabled():
        return None
    try:
        import requests
    except ImportError:
        return None

    gist_id = os.environ["PULSE_GIST_ID"]
    try:
        r = requests.get(f"{_GIST_API}/{gist_id}", headers=_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"[GistSync] State pull failed: {e}")
        return None

    file_meta = (data.get("files") or {}).get(filename)
    if not file_meta:
        return None
    content = file_meta.get("content")
    if file_meta.get("truncated") and file_meta.get("raw_url"):
        try:
            rr = requests.get(file_meta["raw_url"], headers=_headers(), timeout=15)
            rr.raise_for_status()
            content = rr.text
        except Exception:
            return None
    if not content:
        return None
    import json as _json
    try:
        return _json.loads(content)
    except Exception:
        return None
