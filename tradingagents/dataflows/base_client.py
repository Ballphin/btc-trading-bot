"""Shared base for all data source clients — caching, retries, rate limiting."""

import time
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_inflight_lock = threading.Lock()
_inflight: dict[str, threading.Event] = {}
_inflight_results: dict[str, dict] = {}


class BaseDataClient:
    """Base class providing disk caching, exponential backoff retries, and rate limit handling."""

    MAX_RETRIES = 2
    BACKOFF_BASE = 1  # seconds (faster fail on flaky upstreams)

    def __init__(self, cache_subdir: str, cache_ttl: int = 3600):
        self.cache_dir = Path(f"tradingagents/dataflows/data_cache/{cache_subdir}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl
        self.session = requests.Session()

    def _cache_key(self, prefix: str, params: dict) -> Path:
        raw = f"{prefix}:{json.dumps(params, sort_keys=True)}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, path: Path) -> Optional[dict]:
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if age < self.cache_ttl:
                try:
                    return json.loads(path.read_text())
                except (json.JSONDecodeError, OSError):
                    return None
        return None

    def _write_cache(self, path: Path, data):
        try:
            path.write_text(json.dumps(data, default=str))
        except (OSError, TypeError) as e:
            logger.warning(f"Cache write failed: {e}")

    def _request(self, url: str, params: dict = None, cache_prefix: str = None) -> dict:
        cache_path = self._cache_key(cache_prefix or url, params or {})
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        dedup_key = str(cache_path)
        with _inflight_lock:
            if dedup_key in _inflight:
                event = _inflight[dedup_key]
                waiting = True
            else:
                event = threading.Event()
                _inflight[dedup_key] = event
                waiting = False

        if waiting:
            event.wait(timeout=120)
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached
            if dedup_key in _inflight_results:
                return _inflight_results[dedup_key]

        try:
            result = self._do_request(url, params, cache_path)
            with _inflight_lock:
                _inflight_results[dedup_key] = result
            return result
        finally:
            with _inflight_lock:
                _inflight.pop(dedup_key, None)
                _inflight_results.pop(dedup_key, None)
            event.set()

    def _do_request(self, url: str, params: dict, cache_path: Path) -> dict:
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = self.BACKOFF_BASE ** (attempt + 1)
                    logger.warning(f"Rate limited by {url}, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                self._write_cache(cache_path, data)
                return data
            except requests.RequestException as e:
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(f"Request to {url} failed after {self.MAX_RETRIES} retries: {e}")
                    raise
                time.sleep(self.BACKOFF_BASE ** (attempt + 1))
        raise RuntimeError(f"Request to {url} failed after {self.MAX_RETRIES} retries")
