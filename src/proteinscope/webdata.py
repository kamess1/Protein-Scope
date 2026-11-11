"""Shared HTTP helper for the public data services (AlphaFold DB, RCSB CCD, ChEMBL, UniProt).

Responses are cached as JSON below the structure cache. A cached answer is reused for
`PROTEINSCOPE_CACHE_DAYS` days (30 by default), because ChEMBL gains potencies and drugs
and UniProt gains annotation; after that it is fetched again. If the service cannot be
reached, the expired copy is used rather than failing, so a report can still be rebuilt
offline once its context has been fetched.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any

import requests

from . import provenance
from .rcsb import default_cache_dir

_TIMEOUT = 30    # Seconds without any data before a request is abandoned.
_DAY = 86400.0
# A public API that answers 5xx or 429 is usually back a moment later; retry before giving
# up, so one hiccup does not drop a whole section from a report.
_RETRY_DELAYS = (1.0, 3.0, 8.0)
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_DEADLINE = 90   # Seconds for a whole JSON response, however slowly it trickles in.
_HEADERS = {"User-Agent": "ProteinScope/0.2", "Accept": "application/json"}


class DataSourceError(RuntimeError):
    """A public data service could not be reached or returned something unusable."""


def _cache_path(url: str, cache_dir: Path | None) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    return (cache_dir or default_cache_dir()) / "web" / f"{digest}.json"


def _fetch(url: str, deadline: float) -> requests.Response:
    """GET with a bound on the total time, not only on the gap between packets.

    `requests` applies its timeout per socket read, so a server that sends a byte every
    few seconds would hold a request open indefinitely; this reads in chunks and gives up
    once the deadline has passed.
    """
    start = time.monotonic()
    try:
        response = requests.get(url, headers=_HEADERS, timeout=(10, _TIMEOUT), stream=True)
        chunks = []
        for chunk in response.iter_content(1 << 16):
            chunks.append(chunk)
            if time.monotonic() - start > deadline:
                response.close()
                raise DataSourceError(f"{urlsplit(url).netloc} took longer than {deadline:.0f} s to answer")
    except requests.RequestException as exc:
        raise DataSourceError(f"Could not reach {url}: {exc}") from exc
    response._content = b"".join(chunks)  # Make .json() and .content work as usual.
    return response


def cache_max_age() -> float:
    """Seconds a cached answer stays fresh; PROTEINSCOPE_CACHE_DAYS overrides the 30 days."""
    try:
        days = float(os.environ.get("PROTEINSCOPE_CACHE_DAYS", "30"))
    except ValueError:
        days = 30.0
    return max(days, 0.0) * _DAY


def _read_cache(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None


def get_json(url: str, cache_dir: Path | None = None, use_cache: bool = True) -> Any:
    path = _cache_path(url, cache_dir)
    stale = None
    stamp = 0.0
    if use_cache and path.is_file():
        stamp = path.stat().st_mtime
        fresh = time.time() - stamp < cache_max_age()
        cached = _read_cache(path)
        if cached is not None and fresh:
            provenance.record(url, provenance.FROM_CACHE, stamp)
            return cached
        stale = cached  # Expired: fetch again, but keep it in case the service is down.
    try:
        data = _fetch_json(url, path if use_cache else None)
    except DataSourceError:
        if stale is not None:
            provenance.record(url, provenance.STALE_CACHE, stamp)
            return stale
        raise
    provenance.record(url, provenance.FROM_NETWORK)
    return data


def _fetch_retrying(url: str, deadline: float) -> requests.Response:
    """_fetch, retried after a short pause on connection errors and transient statuses."""
    for delay in (*_RETRY_DELAYS, None):
        try:
            response = _fetch(url, deadline)
        except DataSourceError:
            if delay is None:
                raise
        else:
            if response.status_code not in _RETRY_STATUS or delay is None:
                return response
        time.sleep(delay)
    raise AssertionError("unreachable")  # The loop always returns or raises.


def _fetch_json(url: str, cache_path: Path | None) -> Any:
    response = _fetch_retrying(url, _DEADLINE)
    if response.status_code == 404:
        raise DataSourceError(f"Not found: {url}")
    if not response.ok:
        raise DataSourceError(f"{url} returned HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError as exc:
        raise DataSourceError(f"{url} did not return JSON") from exc
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def download(url: str, target: Path) -> Path:
    if target.is_file() and target.stat().st_size > 0:
        provenance.record(url, provenance.FROM_CACHE, target.stat().st_mtime)
        return target
    response = _fetch(url, _DEADLINE * 4)  # Structures can be tens of megabytes.
    if not response.ok:
        raise DataSourceError(f"Download of {url} returned HTTP {response.status_code}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(target)
    provenance.record(url, provenance.FROM_NETWORK)
    return target
