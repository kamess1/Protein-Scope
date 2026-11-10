"""Shared HTTP helper for the public data services (AlphaFold DB, RCSB CCD, ChEMBL).

Responses are cached as JSON below the structure cache, so a report can be rebuilt offline
once its context has been fetched.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any

import requests

from .rcsb import default_cache_dir

_TIMEOUT = 30    # Seconds without any data before a request is abandoned.
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


def get_json(url: str, cache_dir: Path | None = None, use_cache: bool = True) -> Any:
    path = _cache_path(url, cache_dir)
    if use_cache and path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
    response = _fetch(url, _DEADLINE)
    if response.status_code == 404:
        raise DataSourceError(f"Not found: {url}")
    if not response.ok:
        raise DataSourceError(f"{url} returned HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError as exc:
        raise DataSourceError(f"{url} did not return JSON") from exc
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    return data


def download(url: str, target: Path) -> Path:
    if target.is_file() and target.stat().st_size > 0:
        return target
    response = _fetch(url, _DEADLINE * 4)  # Structures can be tens of megabytes.
    if not response.ok:
        raise DataSourceError(f"Download of {url} returned HTTP {response.status_code}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(target)
    return target
