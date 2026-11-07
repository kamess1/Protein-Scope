"""Shared HTTP helper for the public data services (AlphaFold DB, RCSB CCD, ChEMBL).

Responses are cached as JSON below the structure cache, so a report can be rebuilt offline
once its context has been fetched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import requests

from .rcsb import default_cache_dir

_TIMEOUT = 30
_HEADERS = {"User-Agent": "ProteinScope/0.2", "Accept": "application/json"}


class DataSourceError(RuntimeError):
    """A public data service could not be reached or returned something unusable."""


def _cache_path(url: str, cache_dir: Path | None) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    return (cache_dir or default_cache_dir()) / "web" / f"{digest}.json"


def get_json(url: str, cache_dir: Path | None = None, use_cache: bool = True) -> Any:
    path = _cache_path(url, cache_dir)
    if use_cache and path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
    try:
        response = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise DataSourceError(f"Could not reach {url}: {exc}") from exc
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
    try:
        response = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT * 4)
    except requests.RequestException as exc:
        raise DataSourceError(f"Download of {url} failed: {exc}") from exc
    if not response.ok:
        raise DataSourceError(f"Download of {url} returned HTTP {response.status_code}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(target)
    return target
