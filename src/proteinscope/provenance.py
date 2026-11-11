"""Where every report comes from: software, model, parameters, input and data retrieval.

`run_analysis` collects this while it runs and stores it on the result, and every output
(JSON, HTML, Markdown, PDF) carries it, so a report opened months later still says which
version computed it, from which input file and with data fetched when.

Nothing here names a local path: a local input is identified by its checksum and size only.

Network access is recorded through `record`, which the HTTP helpers call. The recorder is
a context variable, so analyses in different threads (the web server's worker, say) keep
separate records.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import __version__
from .sources import BY_NAME, source_for_url

LIBRARIES = ("numpy", "scipy", "biopython", "rdkit", "matplotlib", "reportlab")
FROM_NETWORK, FROM_CACHE, STALE_CACHE = "network", "cache", "expired cache"


@dataclass
class _SourceUse:
    requests: int = 0
    from_cache: int = 0
    stale_fallback: int = 0
    oldest_retrieval: float | None = None
    newest_retrieval: float | None = None

    def add(self, origin: str, retrieved: float) -> None:
        self.requests += 1
        self.from_cache += origin != FROM_NETWORK
        self.stale_fallback += origin == STALE_CACHE
        self.oldest_retrieval = min(filter(None, [self.oldest_retrieval, retrieved]))
        self.newest_retrieval = max(filter(None, [self.newest_retrieval, retrieved]))


@dataclass
class _Recorder:
    uses: dict[str, _SourceUse] = field(default_factory=dict)


_current: ContextVar[_Recorder | None] = ContextVar("proteinscope_provenance", default=None)


def record(url: str, origin: str, retrieved: float | None = None) -> None:
    """Note one read from a public service. `retrieved` is when the data left the service."""
    recorder = _current.get()
    source = source_for_url(url)
    if recorder is None or source is None:
        return
    when = retrieved if retrieved is not None else datetime.now(timezone.utc).timestamp()
    recorder.uses.setdefault(source.name, _SourceUse()).add(origin, when)


@contextmanager
def collecting() -> Iterator[_Recorder]:
    recorder = _Recorder()
    token = _current.set(recorder)
    try:
        yield recorder
    finally:
        _current.reset(token)


def _utc(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def model_summary() -> dict[str, Any]:
    from .druggability import MODEL_FILE, load_model

    model = load_model()
    if model is None:
        return {"kind": "hand-weighted heuristic"}
    from importlib.resources import files

    raw = files("proteinscope.data").joinpath(MODEL_FILE).read_bytes()
    return {
        "kind": "fitted logistic regression",
        "version": model["version"],
        "training_entries": model["training"]["entries"],
        "roc_auc_cv": model["performance"]["roc_auc"],
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def build(
    recorder: _Recorder,
    *,
    identifier: str,
    kind: str,
    structure_path: Path,
    parameters: dict[str, Any],
    started: datetime,
) -> dict[str, Any]:
    """The provenance block of one analysis."""
    structure: dict[str, Any] = {"identifier": identifier, "kind": kind} | file_fingerprint(structure_path)
    if kind != "file":
        # A downloaded structure is cached; its file time is when it was fetched.
        structure["retrieved_utc"] = _utc(structure_path.stat().st_mtime)
        structure["source"] = "AlphaFold DB" if kind == "alphafold" else "RCSB PDB"

    services = []
    for name, use in sorted(recorder.uses.items()):
        source = BY_NAME[name]
        services.append({
            "service": name,
            "licence": source.licence,
            "requests": use.requests,
            "from_cache": use.from_cache,
            "expired_cache_used": use.stale_fallback,
            "oldest_data_utc": _utc(use.oldest_retrieval),
            "newest_data_utc": _utc(use.newest_retrieval),
        })

    return {
        "generated_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "proteinscope": __version__,
        "python": platform.python_version(),
        "libraries": {name: _version(name) for name in LIBRARIES},
        "pocket_model": model_summary(),
        "parameters": parameters,
        "structure": structure,
        "data_sources": services,
    }


def parameters_of(params: Any, probe_pockets: int, online: bool) -> dict[str, Any]:
    values = asdict(params) if params is not None else {}
    return values | {"probe_pockets": probe_pockets, "online_context": online}
