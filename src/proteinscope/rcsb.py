"""Thin client for the public RCSB PDB services: search, metadata and downloads."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import requests

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
ASSEMBLY_URL = "https://data.rcsb.org/rest/v1/core/assembly/{pdb_id}/{assembly_id}"
FILE_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
_TIMEOUT = 30
_HEADERS = {"User-Agent": "ProteinScope/0.1"}


class RCSBError(RuntimeError):
    """A request to an RCSB service failed or returned something unusable."""


def normalize_pdb_id(pdb_id: str) -> str:
    candidate = pdb_id.strip().upper()
    if not _PDB_ID.match(candidate):
        raise ValueError(f"Invalid PDB ID {pdb_id!r}: expected 4 characters such as 1HSG")
    return candidate


def default_cache_dir() -> Path:
    return Path(os.environ.get("PROTEINSCOPE_CACHE", Path.home() / ".cache" / "proteinscope"))


def _get(url: str, deadline: float) -> requests.Response:
    """GET through the shared, deadline-bounded fetch; errors become RCSBError."""
    from .webdata import DataSourceError, _fetch  # webdata imports this module.

    try:
        return _fetch(url, deadline)
    except DataSourceError as exc:
        raise RCSBError(str(exc)) from exc


def _get_json(url: str) -> dict[str, Any]:
    from . import provenance

    response = _get(url, 90)
    if response.status_code == 404:
        raise RCSBError(f"Not found: {url}")
    if not response.ok:
        raise RCSBError(f"{url} returned HTTP {response.status_code}")
    provenance.record(url, provenance.FROM_NETWORK)
    return response.json()


def download_structure(pdb_id: str, cache_dir: Path | None = None) -> Path:
    """Download an entry as mmCIF, which unlike PDB format exists for every entry."""
    pdb_id = normalize_pdb_id(pdb_id)
    directory = cache_dir or default_cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    from . import provenance

    target = directory / f"{pdb_id}.cif"
    url = FILE_URL.format(pdb_id=pdb_id)
    if target.exists() and target.stat().st_size > 0:
        provenance.record(url, provenance.FROM_CACHE, target.stat().st_mtime)
        return target
    try:
        response = _get(url, 360)  # Large assemblies run to tens of megabytes.
    except RCSBError as exc:
        raise RCSBError(f"Download of {pdb_id} failed: {exc}") from exc
    if response.status_code == 404:
        raise RCSBError(f"PDB entry {pdb_id} does not exist")
    if not response.ok:
        raise RCSBError(f"Download of {pdb_id} returned HTTP {response.status_code}")
    temporary = target.with_suffix(".cif.part")
    temporary.write_bytes(response.content)
    temporary.replace(target)
    provenance.record(url, provenance.FROM_NETWORK)
    return target


def search(
    query: str,
    max_results: int = 10,
    max_resolution: float | None = None,
    require_ligand: bool = False,
    organism: str | None = None,
) -> list[dict[str, Any]]:
    """Full-text search with optional filters, returning summaries ordered by relevance."""
    nodes: list[dict[str, Any]] = [{
        "type": "terminal",
        "service": "full_text",
        "parameters": {"value": query},
    }]

    def attribute(name: str, operator: str, value: Any) -> None:
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {"attribute": name, "operator": operator, "value": value},
        })

    if max_resolution is not None:
        attribute("rcsb_entry_info.resolution_combined", "less_or_equal", max_resolution)
    if require_ligand:
        attribute("rcsb_entry_info.nonpolymer_entity_count", "greater", 0)
    if organism:
        attribute("rcsb_entity_source_organism.taxonomy_lineage.name", "exact_match", organism)

    payload = {
        "query": nodes[0] if len(nodes) == 1 else {
            "type": "group", "logical_operator": "and", "nodes": nodes,
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max(1, min(max_results, 50))},
            "results_content_type": ["experimental"],
        },
    }
    try:
        response = requests.post(SEARCH_URL, json=payload, headers=_HEADERS, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise RCSBError(f"RCSB search failed: {exc}") from exc
    if response.status_code == 204:  # RCSB answers "no hits" with an empty 204.
        return []
    if not response.ok:
        raise RCSBError(f"RCSB search returned HTTP {response.status_code}: {response.text[:200]}")
    hits = response.json().get("result_set", [])
    ids = [hit["identifier"] for hit in hits]
    summaries = {entry["pdb_id"]: entry for entry in _summaries(ids)}
    return [
        {**summaries.get(hit["identifier"], {"pdb_id": hit["identifier"]}), "score": round(hit.get("score", 0.0), 3)}
        for hit in hits
    ]


def _summaries(ids: list[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    query = """
    query($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        struct { title }
        exptl { method }
        rcsb_entry_info { resolution_combined molecular_weight deposited_polymer_monomer_count }
        nonpolymer_entities { pdbx_entity_nonpoly { comp_id name } }
      }
    }"""
    try:
        response = requests.post(
            GRAPHQL_URL, json={"query": query, "variables": {"ids": ids}},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RCSBError(f"RCSB metadata query failed: {exc}") from exc
    entries = (response.json().get("data") or {}).get("entries") or []
    result = []
    for entry in entries:
        info = entry.get("rcsb_entry_info") or {}
        resolution = info.get("resolution_combined") or [None]
        result.append({
            "pdb_id": entry["rcsb_id"],
            "title": (entry.get("struct") or {}).get("title"),
            "method": ", ".join(e["method"] for e in entry.get("exptl") or []),
            "resolution_A": resolution[0],
            "molecular_weight_kDa": info.get("molecular_weight"),
            "residues": info.get("deposited_polymer_monomer_count"),
            "ligands": [
                {"id": n["pdbx_entity_nonpoly"]["comp_id"], "name": n["pdbx_entity_nonpoly"]["name"]}
                for n in entry.get("nonpolymer_entities") or []
                if n.get("pdbx_entity_nonpoly")
            ],
        })
    return result


def entry_metadata(pdb_id: str) -> dict[str, Any]:
    """Return the fields of an entry that matter for judging and describing it."""
    pdb_id = normalize_pdb_id(pdb_id)
    raw = _get_json(ENTRY_URL.format(pdb_id=pdb_id))
    info = raw.get("rcsb_entry_info") or {}
    citation = raw.get("rcsb_primary_citation") or {}
    refine = (raw.get("refine") or [{}])[0]
    metadata = {
        "pdb_id": pdb_id,
        "title": (raw.get("struct") or {}).get("title"),
        "keywords": (raw.get("struct_keywords") or {}).get("pdbx_keywords"),
        "method": ", ".join(e.get("method", "") for e in raw.get("exptl") or []),
        "resolution_A": (info.get("resolution_combined") or [None])[0],
        "r_free": refine.get("ls_rfactor_rfree"),
        "r_work": refine.get("ls_rfactor_rwork"),
        "deposited_chains": info.get("deposited_polymer_entity_instance_count"),
        "assembly_count": info.get("assembly_count"),
        "citation_title": citation.get("title"),
        "citation_year": citation.get("year"),
        "citation_doi": citation.get("pdbx_database_id_doi"),
    }
    assemblies = (raw.get("rcsb_entry_container_identifiers") or {}).get("assembly_ids") or []
    if assemblies:
        try:
            assembly = _get_json(ASSEMBLY_URL.format(pdb_id=pdb_id, assembly_id=assemblies[0]))
            metadata["assembly_1_chains"] = (assembly.get("rcsb_assembly_info") or {}).get(
                "polymer_entity_instance_count"
            )
        except RCSBError:
            metadata["assembly_1_chains"] = None
    return metadata
