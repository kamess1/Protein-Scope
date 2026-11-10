"""Target context: UniProt accessions of a PDB entry and what ChEMBL knows about them.

Known bioactivity and approved drugs are the strongest evidence that a target is druggable;
the geometric pocket score is only a structural hint. This module puts both side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

import requests

from .rcsb import GRAPHQL_URL, normalize_pdb_id
from .webdata import DataSourceError, get_json

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
_UNIPROT_QUERY = """
query($id: String!) {
  entry(entry_id: $id) {
    polymer_entities {
      rcsb_polymer_entity { pdbx_description }
      rcsb_polymer_entity_container_identifiers { uniprot_ids auth_asym_ids }
    }
  }
}"""


def uniprot_for_entry(pdb_id: str) -> list[dict[str, Any]]:
    """UniProt accessions of the polymer entities in a PDB entry, with their chains."""
    pdb_id = normalize_pdb_id(pdb_id)
    try:
        response = requests.post(
            GRAPHQL_URL, json={"query": _UNIPROT_QUERY, "variables": {"id": pdb_id}}, timeout=30,
            headers={"User-Agent": "ProteinScope/0.2"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataSourceError(f"RCSB entity query failed: {exc}") from exc
    entry = (response.json().get("data") or {}).get("entry") or {}
    seen: dict[str, dict[str, Any]] = {}
    for entity in entry.get("polymer_entities") or []:
        ids = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        description = (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description")
        for accession in ids.get("uniprot_ids") or []:
            item = seen.setdefault(accession, {"accession": accession, "description": description, "chains": []})
            item["chains"] = sorted(set(item["chains"]) | set(ids.get("auth_asym_ids") or []))
    return list(seen.values())


def _chembl(path: str, params: dict[str, Any], cache_dir: Path | None) -> dict[str, Any]:
    return get_json(f"{CHEMBL}/{path}.json?{urlencode(params)}", cache_dir)


def chembl_target(accession: str, cache_dir: Path | None = None) -> dict[str, Any] | None:
    data = _chembl("target", {"target_components__accession": accession, "limit": 20}, cache_dir)
    targets = data.get("targets") or []
    if not targets:
        return None
    single = [t for t in targets if t.get("target_type") == "SINGLE PROTEIN"]
    chosen = (single or targets)[0]
    return {
        "target_chembl_id": chosen["target_chembl_id"],
        "pref_name": chosen.get("pref_name"),
        "target_type": chosen.get("target_type"),
        "organism": chosen.get("organism"),
    }


def chembl_activities(target_id: str, limit: int = 10, cache_dir: Path | None = None) -> dict[str, Any]:
    """Count of measured potencies and the most potent compounds (by pChEMBL)."""
    data = _chembl("activity", {
        "target_chembl_id": target_id, "pchembl_value__isnull": "false", "order_by": "-pchembl_value",
        "limit": max(1, min(limit * 3, 60)),
        "only": "molecule_chembl_id,molecule_pref_name,pchembl_value,standard_type,canonical_smiles",
    }, cache_dir)
    best: dict[str, dict[str, Any]] = {}
    for activity in data.get("activities") or []:
        molecule = activity.get("molecule_chembl_id")
        if not molecule or molecule in best:
            continue
        best[molecule] = {
            "molecule_chembl_id": molecule,
            "name": activity.get("molecule_pref_name"),
            "pchembl": float(activity["pchembl_value"]) if activity.get("pchembl_value") else None,
            "type": activity.get("standard_type"),
            "smiles": activity.get("canonical_smiles"),
        }
    return {
        "measured_activities": (data.get("page_meta") or {}).get("total_count", 0),
        "top_compounds": list(best.values())[:limit],
    }


def chembl_drugs(target_id: str, cache_dir: Path | None = None) -> list[dict[str, Any]]:
    """Compounds with a recorded mechanism of action on the target, highest phase first."""
    data = _chembl("mechanism", {"target_chembl_id": target_id, "limit": 100}, cache_dir)
    mechanisms = data.get("mechanisms") or []
    ids = sorted({m["molecule_chembl_id"] for m in mechanisms if m.get("molecule_chembl_id")})
    names: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), 40):
        chunk = ids[start:start + 40]
        molecules = _chembl("molecule", {
            "molecule_chembl_id__in": ",".join(chunk), "only": "molecule_chembl_id,pref_name,max_phase",
            "limit": 40,
        }, cache_dir)
        for molecule in molecules.get("molecules") or []:
            names[molecule["molecule_chembl_id"]] = molecule
    drugs = []
    for mechanism in mechanisms:
        molecule = names.get(mechanism.get("molecule_chembl_id"), {})
        phase = molecule.get("max_phase") or mechanism.get("max_phase")
        drugs.append({
            "molecule_chembl_id": mechanism.get("molecule_chembl_id"),
            "name": molecule.get("pref_name"),
            "max_phase": float(phase) if phase not in (None, "") else None,
            "mechanism": mechanism.get("mechanism_of_action"),
            "action_type": mechanism.get("action_type"),
        })
    drugs.sort(key=lambda d: -(d["max_phase"] or 0))
    return drugs


def target_context(
    accessions: list[dict[str, Any]],
    cache_dir: Path | None = None,
    limit: int = 8,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    say = progress or (lambda stage: None)
    rows = []
    for item in accessions:
        row: dict[str, Any] = {**item}
        say(f"Finding the ChEMBL target for {item['accession']}")
        target = chembl_target(item["accession"], cache_dir)
        row["chembl"] = target
        if target:
            say(f"Reading measured potencies from ChEMBL ({target['target_chembl_id']})")
            row.update(chembl_activities(target["target_chembl_id"], limit, cache_dir))
            say(f"Reading known drugs from ChEMBL ({target['target_chembl_id']})")
            row["drugs"] = chembl_drugs(target["target_chembl_id"], cache_dir)
            row["approved_drugs"] = sum(1 for d in row["drugs"] if (d["max_phase"] or 0) >= 4)
        rows.append(row)
    return rows
