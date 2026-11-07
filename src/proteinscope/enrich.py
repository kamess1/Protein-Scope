"""Context beyond the coordinates: ligand chemistry, target bioactivity and model confidence.

Every lookup is optional. A failed or skipped lookup is recorded in `context["notes"]`, and
the report leaves the section out, so the analysis never depends on these services.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import alphafold, chemistry, targets, uniprot
from .webdata import DataSourceError

if TYPE_CHECKING:
    from .pipeline import AnalysisResult


def plddt_context(result: AnalysisResult) -> dict[str, Any]:
    """Per-pocket confidence for AlphaFold models, read from the B-factor column."""
    table = result.structure.protein
    per_residue = alphafold.residue_plddt(table.residues, table.bfactors)
    pockets = {}
    for pocket in result.pockets:
        values = [per_residue[key] for key in pocket.lining_keys if key in per_residue]
        if values:
            mean = float(np.mean(values))
            pockets[pocket.pocket_id] = {
                "mean_plddt": round(mean, 1),
                "min_plddt": round(float(np.min(values)), 1),
                "band": alphafold.plddt_band(mean),
            }
    return {"mean_plddt": round(float(np.mean(list(per_residue.values()))), 1) if per_residue else None,
            "pockets": pockets}


def ligand_chemistry(result: AnalysisResult, cache_dir: Path | None = None) -> dict[str, Any]:
    """CCD SMILES and RDKit descriptors per ligand label (one lookup per component ID)."""
    by_component: dict[str, dict[str, Any]] = {}
    output: dict[str, Any] = {}
    for ligand in result.structure.ligands:
        comp_id = ligand.key.name
        if comp_id not in by_component:
            entry = chemistry.component(comp_id, cache_dir)
            if entry.get("smiles") and chemistry.available():
                try:
                    entry["properties"] = chemistry.properties(entry["smiles"])
                    entry["svg"] = chemistry.depiction_svg(entry["smiles"])
                except ValueError as exc:
                    entry["error"] = str(exc)
            by_component[comp_id] = entry
        output[ligand.label] = by_component[comp_id]
    return output


_SITE_CATEGORIES = ("catalytic", "binding", "functional")

# Category -> the phrase the reports use, in the order they are listed.
CATEGORY_LABELS = {
    "catalytic": "active site", "binding": "cofactor site", "functional": "annotated site",
    "modification": "modification", "disulfide": "disulfide", "mutagenesis": "mutagenesis",
}


def annotation_summary(row: dict[str, Any] | None) -> str:
    """One compact phrase for the functional annotation of a pocket."""
    if not row:
        return "-"
    parts = []
    if row.get("cofactors"):
        parts.append("binds " + ", ".join(row["cofactors"]))
    for category, label in CATEGORY_LABELS.items():
        count = row["categories"].get(category)
        if count:
            parts.append(label if count == 1 else f"{label} ({count})")
    if row.get("variants"):
        parts.append(f"{row['variants']} variants")
    return "; ".join(parts) or "-"


def functional_context(
    result: AnalysisResult, accessions: list[dict[str, Any]], cache_dir: Path | None = None
) -> dict[str, Any]:
    """UniProt features of the pocket-lining residues, per UniProt accession of the entry.

    The structure's residue numbers are mapped onto the reference sequence by alignment, so
    truncated constructs, engineered mutations and gaps do not shift the annotation.
    """
    proteins: list[dict[str, Any]] = []
    pockets: dict[int, dict[str, Any]] = {}
    for item in accessions:
        data = uniprot.entry(item["accession"], cache_dir)
        reference = uniprot.sequence(data)
        chains = item.get("chains") or result.structure.chains
        positions, identities = uniprot.residue_positions(
            result.structure.protein_residues, chains, reference
        )
        by_position: dict[int, list[dict[str, Any]]] = {}
        for feature in uniprot.features(data):
            for position in feature["positions"]:
                by_position.setdefault(position, []).append(feature)
        regions = uniprot.regions(data)
        protein = uniprot.description(data) | {
            "chain_identity": identities,
            "mapped_residues": len(positions),
            "annotated_positions": len(by_position),
            "regions": regions,
        }
        proteins.append(protein)
        if not positions:
            continue

        for pocket in result.pockets:
            residues = []
            counts: dict[str, int] = {}
            for key in pocket.lining_keys:
                position = positions.get(key)
                if position is None:
                    continue
                hits = by_position.get(position, [])
                if not hits:
                    continue
                for feature in hits:
                    counts[feature["category"]] = counts.get(feature["category"], 0) + 1
                residues.append({
                    "residue": key.label,
                    "uniprot_position": position,
                    "features": [
                        {"type": f["type"], "category": f["category"],
                         "description": f["description"], "ligand": f["ligand"]}
                        for f in hits
                    ],
                })
            covered = [positions[key] for key in pocket.lining_keys if key in positions]
            entry_regions = [
                region["description"] or region["type"] for region in regions
                if covered and any(region["start"] <= p <= region["end"] for p in covered)
            ]
            if not residues and not entry_regions:
                continue
            row = pockets.setdefault(pocket.pocket_id, {
                "residues": [], "categories": {}, "regions": [], "accessions": [],
            })
            row["residues"].extend(residues)
            row["regions"] = sorted(set(row["regions"]) | set(entry_regions))
            row["accessions"] = sorted(set(row["accessions"]) | {protein["accession"]})
            for category, count in counts.items():
                row["categories"][category] = row["categories"].get(category, 0) + count

    for row in pockets.values():
        row["is_functional_site"] = any(row["categories"].get(c) for c in _SITE_CATEGORIES)
        row["variants"] = row["categories"].get("variant", 0)
        row["cofactors"] = sorted({
            feature["ligand"]
            for residue in row["residues"] for feature in residue["features"]
            if feature["ligand"]
        })
    return {"proteins": proteins, "pockets": pockets}


def enrich(result: AnalysisResult, cache_dir: Path | None = None, online: bool = True) -> dict[str, Any]:
    context: dict[str, Any] = {"notes": []}
    note = context["notes"].append
    source = result.metadata.get("source")
    if source == "alphafold":
        context["plddt"] = plddt_context(result)
    if not online:
        note("Online context (ligand chemistry, ChEMBL) was not requested.")
        return context

    if result.structure.ligands:
        if not chemistry.available():
            note('RDKit is not installed, so ligand descriptors are missing: pip install "proteinscope[chem]".')
        try:
            context["ligand_chemistry"] = ligand_chemistry(result, cache_dir)
        except DataSourceError as exc:
            note(f"Ligand chemistry unavailable: {exc}")

    accessions: list[dict[str, Any]] = []
    try:
        if source == "alphafold":
            accessions = [{
                "accession": result.metadata["uniprot"],
                "description": result.metadata.get("uniprot_description"),
                "chains": result.structure.chains,
            }]
        else:
            accessions = targets.uniprot_for_entry(result.identifier)
    except (DataSourceError, ValueError, KeyError) as exc:
        note(f"UniProt accessions unavailable: {exc}")

    if accessions:
        try:
            context["targets"] = targets.target_context(accessions, cache_dir)
        except (DataSourceError, ValueError) as exc:
            note(f"Target context unavailable: {exc}")
        try:
            context["annotations"] = functional_context(result, accessions, cache_dir)
        except (DataSourceError, ValueError, KeyError) as exc:
            note(f"UniProt annotation unavailable: {exc}")
    else:
        note("No UniProt accession for this entry, so there is no functional annotation.")
    return context
