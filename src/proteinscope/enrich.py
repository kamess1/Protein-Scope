"""Context beyond the coordinates: ligand chemistry, target bioactivity and model confidence.

Every lookup is optional. A failed or skipped lookup is recorded in `context["notes"]`, and
the report leaves the section out, so the analysis never depends on these services.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import alphafold, chemistry, targets
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

    try:
        if source == "alphafold":
            accessions = [{
                "accession": result.metadata["uniprot"],
                "description": result.metadata.get("uniprot_description"),
                "chains": result.structure.chains,
            }]
        else:
            accessions = targets.uniprot_for_entry(result.identifier)
        context["targets"] = targets.target_context(accessions, cache_dir)
    except (DataSourceError, ValueError) as exc:
        note(f"Target context unavailable: {exc}")
    return context
