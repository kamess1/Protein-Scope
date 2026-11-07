"""AlphaFold DB models: identifiers, download and confidence (pLDDT) handling.

AlphaFold writes the per-residue pLDDT (0-100) into the B-factor column, so the audit and the
pocket table read confidence from there for these models.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .rcsb import default_cache_dir
from .webdata import DataSourceError, download, get_json

PREDICTION_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
_AF_ID = re.compile(r"^AF[-_:]?([A-Z0-9]{6,10})(?:-F1)?$", re.IGNORECASE)
CONFIDENT_PLDDT = 70.0
VERY_LOW_PLDDT = 50.0


def parse_alphafold_id(source: str) -> str | None:
    """Return the UniProt accession for 'AF-P00533' / 'AF:P00533' / 'AF-P00533-F1', else None."""
    match = _AF_ID.match(source.strip())
    return match.group(1).upper() if match else None


def identifier(accession: str) -> str:
    return f"AF-{accession.upper()}"


def download_model(accession: str, cache_dir: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Download the current model of a UniProt accession; returns (path, metadata)."""
    entries = get_json(PREDICTION_URL.format(accession=accession), cache_dir)
    if not entries:
        raise DataSourceError(f"AlphaFold DB has no model for {accession}")
    entry = entries[0]
    directory = (cache_dir or default_cache_dir()) / "alphafold"
    path = download(entry["cifUrl"], directory / f"{entry['entryId']}-v{entry.get('latestVersion', 0)}.cif")
    description = entry.get("uniprotDescription") or accession
    gene = entry.get("gene")
    organism = entry.get("organismScientificName")
    metadata = {
        "source": "alphafold",
        "uniprot": accession,
        "entry_id": entry.get("entryId"),
        "title": f"AlphaFold model of {description}" + (f" ({gene}, {organism})" if gene else ""),
        "method": "PREDICTED MODEL (AlphaFold DB)",
        "model_version": entry.get("latestVersion"),
        "tool": entry.get("toolUsed"),
        "mean_plddt": entry.get("globalMetricValue"),
        "fraction_plddt_very_high": entry.get("fractionPlddtVeryHigh"),
        "fraction_plddt_confident": entry.get("fractionPlddtConfident"),
        "fraction_plddt_low": entry.get("fractionPlddtLow"),
        "fraction_plddt_very_low": entry.get("fractionPlddtVeryLow"),
        "pae_image_url": entry.get("paeImageUrl"),
        "gene": gene,
        "organism": organism,
        "uniprot_description": description,
    }
    return path, metadata


def residue_plddt(residues: list[Any], bfactors: np.ndarray) -> dict[Any, float]:
    """Mean pLDDT per residue key (all atoms of a residue carry the same value in AF files)."""
    sums: dict[Any, list[float]] = {}
    for key, value in zip(residues, bfactors):
        sums.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in sums.items()}


def plddt_band(value: float) -> str:
    if value >= 90:
        return "very high"
    if value >= CONFIDENT_PLDDT:
        return "confident"
    if value >= VERY_LOW_PLDDT:
        return "low"
    return "very low"
