"""Quality audit of a structure: what can and cannot be trusted before analysing it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .constants import MODIFIED_RESIDUES, STANDARD_RESIDUES
from .structure import LoadedStructure, canonical_name, residue_key

_PEPTIDE_BOND_MAX = 1.6
_CLASH_DISTANCE = 2.2


@dataclass
class AuditReport:
    title: str | None
    method: str | None
    resolution_A: float | None
    r_free: float | None
    r_work: float | None
    chains: dict[str, int]
    residue_count: int
    heavy_atom_count: int
    ligands: list[str]
    additives: list[str]
    ions: list[str]
    water_count: int
    nucleic_residue_count: int
    unobserved_residue_count: int
    chain_breaks: list[str]
    incomplete_residues: list[str]
    modified_residues: list[str]
    disordered_atom_count: int
    low_occupancy_fraction: float
    mean_bfactor: float
    high_bfactor_residues: list[str]
    clash_count: int
    warnings: list[str] = field(default_factory=list)
    quality_grade: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Expected heavy atoms per residue, backbone N/CA/C/O included (OXT excluded).
_EXPECTED_HEAVY_ATOMS = {
    "ALA": 5, "ARG": 11, "ASN": 8, "ASP": 8, "CYS": 6, "GLN": 9, "GLU": 9, "GLY": 4,
    "HIS": 10, "ILE": 8, "LEU": 8, "LYS": 9, "MET": 8, "PHE": 11, "PRO": 7, "SER": 6,
    "THR": 7, "TRP": 14, "TYR": 12, "VAL": 7,
}


def _chain_breaks(structure: LoadedStructure) -> list[str]:
    breaks = []
    previous = None
    for residue in structure.protein_residues:
        if previous is not None and previous.get_parent() is residue.get_parent():
            if "C" in previous and "N" in residue:
                distance = float(previous["C"] - residue["N"])
                if distance > _PEPTIDE_BOND_MAX + 0.4:
                    breaks.append(
                        f"{residue_key(previous).label} -> {residue_key(residue).label} ({distance:.1f} A)"
                    )
        previous = residue
    return breaks


def _incomplete(structure: LoadedStructure) -> list[str]:
    counts: dict[Any, int] = {}
    for key in structure.protein.residues:
        counts[key] = counts.get(key, 0) + 1
    return [
        key.label for key, count in counts.items()
        if key.name in STANDARD_RESIDUES and count < _EXPECTED_HEAVY_ATOMS[canonical_name(key.name)]
    ]


def _clashes(structure: LoadedStructure) -> int:
    """Count close contacts between atoms of residues that are not sequence neighbours."""
    table = structure.protein
    pairs = cKDTree(table.coords).query_pairs(_CLASH_DISTANCE, output_type="ndarray")
    count = 0
    for i, j in pairs:
        a, b = table.residues[i], table.residues[j]
        if a.chain == b.chain and abs(a.number - b.number) <= 1:
            continue
        if table.names[i] == "SG" and table.names[j] == "SG":  # Disulfide bonds are ~2.05 A.
            continue
        count += 1
    return count


def _grade(report: AuditReport) -> str:
    resolution = report.resolution_A
    if resolution is None:
        return "unknown"
    penalties = 0
    if report.r_free is not None and report.r_free > 0.30:
        penalties += 1
    if report.heavy_atom_count and report.clash_count / report.heavy_atom_count > 0.01:
        penalties += 1
    if len(report.chain_breaks) > 3:
        penalties += 1
    tier = 0 if resolution <= 2.0 else 1 if resolution <= 2.8 else 2 if resolution <= 3.5 else 3
    return ["high", "good", "moderate", "low"][min(tier + penalties, 3)]


def audit_structure(structure: LoadedStructure, metadata: dict[str, Any] | None = None) -> AuditReport:
    metadata = metadata or {}
    header = structure.header
    table = structure.protein

    chains: dict[str, set] = {}
    for key in table.residues:
        chains.setdefault(key.chain, set()).add(key)

    per_residue: dict[Any, list[float]] = {}
    for key, value in zip(table.residues, table.bfactors):
        per_residue.setdefault(key, []).append(float(value))
    mean_by_residue = {key: float(np.mean(values)) for key, values in per_residue.items()}
    overall = float(np.mean(table.bfactors)) if len(table) else 0.0
    spread = float(np.std(list(mean_by_residue.values()))) if mean_by_residue else 0.0
    high_b = [key.label for key, value in mean_by_residue.items() if spread and value > overall + 2 * spread]

    disordered = sum(
        1 for residue in structure.protein_residues for atom in residue.get_atoms()
        if atom.is_disordered()
    )
    unobserved = header.get("unobserved_residues", [])
    resolution = header.get("resolution_A") or metadata.get("resolution_A")

    report = AuditReport(
        title=header.get("title") or metadata.get("title"),
        method=header.get("method") or metadata.get("method"),
        resolution_A=resolution,
        r_free=header.get("r_free") or metadata.get("r_free"),
        r_work=header.get("r_work") or metadata.get("r_work"),
        chains={chain: len(keys) for chain, keys in sorted(chains.items())},
        residue_count=len(per_residue),
        heavy_atom_count=len(table),
        ligands=[item.label for item in structure.ligands],
        additives=[item.label for item in structure.additives],
        ions=[item.label for item in structure.ions],
        water_count=structure.water_count,
        nucleic_residue_count=structure.nucleic_residue_count,
        unobserved_residue_count=len(unobserved),
        chain_breaks=_chain_breaks(structure),
        incomplete_residues=_incomplete(structure),
        modified_residues=sorted({k.label for k in per_residue if k.name in MODIFIED_RESIDUES}),
        disordered_atom_count=disordered,
        low_occupancy_fraction=float(np.mean(table.occupancies < 0.5)) if len(table) else 0.0,
        mean_bfactor=overall,
        high_bfactor_residues=high_b,
        clash_count=_clashes(structure),
    )

    warn = report.warnings.append
    if resolution is None:
        warn("No resolution available; side-chain positions cannot be assessed.")
    elif resolution > 3.0:
        warn(f"Resolution {resolution:.2f} A: side-chain and ligand poses are uncertain.")
    if report.r_free is not None and report.r_free > 0.30:
        warn(f"R-free {report.r_free:.3f} is high.")
    if report.chain_breaks or report.unobserved_residue_count:
        warn(
            f"{report.unobserved_residue_count} unobserved residues and {len(report.chain_breaks)} "
            "chain breaks: flexible loops are absent and may reshape pockets near the gaps."
        )
    if report.incomplete_residues:
        warn(f"{len(report.incomplete_residues)} residues lack side-chain atoms.")
    if report.clash_count:
        warn(f"{report.clash_count} atom pairs closer than {_CLASH_DISTANCE} A.")
    assembly_chains = metadata.get("assembly_1_chains")
    if assembly_chains and assembly_chains != metadata.get("deposited_chains"):
        warn(
            f"The deposited model has {metadata.get('deposited_chains')} chain(s) but biological "
            f"assembly 1 has {assembly_chains}. This analysis uses the deposited model, so pockets "
            "at subunit interfaces can be missing or open."
        )
    if not structure.ligands:
        warn("No drug-like ligand present: pocket detection cannot be checked against a bound pose.")
    report.quality_grade = _grade(report)
    return report
