"""Deterministic analysis pipeline. Agents and the CLI both call into this module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from . import rcsb
from .audit import AuditReport, audit_structure
from .druggability import rank_pockets
from .electrostatics import ChargeModel, build_charge_model, potential_at
from .interactions import LigandProfile, ProbeResult, profile_ligand, simulate_probes
from .pockets import Grid, Pocket, PocketParams, detect_pockets, ligand_pocket_overlap
from .structure import LigandInstance, LoadedStructure, ResidueKey, load_structure
from .surface import SurfaceResult, analyze_surface

_LIGAND_POCKET_MIN_OVERLAP = 0.5
_SURFACE_OFFSET = 3.0


@dataclass
class AnalysisResult:
    identifier: str
    structure: LoadedStructure
    metadata: dict[str, Any]
    audit: AuditReport
    surface: SurfaceResult
    charge_model: ChargeModel
    atom_surface_potential: np.ndarray
    residue_potential: dict[ResidueKey, float]
    pockets: list[Pocket]
    grid: Grid
    ligand_profiles: dict[str, LigandProfile]
    ligand_pockets: dict[str, list[dict[str, float]]]
    probes: dict[int, list[ProbeResult]] = field(default_factory=dict)

    def pocket(self, pocket_id: int) -> Pocket:
        for pocket in self.pockets:
            if pocket.pocket_id == pocket_id:
                return pocket
        raise KeyError(f"Pocket {pocket_id} does not exist; valid IDs are 1-{len(self.pockets)}")

    def probes_for(self, pocket_id: int) -> list[ProbeResult]:
        if pocket_id not in self.probes:
            self.probes[pocket_id] = simulate_probes(
                self.pocket(pocket_id), self.structure.protein, self.charge_model
            )
        return self.probes[pocket_id]

    def ligand(self, label: str) -> LigandInstance:
        for ligand in self.structure.ligands:
            if ligand.label == label:
                return ligand
        known = [ligand.label for ligand in self.structure.ligands]
        raise KeyError(f"Ligand {label!r} not found; available: {known}")


def resolve_source(source: str, cache_dir: Path | None = None) -> tuple[Path, str, bool]:
    """Turn a file path or PDB ID into (path, identifier, is_remote_entry)."""
    path = Path(source).expanduser()
    if path.is_file():
        return path, path.stem.upper(), False
    try:
        pdb_id = rcsb.normalize_pdb_id(source)
    except ValueError as exc:
        raise ValueError(f"{source!r} is neither an existing file nor a valid PDB ID") from exc
    return rcsb.download_structure(pdb_id, cache_dir), pdb_id, True


def _surface_points(coords: np.ndarray) -> np.ndarray:
    """Push each atom outward along its local density gradient to sample the solvent side."""
    tree = cKDTree(coords)
    points = np.empty_like(coords)
    for index, neighbors in enumerate(tree.query_ball_point(coords, 8.0)):
        direction = coords[index] - coords[neighbors].mean(axis=0)
        norm = np.linalg.norm(direction)
        points[index] = coords[index] + (direction / norm * _SURFACE_OFFSET if norm > 1e-6 else 0.0)
    return points


def run_analysis(
    source: str,
    params: PocketParams | None = None,
    probe_pockets: int = 3,
    cache_dir: Path | None = None,
) -> AnalysisResult:
    path, identifier, remote = resolve_source(source, cache_dir)
    structure = load_structure(path, identifier)

    metadata: dict[str, Any] = {}
    if remote:
        try:
            metadata = rcsb.entry_metadata(identifier)
        except rcsb.RCSBError as exc:  # Coordinates are cached, so metadata alone is not fatal.
            metadata = {"metadata_error": str(exc)}

    audit = audit_structure(structure, metadata)
    surface = analyze_surface(structure)
    charge_model = build_charge_model(structure)
    atom_potential = potential_at(_surface_points(structure.protein.coords), charge_model)

    weights: dict[ResidueKey, list[tuple[float, float]]] = {}
    for key, value, area in zip(structure.protein.residues, atom_potential, surface.atom_sasa):
        weights.setdefault(key, []).append((float(value), float(area)))
    residue_potential = {}
    for key, pairs in weights.items():
        values, areas = zip(*pairs)
        residue_potential[key] = float(np.average(values, weights=areas)) if sum(areas) else float(np.mean(values))

    pockets, grid = detect_pockets(structure.protein, charge_model, params)
    pockets = rank_pockets(pockets)

    profiles = {
        ligand.label: profile_ligand(structure, ligand, charge_model) for ligand in structure.ligands
    }
    mapping: dict[str, list[dict[str, float]]] = {}
    for ligand in structure.ligands:
        overlaps = [
            {"pocket_id": p.pocket_id, "overlap": round(ligand_pocket_overlap(ligand.atoms.coords, p), 3)}
            for p in pockets
        ]
        mapping[ligand.label] = sorted(
            (o for o in overlaps if o["overlap"] >= _LIGAND_POCKET_MIN_OVERLAP),
            key=lambda o: -o["overlap"],
        )

    result = AnalysisResult(
        identifier=identifier,
        structure=structure,
        metadata=metadata,
        audit=audit,
        surface=surface,
        charge_model=charge_model,
        atom_surface_potential=atom_potential,
        residue_potential=residue_potential,
        pockets=pockets,
        grid=grid,
        ligand_profiles=profiles,
        ligand_pockets=mapping,
    )
    for pocket in pockets[:probe_pockets]:
        result.probes_for(pocket.pocket_id)
    return result
