"""Deterministic analysis pipeline. Agents and the CLI both call into this module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree

from . import alphafold, rcsb
from .audit import AuditReport, audit_structure
from .druggability import rank_pockets
from .electrostatics import ChargeModel, build_charge_model, potential_at
from .enrich import enrich
from .interactions import LigandProfile, ProbeResult, profile_ligand, simulate_probes
from .pockets import Grid, Pocket, PocketParams, detect_pockets, ligand_pocket_overlap
from .structure import LigandInstance, LoadedStructure, ResidueKey, load_structure
from .surface import SurfaceResult, analyze_surface

# Called as progress(stage, fraction) at each step; fraction runs from 0 to 1.
Progress = Callable[[str, float], None]

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
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def is_model(self) -> bool:
        """True for predicted models, whose B-factor column holds pLDDT."""
        return self.metadata.get("source") == "alphafold"

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


def resolve_source(
    source: str, cache_dir: Path | None = None, allow_files: bool = True
) -> tuple[Path, str, str]:
    """Turn a file path, PDB ID or AlphaFold ID into (path, identifier, kind).

    kind is "file", "pdb" or "alphafold". With `allow_files=False` the source is only ever
    treated as a database identifier, even if a file of that name exists; the web server
    uses this so a request from the browser can never read a local file.
    """
    if allow_files:
        path = Path(source).expanduser()
        if path.is_file():
            return path, path.stem.upper(), "file"
    accession = alphafold.parse_alphafold_id(source)
    if accession:
        model_path, _ = alphafold.download_model(accession, cache_dir)
        return model_path, alphafold.identifier(accession), "alphafold"
    try:
        pdb_id = rcsb.normalize_pdb_id(source)
    except ValueError as exc:
        raise ValueError(
            f"{source!r} is neither an existing file, a PDB ID (1HSG) nor an AlphaFold ID (AF-P00533)"
        ) from exc
    return rcsb.download_structure(pdb_id, cache_dir), pdb_id, "pdb"


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
    context: bool = True,
    progress: Progress | None = None,
    allow_files: bool = True,
) -> AnalysisResult:
    """Analyse a structure. `context=False` skips the online lookups (chemistry, ChEMBL).

    `progress`, if given, is told which stage starts and how far along the run is.
    """
    step = progress or (lambda stage, fraction: None)
    step("Downloading structure", 0.02)
    path, identifier, kind = resolve_source(source, cache_dir, allow_files=allow_files)
    step("Reading structure", 0.12)
    structure = load_structure(path, identifier)

    metadata: dict[str, Any] = {}
    if kind == "pdb":
        try:
            metadata = rcsb.entry_metadata(identifier)
        except rcsb.RCSBError as exc:  # Coordinates are cached, so metadata alone is not fatal.
            metadata = {"metadata_error": str(exc)}
    elif kind == "alphafold":
        _, metadata = alphafold.download_model(identifier.removeprefix("AF-"), cache_dir)

    step("Checking structure quality", 0.18)
    audit = audit_structure(structure, metadata)
    step("Computing surface", 0.24)
    surface = analyze_surface(structure)
    step("Estimating electrostatics", 0.36)
    charge_model = build_charge_model(structure)
    atom_potential = potential_at(_surface_points(structure.protein.coords), charge_model)

    weights: dict[ResidueKey, list[tuple[float, float]]] = {}
    for key, value, area in zip(structure.protein.residues, atom_potential, surface.atom_sasa):
        weights.setdefault(key, []).append((float(value), float(area)))
    residue_potential = {}
    for key, pairs in weights.items():
        values, areas = zip(*pairs)
        residue_potential[key] = float(np.average(values, weights=areas)) if sum(areas) else float(np.mean(values))

    step("Searching pockets", 0.44)
    pockets, grid = detect_pockets(structure.protein, charge_model, params)
    pockets = rank_pockets(pockets)

    step("Profiling ligands", 0.58)
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
    step("Simulating probes", 0.64)
    for pocket in pockets[:probe_pockets]:
        result.probes_for(pocket.pocket_id)
    # Local files have no database identity, so only the offline part of the context applies.
    step("Looking up UniProt and ChEMBL", 0.74)
    result.context = enrich(result, cache_dir, online=context and kind != "file")
    return result
