"""Surface analysis: SASA, molecular dimensions and hydrophobic surface patches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Structure import Structure
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.spatial.distance import pdist

from .constants import KYTE_DOOLITTLE, MAX_ASA, VDW_RADII
from .structure import LoadedStructure, ResidueKey, canonical_name, residue_key

EXPOSED_THRESHOLD = 0.20
_PATCH_LINK_DISTANCE = 7.0
_PATCH_MIN_RESIDUES = 3
_APOLAR_ELEMENTS = {"C", "S"}


@dataclass
class ResidueSurface:
    key: ResidueKey
    sasa: float
    apolar_sasa: float
    relative_sasa: float
    hydropathy: float
    centroid: np.ndarray

    @property
    def exposed(self) -> bool:
        return self.relative_sasa >= EXPOSED_THRESHOLD


@dataclass
class HydrophobicPatch:
    rank: int
    residues: list[str]
    apolar_area: float
    center: list[float]


@dataclass
class SurfaceSummary:
    total_sasa: float
    apolar_sasa: float
    polar_sasa: float
    apolar_fraction: float
    exposed_residue_fraction: float
    max_diameter: float
    bounding_box: list[float]
    principal_extents: list[float]
    radius_of_gyration: float
    patches: list[HydrophobicPatch]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SurfaceResult:
    summary: SurfaceSummary
    residues: list[ResidueSurface]
    atom_sasa: np.ndarray


def max_diameter(coords: np.ndarray) -> float:
    """Largest atom-to-atom distance, taken over the convex hull to stay fast."""
    if len(coords) < 2:
        return 0.0
    points = coords
    if len(coords) >= 4:
        try:
            points = coords[ConvexHull(coords).vertices]
        except QhullError:
            points = coords
    return float(pdist(points).max())


def dimensions(coords: np.ndarray) -> dict[str, Any]:
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes.T
    return {
        "max_diameter": max_diameter(coords),
        "bounding_box": (coords.max(axis=0) - coords.min(axis=0)).round(2).tolist(),
        "principal_extents": (projected.max(axis=0) - projected.min(axis=0)).round(2).tolist(),
        "radius_of_gyration": float(np.sqrt((centered**2).sum(axis=1).mean())),
    }


def _protein_only(structure: LoadedStructure) -> Structure:
    """Copy the first model and drop water, ligands and hydrogens, so they do not bury surface."""
    copy = structure.biopython.copy()
    for model in list(copy)[1:]:
        copy.detach_child(model.get_id())
    keep = {residue_key(r) for r in structure.protein_residues}
    model = next(iter(copy))
    for chain in list(model):
        for residue in list(chain):
            if residue_key(residue) not in keep:
                chain.detach_child(residue.get_id())
                continue
            for atom in list(residue):
                if (atom.element or "").upper() in {"H", "D"}:
                    residue.detach_child(atom.get_id())
        if not len(chain):
            model.detach_child(chain.get_id())
    return copy


def analyze_surface(structure: LoadedStructure, n_points: int = 100) -> SurfaceResult:
    protein_copy = _protein_only(structure)
    radii = {element: radius for element, radius in VDW_RADII.items()}
    # Biopython looks radii up by element as written in the file (upper-cased by its parser).
    calculator = ShrakeRupley(probe_radius=1.4, n_points=n_points, radii_dict=radii)
    model = next(iter(protein_copy))
    calculator.compute(model, level="A")

    area: dict[tuple[ResidueKey, str], float] = {}
    for chain in model:
        for residue in chain:
            key = residue_key(residue)
            for atom in residue.get_atoms():
                area[(key, atom.get_name().strip())] = float(atom.sasa)

    table = structure.protein
    atom_sasa = np.array(
        [area.get((key, name), 0.0) for key, name in zip(table.residues, table.names)], dtype=float
    )

    grouped: dict[ResidueKey, list[int]] = {}
    for index, key in enumerate(table.residues):
        grouped.setdefault(key, []).append(index)

    residues = []
    for key, indices in grouped.items():
        name = canonical_name(key.name)
        idx = np.array(indices)
        total = float(atom_sasa[idx].sum())
        apolar = float(atom_sasa[idx][np.isin(table.elements[idx], list(_APOLAR_ELEMENTS))].sum())
        residues.append(ResidueSurface(
            key=key,
            sasa=total,
            apolar_sasa=apolar,
            relative_sasa=min(total / MAX_ASA.get(name, 200.0), 1.0),
            hydropathy=KYTE_DOOLITTLE.get(name, 0.0),
            centroid=table.coords[idx].mean(axis=0),
        ))

    total = float(atom_sasa.sum())
    apolar_total = float(sum(r.apolar_sasa for r in residues))
    dims = dimensions(table.coords)
    summary = SurfaceSummary(
        total_sasa=total,
        apolar_sasa=apolar_total,
        polar_sasa=total - apolar_total,
        apolar_fraction=apolar_total / total if total else 0.0,
        exposed_residue_fraction=float(np.mean([r.exposed for r in residues])),
        patches=_hydrophobic_patches(residues),
        **dims,
    )
    return SurfaceResult(summary, residues, atom_sasa)


def _hydrophobic_patches(residues: list[ResidueSurface]) -> list[HydrophobicPatch]:
    """Cluster exposed hydrophobic residues whose centroids lie close to each other."""
    candidates = [r for r in residues if r.hydropathy >= 1.8 and r.relative_sasa >= EXPOSED_THRESHOLD]
    if len(candidates) < _PATCH_MIN_RESIDUES:
        return []
    centroids = np.array([r.centroid for r in candidates])
    graph = cKDTree(centroids).sparse_distance_matrix(
        cKDTree(centroids), _PATCH_LINK_DISTANCE, output_type="coo_matrix"
    )
    _, labels = connected_components(graph, directed=False)
    patches = []
    for label in set(labels.tolist()):
        members = [candidates[i] for i in np.flatnonzero(labels == label)]
        if len(members) < _PATCH_MIN_RESIDUES:
            continue
        patches.append(HydrophobicPatch(
            rank=0,
            residues=[m.key.label for m in members],
            apolar_area=round(sum(m.apolar_sasa for m in members), 1),
            center=np.mean([m.centroid for m in members], axis=0).round(2).tolist(),
        ))
    patches.sort(key=lambda p: -p.apolar_area)
    for rank, patch in enumerate(patches, start=1):
        patch.rank = rank
    return patches
