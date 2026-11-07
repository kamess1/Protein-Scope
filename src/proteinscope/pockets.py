"""Grid-based binding pocket detection in the spirit of LIGSITE.

Every free grid point is scanned along seven lines (three axes, four cube diagonals).
A line counts as a protein-solvent-protein event when protein is hit in both directions
within a limited range. Points with many events are buried in a concavity; connected
groups of such points form pocket candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .constants import KYTE_DOOLITTLE, vdw_radius
from .electrostatics import ChargeModel, potential_at
from .structure import AtomTable, canonical_name

DIRECTIONS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]


@dataclass
class PocketParams:
    grid_spacing: float = 1.0
    probe_radius: float = 0.7
    psp_threshold: int = 4
    ray_length: float = 14.0
    min_volume: float = 60.0
    lining_distance: float = 4.5
    padding: float = 6.0
    max_pockets: int = 12


@dataclass
class Pocket:
    pocket_id: int
    volume: float
    center: np.ndarray
    points: np.ndarray
    enclosure: float
    depth: float
    lining_residues: list[str]
    lining_keys: list[Any]
    apolar_fraction: float
    mean_hydropathy: float
    net_charge: float
    charged_residues: int
    donor_acceptor_atoms: int
    mean_potential: float
    druggability: float = 0.0
    druggability_class: str = "not scored"
    heuristic_score: float = 0.0
    score_model: str = "not scored"
    score_components: dict[str, float] = field(default_factory=dict)
    site_tags: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "pocket_id": self.pocket_id,
            "volume_A3": round(self.volume, 1),
            "center": self.center.round(2).tolist(),
            "enclosure": round(self.enclosure, 3),
            "depth_A": round(self.depth, 2),
            "apolar_fraction": round(self.apolar_fraction, 3),
            "mean_hydropathy": round(self.mean_hydropathy, 2),
            "net_charge": round(self.net_charge, 2),
            "charged_residues": self.charged_residues,
            "polar_atoms": self.donor_acceptor_atoms,
            "mean_potential_kT_e": round(self.mean_potential, 2),
            "druggability": round(self.druggability, 3),
            "druggability_class": self.druggability_class,
            "score_model": self.score_model,
            "heuristic_score": round(self.heuristic_score, 3),
            "score_components": {k: round(v, 3) for k, v in self.score_components.items()},
            "site_tags": self.site_tags,
            "lining_residues": self.lining_residues,
        }


@dataclass
class Grid:
    origin: np.ndarray
    spacing: float
    shape: tuple[int, int, int]

    def to_world(self, indices: np.ndarray) -> np.ndarray:
        return self.origin + np.asarray(indices) * self.spacing


def build_occupancy(coords: np.ndarray, radii: np.ndarray, params: PocketParams) -> tuple[Grid, np.ndarray]:
    origin = coords.min(axis=0) - params.padding
    extent = coords.max(axis=0) + params.padding - origin
    shape = tuple(int(n) for n in np.ceil(extent / params.grid_spacing).astype(int) + 1)
    if int(np.prod(shape)) > 150_000_000:
        raise ValueError(
            f"Grid of {shape} points is too large. Analyse fewer chains or use a coarser spacing."
        )
    grid = Grid(origin, params.grid_spacing, shape)
    occupied = np.zeros(shape, dtype=bool)
    stamps: dict[float, np.ndarray] = {}
    for position, radius in zip(coords, radii):
        total = round(float(radius) + params.probe_radius, 2)
        offsets = stamps.get(total)
        if offsets is None:
            reach = int(np.ceil(total / params.grid_spacing))
            axis = np.arange(-reach, reach + 1)
            mesh = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
            offsets = mesh[(mesh**2).sum(axis=1) * params.grid_spacing**2 <= total**2]
            stamps[total] = offsets
        center = np.round((position - origin) / params.grid_spacing).astype(int)
        hit = offsets + center
        inside = np.all((hit >= 0) & (hit < np.array(shape)), axis=1)
        hit = hit[inside]
        occupied[hit[:, 0], hit[:, 1], hit[:, 2]] = True
    return grid, occupied


def _ray_hits(occupied: np.ndarray, direction: tuple[int, int, int], max_steps: int) -> np.ndarray:
    """For each voxel: does walking along `direction` reach an occupied voxel in max_steps?

    Works plane by plane, propagating the step count from the previous plane.
    """
    # Walking towards +d from a voxel equals sweeping from the far side, so flip until
    # the sweep runs from high to low index along every moving axis.
    flips = [axis for axis, step in enumerate(direction) if step > 0]
    view = np.flip(occupied, axis=flips) if flips else occupied
    moving = [abs(step) for step in direction]
    sweep_axis = moving.index(1)
    view = np.moveaxis(view, sweep_axis, 0)
    other = [moving[axis] for axis in range(3) if axis != sweep_axis]

    limit = np.iinfo(np.int32).max // 2
    steps = np.full(view.shape, limit, dtype=np.int32)
    steps[0][view[0]] = 0
    for i in range(1, view.shape[0]):
        previous = np.full(view.shape[1:], limit, dtype=np.int32)
        previous[other[0]:, other[1]:] = steps[i - 1][
            : view.shape[1] - other[0], : view.shape[2] - other[1]
        ]
        steps[i] = np.where(view[i], 0, np.minimum(previous + 1, limit))
    hits = steps <= max_steps
    hits = np.moveaxis(hits, 0, sweep_axis)
    return np.flip(hits, axis=flips) if flips else hits


def psp_events(occupied: np.ndarray, params: PocketParams) -> np.ndarray:
    events = np.zeros(occupied.shape, dtype=np.uint8)
    for direction in DIRECTIONS:
        length = np.linalg.norm(direction) * params.grid_spacing
        max_steps = int(params.ray_length / length)
        negative = tuple(-step for step in direction)
        events += (
            _ray_hits(occupied, direction, max_steps) & _ray_hits(occupied, negative, max_steps)
        ).astype(np.uint8)
    events[occupied] = 0
    return events


def detect_pockets(
    protein: AtomTable,
    charge_model: ChargeModel,
    params: PocketParams | None = None,
) -> tuple[list[Pocket], Grid]:
    """Find pockets in the protein alone: bound ligands and water are ignored on purpose."""
    params = params or PocketParams()
    radii = np.array([vdw_radius(e) for e in protein.elements])
    grid, occupied = build_occupancy(protein.coords, radii, params)
    events = psp_events(occupied, params)
    mask = events >= params.psp_threshold
    # Opening removes thin bridges so neighbouring cavities do not merge through a crevice.
    mask = ndimage.binary_opening(mask, structure=ndimage.generate_binary_structure(3, 1))
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 2))
    if not count:
        return [], grid

    voxel_volume = params.grid_spacing**3
    sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, count + 1))
    keep = [i + 1 for i, size in enumerate(sizes) if size * voxel_volume >= params.min_volume]
    keep.sort(key=lambda label: -sizes[label - 1])

    tree = cKDTree(protein.coords)
    pockets = []
    for label in keep[: params.max_pockets]:
        indices = np.argwhere(labels == label)
        points = grid.to_world(indices)
        pockets.append(_describe(
            points=points,
            enclosure=float(events[labels == label].mean() / len(DIRECTIONS)),
            depth=_depth(labels == label, params),
            protein=protein,
            tree=tree,
            charge_model=charge_model,
            params=params,
        ))
    return pockets, grid


def _depth(mask: np.ndarray, params: PocketParams) -> float:
    """Distance from the most buried pocket point to the pocket's own boundary."""
    distance = ndimage.distance_transform_edt(np.pad(mask, 1), sampling=params.grid_spacing)
    return float(distance.max())


def _describe(
    points: np.ndarray,
    enclosure: float,
    depth: float,
    protein: AtomTable,
    tree: cKDTree,
    charge_model: ChargeModel,
    params: PocketParams,
) -> Pocket:
    nearby = sorted({
        index
        for row in tree.query_ball_point(points, params.lining_distance)
        for index in row
    })
    keys = []
    seen = set()
    for index in nearby:
        key = protein.residues[index]
        if key not in seen:
            seen.add(key)
            keys.append(key)

    elements = protein.elements[nearby]
    names = protein.names[nearby]
    apolar = np.isin(elements, ["C", "S"])
    # Backbone carbonyl carbon sits between polar atoms, and so does not count as apolar.
    apolar &= names != "C"
    polar_atoms = int(np.isin(elements, ["N", "O"]).sum())

    residue_names = [canonical_name(key.name) for key in keys]
    charge_lookup = {"ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0, "HIS": 0.1}
    return Pocket(
        pocket_id=0,
        volume=len(points) * params.grid_spacing**3,
        center=points.mean(axis=0),
        points=points,
        enclosure=enclosure,
        depth=depth,
        lining_residues=[key.label for key in keys],
        lining_keys=keys,
        apolar_fraction=float(apolar.mean()) if len(nearby) else 0.0,
        mean_hydropathy=float(np.mean([KYTE_DOOLITTLE.get(n, 0.0) for n in residue_names])) if keys else 0.0,
        net_charge=float(sum(charge_lookup.get(n, 0.0) for n in residue_names)),
        charged_residues=sum(1 for n in residue_names if n in {"ASP", "GLU", "LYS", "ARG"}),
        donor_acceptor_atoms=polar_atoms,
        mean_potential=float(potential_at(points, charge_model).mean()),
    )


def ligand_pocket_overlap(ligand_coords: np.ndarray, pocket: Pocket, cutoff: float = 2.5) -> float:
    """Fraction of ligand atoms that lie within `cutoff` of a pocket grid point."""
    distances, _ = cKDTree(pocket.points).query(ligand_coords)
    return float(np.mean(distances <= cutoff))
