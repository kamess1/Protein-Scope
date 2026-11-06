"""Protein-ligand contact profiling and a rigid probe-based interaction simulation.

Contacts come from heavy-atom geometry only. Crystal structures have no hydrogens and
ligand protonation is unknown, so hydrogen bonds and salt bridges are geometric candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .constants import (
    ACCEPTOR_ATOMS,
    AROMATIC_RING_ATOMS,
    DONOR_ATOMS,
    METAL_ELEMENTS,
    vdw_radius,
)
from .electrostatics import ChargeModel, potential_at
from .pockets import Pocket
from .structure import AtomTable, LigandInstance, LoadedStructure, canonical_name

_HBOND_RANGE = (2.4, 3.6)
_SALT_BRIDGE_MAX = 4.0
_HYDROPHOBIC_MAX = 4.5
_METAL_MAX = 2.9
_HALOGEN_MAX = 3.6
_STACK_PARALLEL_MAX = 5.5
_STACK_TSHAPED_MAX = 6.0
_LJ_CUTOFF = 8.0

_ACIDIC = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
_BASIC = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}, "HIS": {"ND1", "NE2"}}
_HALOGENS = {"F", "CL", "BR", "I"}
# Carbons bonded to several polar atoms carry significant partial charge: not hydrophobic.
_POLAR_CARBONS = {"C", "CA", "CG:ASP", "CG:ASN", "CD:GLU", "CD:GLN", "CZ:ARG", "CE1:HIS", "CZ:TYR"}
_SPHERE_POINTS = 200
_WATER_RADIUS = 1.4


def _is_apolar_carbon(atom_name: str, residue: str) -> bool:
    return atom_name not in _POLAR_CARBONS and f"{atom_name}:{residue}" not in _POLAR_CARBONS


def _fibonacci_sphere(count: int) -> np.ndarray:
    index = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)
    azimuth = np.pi * (1.0 + 5.0**0.5) * index
    return np.column_stack([
        np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar),
    ])


def buried_surface_fraction(atoms: AtomTable, protein: AtomTable) -> float:
    """Fraction of the free ligand's solvent-accessible surface that the protein covers."""
    sphere = _fibonacci_sphere(_SPHERE_POINTS)
    radii = np.array([vdw_radius(e) for e in atoms.elements]) + _WATER_RADIUS
    protein_radii = np.array([vdw_radius(e) for e in protein.elements]) + _WATER_RADIUS
    ligand_tree = cKDTree(atoms.coords)
    protein_tree = cKDTree(protein.coords)
    reach = float(protein_radii.max())
    free_area = covered_area = 0.0
    for index, coord in enumerate(atoms.coords):
        points = coord + sphere * radii[index]
        # Remove surface points inside other ligand atoms to get the free ligand surface.
        distances, neighbors = ligand_tree.query(points, k=min(len(atoms), 12))
        distances = distances.reshape(len(points), -1)
        neighbors = neighbors.reshape(len(points), -1)
        inside = (distances < radii[neighbors]) & (neighbors != index)
        accessible = points[~inside.any(axis=1)]
        area_per_point = 4.0 * np.pi * radii[index] ** 2 / _SPHERE_POINTS
        free_area += len(accessible) * area_per_point
        if not len(accessible):
            continue
        for point, candidates in zip(accessible, protein_tree.query_ball_point(accessible, reach)):
            if candidates and np.any(
                np.linalg.norm(protein.coords[candidates] - point, axis=1) < protein_radii[candidates]
            ):
                covered_area += area_per_point
    return covered_area / free_area if free_area else 0.0


@dataclass
class Contact:
    kind: str
    ligand_atom: str
    residue: str
    residue_atom: str
    distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "ligand_atom": self.ligand_atom, "residue": self.residue,
            "residue_atom": self.residue_atom, "distance_A": round(self.distance, 2),
        }


@dataclass
class LigandProfile:
    ligand: str
    atom_count: int
    contacts: list[Contact]
    buried_fraction: float
    residue_vdw: dict[str, float]
    ligand_potential: float
    rings: int

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for contact in self.contacts:
            result[contact.kind] = result.get(contact.kind, 0) + 1
        return result

    def summary(self) -> dict[str, Any]:
        strongest = sorted(self.residue_vdw.items(), key=lambda item: item[1])[:10]
        return {
            "ligand": self.ligand,
            "heavy_atoms": self.atom_count,
            "aromatic_or_planar_rings": self.rings,
            "buried_fraction": round(self.buried_fraction, 3),
            "contact_counts": self.counts(),
            "contacts": [
                contact.to_dict() for contact in self.contacts if contact.kind != "hydrophobic"
            ],
            "hydrophobic_residues": sorted({
                contact.residue for contact in self.contacts if contact.kind == "hydrophobic"
            }),
            "top_residue_vdw_kcal": {name: round(value, 2) for name, value in strongest},
            "mean_potential_at_ligand_kT_e": round(self.ligand_potential, 2),
        }


def is_polar_protein_atom(name: str, residue: str) -> tuple[bool, bool]:
    """Return (can donate, can accept) for a protein heavy atom."""
    residue = canonical_name(residue)
    donor = name == "N" or name in DONOR_ATOMS.get(residue, set())
    acceptor = name == "O" or name == "OXT" or name in ACCEPTOR_ATOMS.get(residue, set())
    if name == "N" and residue == "PRO":  # Proline's backbone nitrogen carries no hydrogen.
        donor = False
    return donor, acceptor


def ligand_bonds(atoms: AtomTable) -> list[tuple[int, int]]:
    """Infer covalent bonds from distances, as PDB coordinates carry no ligand bond orders."""
    bonds = []
    for i, j in cKDTree(atoms.coords).query_pairs(2.2):
        distance = float(np.linalg.norm(atoms.coords[i] - atoms.coords[j]))
        limit = 2.0 if {"S", "P", "CL", "BR", "I", "FE", "ZN", "MG"} & {atoms.elements[i], atoms.elements[j]} else 1.75
        if distance <= limit:
            bonds.append((int(i), int(j)))
    return bonds


def find_planar_rings(atoms: AtomTable, bonds: list[tuple[int, int]]) -> list[np.ndarray]:
    """Find planar 5- and 6-membered rings, which are the stacking candidates in a ligand."""
    neighbors: dict[int, list[int]] = {i: [] for i in range(len(atoms))}
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    rings: set[frozenset[int]] = set()

    def walk(start: int, node: int, path: list[int]) -> None:
        for neighbor in neighbors[node]:
            if neighbor == start and len(path) in (5, 6):
                rings.add(frozenset(path))
            elif neighbor > start and neighbor not in path and len(path) < 6:
                walk(start, neighbor, path + [neighbor])

    for atom in range(len(atoms)):
        walk(atom, atom, [atom])

    planar = []
    for ring in rings:
        points = atoms.coords[list(ring)]
        centered = points - points.mean(axis=0)
        if np.linalg.svd(centered, compute_uv=False)[-1] / np.sqrt(len(ring)) < 0.15:
            planar.append(np.array(sorted(ring)))
    return planar


def _plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    normal = np.linalg.svd(points - center)[2][-1]
    return center, normal


def _stacking(ligand: LigandInstance, rings: list[np.ndarray], protein: AtomTable) -> list[Contact]:
    residue_atoms: dict[Any, dict[str, np.ndarray]] = {}
    for coord, name, key in zip(protein.coords, protein.names, protein.residues):
        ring_names = AROMATIC_RING_ATOMS.get(canonical_name(key.name))
        if ring_names and name in ring_names:
            residue_atoms.setdefault(key, {})[str(name)] = coord

    contacts = []
    for key, positions in residue_atoms.items():
        if len(positions) < 5:
            continue
        center_r, normal_r = _plane(np.array(list(positions.values())))
        best = None
        for ring in rings:
            center_l, normal_l = _plane(ligand.atoms.coords[ring])
            distance = float(np.linalg.norm(center_l - center_r))
            angle = float(np.degrees(np.arccos(min(abs(float(normal_l @ normal_r)), 1.0))))
            if distance <= _STACK_PARALLEL_MAX and angle <= 30.0:
                kind = "pi-stack (parallel)"
            elif distance <= _STACK_TSHAPED_MAX and angle >= 60.0:
                kind = "pi-stack (T-shaped)"
            else:
                continue
            if best is None or distance < best[1]:
                best = (kind, distance)
        if best:
            contacts.append(Contact(best[0], "ring", key.label, "ring", best[1]))
    return contacts


def _lj_energy(distance: np.ndarray, radius_sum: np.ndarray, epsilon: float = 0.15) -> np.ndarray:
    """Soft-capped 12-6 Lennard-Jones energy in kcal/mol, with its minimum at radius_sum."""
    ratio = radius_sum / np.maximum(distance, 0.5)
    ratio6 = ratio**6
    return np.minimum(epsilon * (ratio6**2 - 2.0 * ratio6), 10.0)


def profile_ligand(
    structure: LoadedStructure, ligand: LigandInstance, charge_model: ChargeModel
) -> LigandProfile:
    protein = structure.protein
    atoms = ligand.atoms
    tree = cKDTree(protein.coords)
    contacts: list[Contact] = []

    for index, (coord, element, name) in enumerate(zip(atoms.coords, atoms.elements, atoms.names)):
        element = str(element)
        for neighbor in tree.query_ball_point(coord, 6.0):
            distance = float(np.linalg.norm(coord - protein.coords[neighbor]))
            key = protein.residues[neighbor]
            residue = canonical_name(key.name)
            atom_name = str(protein.names[neighbor])
            atom_element = str(protein.elements[neighbor])
            polar_ligand = element in {"N", "O"}
            can_donate, can_accept = is_polar_protein_atom(atom_name, residue)

            if polar_ligand and (can_donate or can_accept) and _HBOND_RANGE[0] <= distance <= _HBOND_RANGE[1]:
                contacts.append(Contact("hydrogen bond", str(name), key.label, atom_name, distance))
            if distance <= _SALT_BRIDGE_MAX and (
                (element == "O" and atom_name in _BASIC.get(residue, set()))
                or (element == "N" and atom_name in _ACIDIC.get(residue, set()))
            ):
                contacts.append(Contact("salt bridge (candidate)", str(name), key.label, atom_name, distance))
            if (
                element == "C" and atom_element in {"C", "S"}
                and _is_apolar_carbon(atom_name, residue) and distance <= _HYDROPHOBIC_MAX
            ):
                contacts.append(Contact("hydrophobic", str(name), key.label, atom_name, distance))
            if element in _HALOGENS and atom_element in {"N", "O"} and distance <= _HALOGEN_MAX:
                contacts.append(Contact("halogen contact", str(name), key.label, atom_name, distance))
            if element in METAL_ELEMENTS and atom_element in {"N", "O", "S"} and distance <= _METAL_MAX:
                contacts.append(Contact("metal coordination", str(name), key.label, atom_name, distance))

    # Bound metal ions are separate residues, so check coordination to them explicitly.
    for ion in structure.ions:
        if str(ion.atoms.elements[0]) not in METAL_ELEMENTS:
            continue
        for coord, element, name in zip(atoms.coords, atoms.elements, atoms.names):
            distance = float(np.linalg.norm(coord - ion.atoms.coords[0]))
            if str(element) in {"N", "O", "S"} and distance <= _METAL_MAX:
                contacts.append(Contact("metal coordination", str(name), ion.label, str(ion.atoms.names[0]), distance))

    bonds = ligand_bonds(atoms)
    rings = find_planar_rings(atoms, bonds)
    contacts.extend(_stacking(ligand, rings, protein))

    residue_vdw: dict[str, float] = {}
    sparse = cKDTree(atoms.coords).sparse_distance_matrix(tree, _LJ_CUTOFF, output_type="coo_matrix")
    ligand_radii = np.array([vdw_radius(e) for e in atoms.elements])[sparse.row]
    protein_radii = np.array([vdw_radius(e) for e in protein.elements])[sparse.col]
    energies = _lj_energy(np.asarray(sparse.data), ligand_radii + protein_radii)
    for column, energy in zip(sparse.col, energies):
        label = protein.residues[column].label
        residue_vdw[label] = residue_vdw.get(label, 0.0) + float(energy)

    return LigandProfile(
        ligand=ligand.label,
        atom_count=len(atoms),
        contacts=sorted(contacts, key=lambda c: (c.kind, c.distance)),
        buried_fraction=buried_surface_fraction(atoms, protein),
        residue_vdw=residue_vdw,
        ligand_potential=float(potential_at(atoms.coords, charge_model).mean()),
        rings=len(rings),
    )


# name, radius (A), charge (e), H-bond role. Roles are matched to complementary protein atoms.
PROBES = {
    "hydrophobic (C)": {"radius": 1.9, "charge": 0.0, "hbond": None},
    "aromatic (C.ar)": {"radius": 1.8, "charge": 0.0, "hbond": None},
    "H-bond donor (N-H)": {"radius": 1.6, "charge": 0.2, "hbond": "donor"},
    "H-bond acceptor (O)": {"radius": 1.5, "charge": -0.2, "hbond": "acceptor"},
    "cation (N+)": {"radius": 1.6, "charge": 1.0, "hbond": "donor"},
    "anion (O-)": {"radius": 1.5, "charge": -1.0, "hbond": "acceptor"},
}
_HBOND_WELL = -1.5
_HBOND_OPTIMUM = 2.9
_HBOND_WIDTH = 0.4
_KT_TO_KCAL = 0.593


@dataclass
class ProbeResult:
    probe: str
    best_energy: float
    mean_energy: float
    favorable_fraction: float
    hotspot: list[float]
    hotspot_residues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "best_energy_kcal": round(self.best_energy, 2),
            "mean_energy_kcal": round(self.mean_energy, 2),
            "favorable_fraction": round(self.favorable_fraction, 3),
            "hotspot": self.hotspot,
            "hotspot_residues": self.hotspot_residues,
        }


def simulate_probes(
    pocket: Pocket, protein: AtomTable, charge_model: ChargeModel
) -> list[ProbeResult]:
    """Evaluate chemical probes on the pocket grid against a rigid protein.

    Energy = Lennard-Jones + screened Coulomb + a Gaussian H-bond well towards complementary
    protein atoms, and no desolvation. The numbers compare probes and pockets. They are not
    binding free energies.
    """
    points = pocket.points
    tree = cKDTree(protein.coords)
    sparse = cKDTree(points).sparse_distance_matrix(tree, _LJ_CUTOFF, output_type="coo_matrix")
    rows, columns, distances = sparse.row, sparse.col, np.asarray(sparse.data)
    protein_radii = np.array([vdw_radius(e) for e in protein.elements])
    electric = potential_at(points, charge_model) * _KT_TO_KCAL

    polar = [is_polar_protein_atom(str(n), k.name) for n, k in zip(protein.names, protein.residues)]
    can_donate = np.array([p[0] for p in polar])
    can_accept = np.array([p[1] for p in polar])

    results = []
    for name, probe in PROBES.items():
        energies = np.zeros(len(points))
        np.add.at(energies, rows, _lj_energy(distances, probe["radius"] + protein_radii[columns]))
        energies += probe["charge"] * electric
        if probe["hbond"]:
            # A donor probe needs a protein acceptor, and an acceptor probe a protein donor.
            partner = can_accept if probe["hbond"] == "donor" else can_donate
            selected = partner[columns]
            well = _HBOND_WELL * np.exp(-(((distances[selected] - _HBOND_OPTIMUM) / _HBOND_WIDTH) ** 2))
            per_point = np.zeros(len(points))
            np.minimum.at(per_point, rows[selected], well)
            energies += per_point
        best = int(np.argmin(energies))
        spot = points[best]
        residues = sorted({protein.residues[i].label for i in tree.query_ball_point(spot, 4.5)})
        results.append(ProbeResult(
            probe=name,
            best_energy=float(energies[best]),
            mean_energy=float(energies.mean()),
            favorable_fraction=float(np.mean(energies < -1.0)),
            hotspot=spot.round(2).tolist(),
            hotspot_residues=residues,
        ))
    return results


__all__ = [
    "Contact", "LigandProfile", "ProbeResult", "PROBES", "find_planar_rings", "ligand_bonds",
    "profile_ligand", "simulate_probes",
]
