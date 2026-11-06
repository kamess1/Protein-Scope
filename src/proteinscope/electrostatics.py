"""Screened-Coulomb estimate of the electrostatic potential around a protein.

This is a fast approximation, not a Poisson-Boltzmann solution: charges sit only on
titratable groups, the dielectric is distance dependent and ionic screening uses a
Debye-Huckel term. It is good for the sign and rough size of the potential near the
surface. For publication-grade potentials use APBS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .constants import ION_CHARGES, SIDECHAIN_CHARGES
from .structure import LoadedStructure, canonical_name

COULOMB_KT = 560.0  # e^2 / (4 pi eps0 A) at 298 K expressed in kT, i.e. 332 / 0.593.
DEBYE_LENGTH = 7.85  # Angstrom at 150 mM ionic strength.
MAX_DIELECTRIC = 80.0
CUTOFF = 30.0


@dataclass
class ChargeModel:
    coords: np.ndarray
    charges: np.ndarray

    @property
    def net_charge(self) -> float:
        return float(self.charges.sum())


def build_charge_model(structure: LoadedStructure) -> ChargeModel:
    """Assign pH 7 charges to side chains and chain termini, plus bound metal ions."""
    table = structure.protein
    charges = np.zeros(len(table))
    for index, (key, atom) in enumerate(zip(table.residues, table.names)):
        charges[index] = SIDECHAIN_CHARGES.get(canonical_name(key.name), {}).get(atom, 0.0)

    # Only true termini get a charge: a chain's first and last observed residues may be
    # next to unobserved segments, in which case they are not termini at all.
    observed_gaps = {str(m.get("chain")) for m in structure.header.get("unobserved_residues", [])}
    by_chain: dict[str, list[int]] = {}
    for index, key in enumerate(table.residues):
        by_chain.setdefault(key.chain, []).append(index)
    for chain, indices in by_chain.items():
        if chain in observed_gaps:
            continue
        first_key, last_key = table.residues[indices[0]], table.residues[indices[-1]]
        for index in indices:
            if table.residues[index] == first_key and table.names[index] == "N":
                charges[index] += 1.0
        terminal_oxygens = [
            i for i in indices if table.residues[i] == last_key and table.names[i] in {"O", "OXT"}
        ]
        for index in terminal_oxygens:
            charges[index] += -1.0 / len(terminal_oxygens)

    coords = [table.coords]
    values = [charges]
    for ion in structure.ions:
        element = str(ion.atoms.elements[0])
        if element in ION_CHARGES:
            coords.append(ion.atoms.coords)
            values.append(np.array([ION_CHARGES[element]]))
    all_coords, all_charges = np.vstack(coords), np.concatenate(values)
    mask = all_charges != 0
    return ChargeModel(all_coords[mask], all_charges[mask])


def potential_at(points: np.ndarray, model: ChargeModel, cutoff: float = CUTOFF) -> np.ndarray:
    """Potential in kT/e at each point: sum of q exp(-r/lambda) / (eps(r) r), eps(r) = 4r."""
    points = np.asarray(points, dtype=float)
    potential = np.zeros(len(points))
    if not len(points) or not len(model.coords):
        return potential
    sparse = cKDTree(points).sparse_distance_matrix(
        cKDTree(model.coords), cutoff, output_type="coo_matrix"
    )
    distance = np.maximum(sparse.data, 1.0)  # The matrix omits exact zero distances.
    dielectric = np.minimum(4.0 * distance, MAX_DIELECTRIC)
    contribution = (
        COULOMB_KT * model.charges[sparse.col] * np.exp(-distance / DEBYE_LENGTH) / (dielectric * distance)
    )
    np.add.at(potential, sparse.row, contribution)
    return potential
