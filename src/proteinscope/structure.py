"""Load structures with BioPython and split them into protein, ligands, ions and water."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from .constants import (
    CRYSTAL_ADDITIVES,
    METAL_ELEMENTS,
    MODIFIED_RESIDUES,
    STANDARD_RESIDUES,
    WATER_NAMES,
)


@dataclass(frozen=True)
class ResidueKey:
    chain: str
    number: int
    insertion: str
    name: str

    @property
    def label(self) -> str:
        return f"{self.name}{self.number}{self.insertion.strip()}:{self.chain}"


@dataclass
class AtomTable:
    """Flat arrays describing a set of heavy atoms; all arrays share an index."""

    coords: np.ndarray
    elements: np.ndarray
    names: np.ndarray
    residues: list[ResidueKey]
    bfactors: np.ndarray
    occupancies: np.ndarray

    def __len__(self) -> int:
        return len(self.coords)


@dataclass
class LigandInstance:
    key: ResidueKey
    atoms: AtomTable

    @property
    def label(self) -> str:
        return self.key.label


@dataclass
class LoadedStructure:
    identifier: str
    path: Path
    biopython: Structure
    protein_residues: list[Residue]
    protein: AtomTable
    ligands: list[LigandInstance]
    additives: list[LigandInstance]
    ions: list[LigandInstance]
    water_count: int
    nucleic_residue_count: int
    header: dict[str, Any] = field(default_factory=dict)

    @property
    def chains(self) -> list[str]:
        return sorted({key.chain for key in self.protein.residues})


def residue_key(residue: Residue) -> ResidueKey:
    _, number, insertion = residue.get_id()
    return ResidueKey(residue.get_parent().get_id(), number, insertion, residue.get_resname().strip())


def is_amino_acid(residue: Residue) -> bool:
    name = residue.get_resname().strip()
    if name in STANDARD_RESIDUES:
        return True
    return name in MODIFIED_RESIDUES and "CA" in residue


def canonical_name(name: str) -> str:
    """Map a modified residue to its parent amino acid; other names stay unchanged."""
    return MODIFIED_RESIDUES.get(name, name)


def _heavy_atoms(residue: Residue) -> list[Atom]:
    atoms = []
    for atom in residue.get_atoms():
        chosen = atom.selected_child if isinstance(atom, DisorderedAtom) else atom
        if (chosen.element or "").upper() in {"H", "D"}:
            continue
        atoms.append(chosen)
    return atoms


def _element(atom: Atom) -> str:
    element = (atom.element or "").strip().upper()
    if element:
        return element
    return atom.get_name().strip()[:1].upper()


def _table(residues: list[Residue]) -> AtomTable:
    coords, elements, names, keys, bfactors, occupancies = [], [], [], [], [], []
    for residue in residues:
        key = residue_key(residue)
        for atom in _heavy_atoms(residue):
            coords.append(atom.get_coord())
            elements.append(_element(atom))
            names.append(atom.get_name().strip())
            keys.append(key)
            bfactors.append(atom.get_bfactor() or 0.0)
            occupancies.append(1.0 if atom.get_occupancy() is None else atom.get_occupancy())
    return AtomTable(
        coords=np.asarray(coords, dtype=float).reshape(-1, 3),
        elements=np.asarray(elements, dtype="U4"),
        names=np.asarray(names, dtype="U6"),
        residues=keys,
        bfactors=np.asarray(bfactors, dtype=float),
        occupancies=np.asarray(occupancies, dtype=float),
    )


def _read_header(path: Path, is_cif: bool) -> dict[str, Any]:
    if not is_cif:
        from Bio.PDB.parse_pdb_header import parse_pdb_header

        raw = parse_pdb_header(str(path))
        return {
            "title": raw.get("name"),
            "method": raw.get("structure_method"),
            "resolution_A": raw.get("resolution"),
            "unobserved_residues": [
                {"chain": m.get("chain"), "number": m.get("ssseq"), "name": m.get("res_name")}
                for m in raw.get("missing_residues", [])
                if m.get("model") in (None, 1, "1", "")
            ],
        }

    data = MMCIF2Dict(str(path))

    def first(tag: str) -> str | None:
        values = data.get(tag)
        if not values or values[0] in ("?", "."):
            return None
        return values[0]

    def as_float(value: str | None) -> float | None:
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    resolution = as_float(first("_refine.ls_d_res_high")) or as_float(
        first("_em_3d_reconstruction.resolution")
    )
    unobserved = []
    numbers = data.get("_pdbx_unobs_or_zero_occ_residues.auth_seq_id", [])
    if numbers:
        chains = data.get("_pdbx_unobs_or_zero_occ_residues.auth_asym_id", [])
        names = data.get("_pdbx_unobs_or_zero_occ_residues.auth_comp_id", [])
        flags = data.get("_pdbx_unobs_or_zero_occ_residues.occupancy_flag", ["1"] * len(numbers))
        models = data.get("_pdbx_unobs_or_zero_occ_residues.PDB_model_num", ["1"] * len(numbers))
        for chain, number, name, flag, model in zip(chains, numbers, names, flags, models):
            if flag == "1" and model == "1":
                unobserved.append({"chain": chain, "number": number, "name": name})
    return {
        "title": first("_struct.title"),
        "method": first("_exptl.method"),
        "resolution_A": resolution,
        "r_free": as_float(first("_refine.ls_R_factor_R_free")),
        "r_work": as_float(first("_refine.ls_R_factor_R_work")),
        "unobserved_residues": unobserved,
    }


def load_structure(path: str | Path, identifier: str | None = None) -> LoadedStructure:
    """Parse the first model of a .pdb/.ent/.cif/.mmcif file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix not in {".pdb", ".ent", ".cif", ".mmcif"}:
        raise ValueError(f"Unsupported structure format: {suffix}")
    is_cif = suffix in {".cif", ".mmcif"}
    identifier = identifier or path.stem.upper()
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    structure = parser.get_structure(identifier, str(path))
    models = list(structure)
    if not models:
        raise ValueError(f"{path} contains no models")
    model = models[0]

    protein, ligands, additives, ions = [], [], [], []
    water_count = 0
    nucleic_count = 0
    nucleotides = {"A", "C", "G", "U", "T", "I", "DA", "DC", "DG", "DT", "DU", "DI", "N"}
    for chain in model:
        for residue in chain:
            hetflag = residue.get_id()[0]
            name = residue.get_resname().strip()
            if hetflag == "W" or name in WATER_NAMES:
                water_count += 1
            elif is_amino_acid(residue):
                protein.append(residue)
            elif name in nucleotides:
                nucleic_count += 1
            elif len(_heavy_atoms(residue)) == 1 and (
                _element(_heavy_atoms(residue)[0]) in METAL_ELEMENTS or name in CRYSTAL_ADDITIVES
            ):
                ions.append(residue)
            elif name in CRYSTAL_ADDITIVES:
                additives.append(residue)
            elif _heavy_atoms(residue):
                ligands.append(residue)

    if not protein:
        raise ValueError(f"{identifier} contains no protein residues to analyse")

    def instances(residues: list[Residue]) -> list[LigandInstance]:
        return [LigandInstance(residue_key(r), _table([r])) for r in residues]

    return LoadedStructure(
        identifier=identifier,
        path=path,
        biopython=structure,
        protein_residues=protein,
        protein=_table(protein),
        ligands=instances(ligands),
        additives=instances(additives),
        ions=instances(ions),
        water_count=water_count,
        nucleic_residue_count=nucleic_count,
        header=_read_header(path, is_cif),
    )
