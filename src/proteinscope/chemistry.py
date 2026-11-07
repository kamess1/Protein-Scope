"""Ligand chemistry with RDKit: drug-likeness descriptors and 2D depictions.

SMILES come from the wwPDB Chemical Component Dictionary through RCSB, because a crystal
ligand has no hydrogens and no bond orders. RDKit is optional (`pip install "proteinscope[chem]"`);
without it this module reports that it is unavailable and the report omits the section.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .webdata import get_json

CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"


def available() -> bool:
    return importlib.util.find_spec("rdkit") is not None


def component(comp_id: str, cache_dir: Path | None = None) -> dict[str, Any]:
    """Name, formula and SMILES of a chemical component from the CCD."""
    data = get_json(CHEMCOMP_URL.format(comp_id=comp_id.upper()), cache_dir)
    chem = data.get("chem_comp") or {}
    descriptors = data.get("rcsb_chem_comp_descriptor") or {}
    return {
        "comp_id": comp_id.upper(),
        "name": (chem.get("name") or "").replace("\n", " ").strip() or None,
        "formula": chem.get("formula"),
        "smiles": descriptors.get("SMILES_stereo") or descriptors.get("SMILES"),
        "inchikey": descriptors.get("InChIKey"),
    }


def _mol(smiles: str) -> Any:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    return mol


def properties(smiles: str) -> dict[str, Any]:
    """Descriptors used in drug-likeness filters, with Lipinski and Veber checks."""
    from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

    mol = _mol(smiles)
    values = {
        "molecular_weight": round(Descriptors.MolWt(mol), 1),
        "clogp": round(Crippen.MolLogP(mol), 2),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 2),
        "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        "qed": round(QED.qed(mol), 2),
    }
    lipinski = [
        rule for rule, broken in [
            ("MW > 500", values["molecular_weight"] > 500),
            ("cLogP > 5", values["clogp"] > 5),
            ("HBD > 5", values["hbd"] > 5),
            ("HBA > 10", values["hba"] > 10),
        ] if broken
    ]
    veber = [
        rule for rule, broken in [
            ("TPSA > 140", values["tpsa"] > 140),
            ("rotatable bonds > 10", values["rotatable_bonds"] > 10),
        ] if broken
    ]
    values["lipinski_violations"] = lipinski
    values["veber_violations"] = veber
    values["rule_of_three"] = (  # Fragment-likeness (Congreve et al. 2003).
        values["molecular_weight"] <= 300 and values["clogp"] <= 3 and values["hbd"] <= 3 and values["hba"] <= 3
    )
    return values


def depiction_svg(smiles: str, width: int = 320, height: int = 240) -> str:
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = _mol(smiles)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 1.6
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.index("<svg"):]  # Drop the XML prolog so it can be inlined in HTML.


def depiction_png(smiles: str, path: Path, width: int = 600, height: int = 450) -> Path:
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = _mol(smiles)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(drawer.GetDrawingText())
    return path
