"""Fixtures. Offline tests build structures from text; network tests are marked and skipped."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from proteinscope.structure import load_structure

# A hollow sphere of residues with a ligand in the cavity. Hand-built, so the offline tests
# never touch the network while still exercising real geometry: a closed cavity, a buried
# ligand, charged side chains and two shells of atoms.
_ATOMS = {"ASP": ["N", "CA", "CB", "OD1", "OD2"], "LEU": ["N", "CA", "CB", "CG", "CD1"]}
_SHELL_RADIUS = 9.0


def _sphere_points(count: int) -> np.ndarray:
    index = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)
    azimuth = np.pi * (1.0 + 5.0**0.5) * index
    return np.column_stack([
        np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar),
    ])


def _shell_atoms() -> list[tuple[str, str, int, np.ndarray]]:
    atoms = []
    for number, unit in enumerate(_sphere_points(60), start=1):
        name = "ASP" if number % 4 == 0 else "LEU"
        tangent = np.cross(unit, [0.0, 0.0, 1.0])
        norm = np.linalg.norm(tangent)
        tangent = tangent / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        for offset, atom in enumerate(_ATOMS[name]):
            position = unit * (_SHELL_RADIUS + 1.3 * offset) + tangent * (0.7 * (offset % 2))
            atoms.append((name, atom, number, position))
    return atoms


def _pdb_text() -> str:
    lines = []
    serial = 1
    for residue, atom, number, position in _shell_atoms():
        lines.append(
            f"ATOM  {serial:5d}  {atom:<3s} {residue} A{number:4d}    "
            f"{position[0]:8.3f}{position[1]:8.3f}{position[2]:8.3f}  1.00 20.00          {atom[0]:>2s}"
        )
        serial += 1
    for offset, (atom, element) in enumerate([("C1", "C"), ("O1", "O"), ("N1", "N")]):
        position = np.array([1.4 * offset - 1.4, 0.0, 0.0])
        lines.append(
            f"HETATM{serial:5d}  {atom:<3s} LIG A 900    "
            f"{position[0]:8.3f}{position[1]:8.3f}{position[2]:8.3f}  1.00 25.00          {element:>2s}"
        )
        serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="session")
def toy_pdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("structures") / "TOY1.pdb"
    path.write_text(_pdb_text(), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def toy_structure(toy_pdb: Path):
    return load_structure(toy_pdb, "TOY1")


@pytest.fixture(scope="session")
def toy_analysis(toy_pdb: Path):
    from proteinscope.pipeline import run_analysis

    return run_analysis(str(toy_pdb), probe_pockets=1)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("PROTEINSCOPE_NETWORK_TESTS"):
        return
    skip = pytest.mark.skip(reason="set PROTEINSCOPE_NETWORK_TESTS=1 to run tests that use RCSB")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
